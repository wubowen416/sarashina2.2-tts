import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("WANDB_PROJECT", "sarashina-tts-disfluency")

import pandas as pd
import torch
import tyro
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from sarashina_tts.additional_tokens import SPEECH_START_TOKEN
from sarashina_tts.generate.generate import _semantic_ids_to_str
from sarashina_tts.text_frontend.text_preprocess import preprocess_text

# Codec assets that live next to the LLM weights but aren't touched by LLM
# fine-tuning. Copied into each checkpoint dir so it stays directly loadable
# by SarashinaTTSGenerator.
CODEC_ASSET_FILES = ["flow.pt", "hift.pt", "campplus_cn_common.bin"]


@dataclass
class TrainConfig:
    model_dir: Path = Path("pretrained_models")
    data_root: Path = Path("/home/wu/workspace/disfluency_tts")
    train_csv: Path = Path(
        "data/splits/train.csv"
    )  # a .csv file or a dir of .csv files, relative to data_root
    wav_dirname: str = "processed_wav"
    token_dirname: str = "processed_wav_tokens"
    cache_path: Path = Path("data/cache/train_llm_cache.pkl")  # relative to data_root
    run_dir: Path = Path("runs")
    run_name: str = "sarashina-tts-disfluency"
    resume_from_checkpoint: Optional[Path] = None
    # Sequence length cutoff. Examples whose tokenized length exceeds this are
    # dropped (not truncated -- truncating audio tokens would cut off the EOS
    # the model needs to learn to stop generating). Defaults to the model's
    # max_position_embeddings.
    max_length: Optional[int] = None
    # Training
    num_epochs: int = 2
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    val_ratio: float = 0.05
    gradient_checkpointing: bool = True
    save_steps: int = 1000
    eval_steps: int = 1000
    cleanup_keep_steps: int = 10000  # delete checkpoints not divisible by this


# --- Dataset ---


class LLMDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input_ids": list(row["input_ids"]),
            "labels": list(row["labels"]),
        }


def build_example(
    text: str,
    audio_path: str,
    tokenizer,
    data_root: Path,
    wav_dirname: str,
    token_dirname: str,
    ignore_id: int = -100,
    max_length: Optional[int] = None,
) -> Optional[dict]:

    # Get prompt str
    try:
        text = preprocess_text(text)

    except TypeError as e:
        print(f"Error preprocessing text: {text}. Error: {e}")
        return None

    prompt_str = f"{text}{SPEECH_START_TOKEN}"
    # print(f"Prompt: {prompt_str}")

    # Get audio tokens
    audio_token_path = data_root / Path(
        str(audio_path).replace(wav_dirname, token_dirname)
    ).with_suffix(".json")
    # print(f"Audio token path: {audio_token_path}")
    try:
        with open(audio_token_path, "r") as f:
            audio_tokens = json.load(f)
    except FileNotFoundError:
        return None
    audio_str = _semantic_ids_to_str(audio_tokens)

    prompt_ids = tokenizer(prompt_str, add_special_tokens=True).input_ids
    audio_ids = tokenizer(audio_str, add_special_tokens=False).input_ids
    input_ids = prompt_ids + audio_ids + [tokenizer.eos_token_id]

    if max_length is not None and len(input_ids) > max_length:
        return None

    prompt_len = len(prompt_ids)
    labels = [ignore_id] * prompt_len + input_ids[prompt_len:]

    return {"input_ids": input_ids, "labels": labels}


def load_dataset(cfg: TrainConfig, tokenizer, max_length: int) -> LLMDataset:
    cache_path = cfg.data_root / cfg.cache_path
    if cache_path.exists():
        print(f"Loading dataset from cache: {cache_path}")
        df = pd.read_pickle(cache_path)
    else:
        print("Processing dataset...")
        train_csv = cfg.data_root / cfg.train_csv
        df = pd.read_csv(train_csv)

        tqdm.pandas(desc="Tokenizing")
        processed = df.progress_apply(
            lambda row: build_example(
                row["tagged_text"],
                row["audio_path"],
                tokenizer,
                cfg.data_root,
                cfg.wav_dirname,
                cfg.token_dirname,
                max_length=max_length,
            ),
            axis=1,
        )

        num_dropped = int(processed.isna().sum())
        if num_dropped:
            print(
                f"Dropped {num_dropped}/{len(df)} examples (missing tokens or over max_length={max_length})"
            )
        keep = processed.notna()
        df = df[keep].copy()
        df["input_ids"] = processed[keep].apply(lambda x: x["input_ids"])
        df["labels"] = processed[keep].apply(lambda x: x["labels"])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache_path)
        print(f"Dataset cached to: {cache_path}")

    return LLMDataset(df)


def load_model(cfg: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(cfg.model_dir)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def copy_codec_assets(model_dir: Path, dst_dir: Path) -> None:
    if not dst_dir.exists():
        return
    for filename in CODEC_ASSET_FILES:
        src = model_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)


class CheckpointCleanupCallback(TrainerCallback):
    def __init__(self, keep_steps: int, model_dir: Path):
        self.keep_steps = keep_steps
        self.model_dir = model_dir

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_root = Path(args.output_dir)
        ckpt_dir = ckpt_root / f"checkpoint-{state.global_step}"

        # Make the checkpoint directly loadable by SarashinaTTSGenerator.
        copy_codec_assets(self.model_dir, ckpt_dir)

        # Delete non-keep_steps checkpoints when we hit a keep_steps boundary
        if state.global_step % self.keep_steps != 0:
            return
        for ckpt in ckpt_root.glob("checkpoint-*"):
            try:
                step = int(ckpt.name.removeprefix("checkpoint-"))
                if step % self.keep_steps != 0:
                    shutil.rmtree(ckpt)
                    print(f"Deleted checkpoint: {ckpt.name}")
            except ValueError:
                pass


def main(cfg: TrainConfig) -> None:
    if cfg.resume_from_checkpoint is not None:
        run_dir = cfg.resume_from_checkpoint.parent
        wandb_run_name = f"{run_dir.parent.name}/{run_dir.name}"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = cfg.run_dir / cfg.run_name / timestamp
        wandb_run_name = f"{cfg.run_name}/{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|pad|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    model = load_model(cfg)

    max_length = (
        cfg.max_length
        or AutoConfig.from_pretrained(cfg.model_dir).max_position_embeddings
    )

    dataset = load_dataset(cfg, tokenizer, max_length)
    print(f"Dataset size: {len(dataset):,}")

    val_size = int(cfg.val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if cfg.gradient_checkpointing else {}
        ),
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2500,
        bf16=True,
        logging_steps=10,
        run_name=wandb_run_name,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        processing_class=tokenizer,
        callbacks=[
            CheckpointCleanupCallback(
                keep_steps=cfg.cleanup_keep_steps, model_dir=cfg.model_dir
            )
        ],
    )

    trainer.train(
        resume_from_checkpoint=(
            str(cfg.resume_from_checkpoint) if cfg.resume_from_checkpoint else None
        )
    )

    # --- Save final model ---
    best_dir = run_dir / "checkpoint-best"
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    copy_codec_assets(cfg.model_dir, best_dir)
    print(f"Saved final model to {best_dir}")


if __name__ == "__main__":
    tyro.cli(main)
