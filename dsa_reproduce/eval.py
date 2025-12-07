"""
本脚本直接运行即可对比模型训练前后的吞吐（tokens/s）与时延（单 batch 耗时），无需额外传参。
默认使用 config.yaml 中的路径：基座模型 vs 联合训练后的模型，取前 max_samples 条（默认 32）评估。
运行：python eval.py
"""

import time
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import SFTDataset
from load_config import CONFIG
from model import Qwen2ForCausalLM


@dataclass
class BenchConfig:
    before_model_path: str = CONFIG["base_model_path"]
    after_model_path: str = CONFIG["train_model_path"]
    tokenizer_path: str = CONFIG["tokenizer_path"]
    data_path: str = CONFIG["train_data_path"]
    max_samples: int = 32
    max_seq_len: int = 1024
    batch_size: int = 1
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    warmup_steps: int = 1  # 每个模型预热多少个 batch（不计时）


@dataclass
class BenchResult:
    latency_s: float
    throughput_tps: float
    avg_loss: float | None
    steps: int


def build_loader(tokenizer, cfg: BenchConfig) -> DataLoader:
    dataset = SFTDataset(cfg.data_path, tokenizer=tokenizer, max_seq_len=cfg.max_seq_len)
    if cfg.max_samples > 0:
        max_n = min(cfg.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_n)))
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)


def bench_model(model: Qwen2ForCausalLM, loader: DataLoader, device: str, warmup_steps: int) -> BenchResult:
    model.to(device)
    model.eval()

    latencies = []
    total_tokens = 0
    total_loss = 0.0
    steps = 0

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 预热：跑若干 batch，不计时，避免首次编译/缓存影响
    if warmup_steps > 0:
        with torch.no_grad():
            warmup_iter = iter(loader)
            for _ in range(warmup_steps):
                try:
                    batch = next(warmup_iter)
                except StopIteration:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                _ = model(**batch)
                if device.startswith("cuda"):
                    torch.cuda.synchronize()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            start = time.time()
            outputs = model(**batch)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - start

            latencies.append(elapsed)
            total_tokens += batch["input_ids"].numel()
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            steps += 1

    total_time = sum(latencies)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    avg_latency = total_time / steps if steps > 0 else 0.0
    avg_loss = total_loss / steps if steps > 0 else None

    return BenchResult(latency_s=avg_latency, throughput_tps=throughput, avg_loss=avg_loss, steps=steps)


def main():
    cfg = BenchConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, padding_side="right")
    loader = build_loader(tokenizer, cfg)

    print(f"评估数据条数: {len(loader.dataset)}，batch_size={cfg.batch_size}，max_seq_len={cfg.max_seq_len}")
    print(f"device: {cfg.device}")

    model_before = AutoModelForCausalLM.from_pretrained(cfg.before_model_path)
    #model_before = Qwen2ForCausalLM.from_pretrained(cfg.before_model_path)
    before_metrics = bench_model(model_before, loader, cfg.device, cfg.warmup_steps)
    print("[训练前] ", asdict(before_metrics))

    model_after = Qwen2ForCausalLM.from_pretrained(cfg.after_model_path)
    after_metrics = bench_model(model_after, loader, cfg.device, cfg.warmup_steps)
    print("[训练后] ", asdict(after_metrics))


if __name__ == "__main__":
    main()

