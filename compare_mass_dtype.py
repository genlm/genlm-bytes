import argparse
import asyncio
import time

import torch
from datasets import load_dataset

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams, ByteBeamState


def load_wikitext_chars(num_chars: int) -> str:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = ""
    for entry in dataset:
        if entry["text"].strip():
            text += entry["text"]
            if len(text) >= num_chars:
                break
    return text[:num_chars]


def torch_dtype_from_arg(arg: str | None) -> torch.dtype | None:
    if arg is None:
        return None
    if hasattr(torch, arg):
        return getattr(torch, arg)
    raise argparse.ArgumentTypeError(f"Unknown torch dtype: {arg}")


async def run_decode(llm, text: str, dtype: torch.dtype | None, k: int) -> dict:
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    params = BeamParams(
        K=k,
        eos_tokens=[eos_token],
        heal=True,
        mass_dtype=dtype or torch.float32,
    )

    state = await ByteBeamState.initial(llm, params)
    try:
        text_bytes = text.encode("utf-8")
        start = time.perf_counter()
        for byte in text_bytes:
            state = await (state.prune() << byte)
            if len(state) == 0:
                raise RuntimeError("Beam emptied during decoding")
        elapsed = time.perf_counter() - start
        chars_per_second = len(text_bytes) / elapsed if elapsed > 0 else 0.0
        return {
            "elapsed": elapsed,
            "chars_per_second": chars_per_second,
            "log_prob": float(state.logZ),
        }
    finally:
        await state.cleanup()


async def main(args):
    llm = load_model_by_name(args.model_name, backend=args.backend)
    sample_text = load_wikitext_chars(1000)

    dtype_candidates = [("float32", torch.float32)]
    if hasattr(torch, "float16"):
        dtype_candidates.append(("float16", torch.float16))
    if hasattr(torch, "float8_e4m3fn"):
        dtype_candidates.append(("float8_e4m3fn", torch.float8_e4m3fn))
    if hasattr(torch, "float8_e5m2"):
        dtype_candidates.append(("float8_e5m2", torch.float8_e5m2))

    results = {}
    for label, dtype in dtype_candidates:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = await run_decode(llm, sample_text, dtype, args.beam_k)
        if torch.cuda.is_available():
            stats["peak_memory"] = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            stats["peak_memory"] = None
        results[label] = stats

    print("\nComparison (beam K = {}):".format(args.beam_k))
    for label, stats in results.items():
        print(f"  dtype={label}")
        print(f"    chars/s: {stats['chars_per_second']:.2f}")
        print(f"    elapsed: {stats['elapsed']:.2f}s")
        if stats["peak_memory"] is not None:
            print(f"    peak GPU memory: {stats['peak_memory']:.1f} MB")
        else:
            print("    peak GPU memory: n/a")
        print(f"    log_prob: {stats['log_prob']:.4f}")
    if "float16" in results:
        diff = results["float16"]["log_prob"] - results["float32"]["log_prob"]
        print(f"\nlog_prob difference (float16 - float32): {diff:.6f}")
    if "float8_e4m3fn" in results:
        diff = results["float8_e4m3fn"]["log_prob"] - results["float32"]["log_prob"]
        print(f"log_prob difference (float8_e4m3fn - float32): {diff:.6f}")
    if "float8_e5m2" in results:
        diff = results["float8_e5m2"]["log_prob"] - results["float32"]["log_prob"]
        print(f"log_prob difference (float8_e5m2 - float32): {diff:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare mass dtype configurations.")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B", help="Model to load (default: gpt2)")
    parser.add_argument("--backend", default="vllm", help="Backend to use (default: hf)")
    parser.add_argument("--beam-k", type=int, default=8, help="Beam width")
    asyncio.run(main(parser.parse_args()))
