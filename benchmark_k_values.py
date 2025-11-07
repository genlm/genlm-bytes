import argparse
import asyncio
import time

import matplotlib.pyplot as plt

from datasets import load_dataset
from genlm.backend import load_model_by_name
from genlm.backend.tokenization.bytes import get_byte_vocab
from genlm.bytes import ByteBeamState, BeamParams, advance_byte_states
from genlm.bytes.byte_lm.trie_state import LazyTrieState, TrieMode
from genlm.bytes.trie import AsyncTokenByteTrie


async def benchmark_k_value(
    llm,
    text_bytes: bytes,
    K: int,
    verbose: bool = False,
    *,
    batch_size: int = 1,
    trie_max_batch: int | None = None,
):
    """Benchmark beam search with a specific K value.

    Supports sentence-level batching by advancing multiple beams in lockstep.

    Returns:
        tuple: (average_negative_log_prob, chars_per_second, success)
    """

    batch_size = max(1, batch_size)
    text_batch = [text_bytes for _ in range(batch_size)]

    params = BeamParams(
        K=K,
        verbose=verbose,
        eos_tokens=[llm.byte_vocab[llm.tokenizer.eos_token_id]],
        heal=True,
    )

    max_batch = trie_max_batch if trie_max_batch is not None else max(64, batch_size)
    async_trie = AsyncTokenByteTrie.from_vocab(
        get_byte_vocab(llm.tokenizer),
        eos_tokens=params.eos_tokens,
        max_batch_size=max_batch,
    )

    states: list[ByteBeamState] = []

    try:
        for _ in range(batch_size):
            lazy_state = LazyTrieState.initial(llm, async_trie, mode=TrieMode.WITH_EOS)
            states.append(ByteBeamState([await lazy_state.materialize()], params))

        positions = [0] * batch_size
        total_chars = sum(len(bs) for bs in text_batch)
        start_time = time.perf_counter()

        while True:
            active_indices = [
                i for i, bs in enumerate(text_batch) if positions[i] < len(bs)
            ]
            if not active_indices:
                break

            step_states = [states[i] for i in active_indices]
            step_bytes = [text_batch[i][positions[i]] for i in active_indices]

            advanced_states = await advance_byte_states(step_states, step_bytes)

            for offset, idx in enumerate(active_indices):
                next_state = advanced_states[offset]
                if len(next_state) == 0:
                    return (None, None, False)
                states[idx] = next_state
                positions[idx] += 1

        elapsed_time = time.perf_counter() - start_time

        total_neg_log_prob = -sum(state.logZ for state in states)
        avg_neg_log_prob = total_neg_log_prob / batch_size
        chars_per_second = total_chars / elapsed_time if elapsed_time > 0 else 0

        return (avg_neg_log_prob, chars_per_second, True)

    except Exception as e:
        if verbose:
            print(f"Error with K={K}: {e}")
        return (None, None, False)
    finally:
        if states:
            await asyncio.gather(*(state.cleanup() for state in states))
        else:
            await async_trie.cleanup()


async def run_benchmark(
    text: str,
    K_values: list[int],
    batch_size: int = 1,
    verbose: bool = False,
    trie_max_batch: int | None = None,
):
    """Run benchmark for multiple K values."""
    print("Loading model...")
    llm = load_model_by_name("meta-llama/Llama-3.2-1B")
    text_bytes = text.encode("utf-8")
    
    results = []
    
    for K in K_values:
        print(f"Benchmarking K={K} (batch={batch_size})...")
        neg_log_prob, chars_per_sec, success = await benchmark_k_value(
            llm,
            text_bytes,
            K,
            verbose=verbose,
            batch_size=batch_size,
            trie_max_batch=trie_max_batch,
        )
        
        if success:
            results.append({
                'K': K,
                'negative_log_prob': neg_log_prob,
                'chars_per_second': chars_per_sec,
                'batch_size': batch_size,
                'total_chars': len(text_bytes) * batch_size,
            })
            print(
                f"  K={K}: -logP(avg)={neg_log_prob:.2f}, "
                f"speed={chars_per_sec:.1f} chars/s (batch={batch_size})"
            )
        else:
            print(f"  K={K}: FAILED")
    
    return results


def plot_results(results: list[dict], output_file: str = "benchmark_k_values.png"):
    """Plot negative log probability vs speed for different K values."""
    if not results:
        print("No results to plot")
        return
    
    K_vals = [r['K'] for r in results]
    neg_log_probs = [r['negative_log_prob'] for r in results]
    speeds = [r['chars_per_second'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with K values as colors
    scatter = ax.scatter(
        speeds,
        neg_log_probs,
        c=K_vals,
        s=100,
        alpha=0.7,
        cmap='viridis',
        edgecolors='black',
        linewidths=1.5,
    )
    
    # Add labels for each point
    for i, k in enumerate(K_vals):
        ax.annotate(
            f'K={k}',
            (speeds[i], neg_log_probs[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )
    
    ax.set_xlabel('Speed (chars/second)', fontsize=12)
    ax.set_ylabel('Negative Log Probability', fontsize=12)
    ax.set_title('Beam Search Performance: Speed vs Negative Log Probability\n(Wikitext2, first 1000 chars)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Beam Width (K)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark beam search K values")
    parser.add_argument(
        "--chars",
        type=int,
        default=1000,
        help="Number of leading characters from Wikitext2 test split to benchmark",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of sentences to decode concurrently",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose beam logging",
    )
    parser.add_argument(
        "--trie-max-batch",
        type=int,
        default=None,
        help="Override trie max_batch_size (default: max(64, batch_size))",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip rendering the matplotlib plot",
    )

    args = parser.parse_args()

    batch_size = max(1, args.batch_size)

    print("Loading Wikitext2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    text = ""
    for entry in dataset:
        if entry["text"].strip():
            text += entry["text"]
            if len(text) >= args.chars:
                break

    text = text[: args.chars]
    print(f"Using first {len(text)} characters from Wikitext2")
    print(f"Preview: {text[:100]}...")

    # K_values = [1, 2, 4, 8, 16, 32]
    K_values = [16]

    results = asyncio.run(
        run_benchmark(
            text,
            K_values,
            batch_size=batch_size,
            verbose=args.verbose,
            trie_max_batch=args.trie_max_batch,
        )
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'K':<5} {'-logP(avg)':<15} {'chars/s':<15} {'batch':<6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['K']:<5} {r['negative_log_prob']:<15.2f} "
            f"{r['chars_per_second']:<15.1f} {r['batch_size']:<6}"
        )

    if batch_size > 1 and results:
        total_chars = results[0]['total_chars']
        print(
            f"\nchars/s reflects total throughput across {batch_size} sentences "
            f"({total_chars} chars per run)."
        )

    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()

