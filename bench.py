"""Shared device selection and timing helpers for the quickstart scripts."""

import os
import time
from contextlib import contextmanager

# Let unsupported ops fall back to CPU on Apple GPUs instead of crashing.
if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

if torch.cuda.is_available():
    # An explicit index is required: torch.cuda.set_device(torch.device("cuda")) raises.
    DEVICE = f"cuda:{torch.cuda.current_device()}"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# How many times each example runs in bench().
REPEATS = int(os.environ.get("BENCH_REPEATS", "3"))


def synchronize():
    kind = torch.device(DEVICE).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


@contextmanager
def timed(label):
    """Time a one-off block once, such as model loading or asset downloads."""
    synchronize()
    t0 = time.perf_counter()
    yield
    synchronize()
    print(f"[time] {label}: {time.perf_counter() - t0:.2f}s", flush=True)


def _elapsed(fn):
    synchronize()
    t0 = time.perf_counter()
    result = fn()
    synchronize()
    return result, time.perf_counter() - t0


def bench(label, fn, repeats=None):
    """Run fn several times, then report the mean and the fastest run."""
    repeats = REPEATS if repeats is None else repeats
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    times = []
    result = None
    for i in range(repeats):
        result, elapsed = _elapsed(fn)
        times.append(elapsed)
        print(f"[time] {label} run {i + 1}/{repeats}: {elapsed:.2f}s", flush=True)

    print(
        f"[time] {label}: mean {sum(times) / len(times):.2f}s, "
        f"fastest {min(times):.2f}s ({repeats} runs)",
        flush=True,
    )
    return result
