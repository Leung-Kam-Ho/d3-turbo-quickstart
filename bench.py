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


def synchronize():
    kind = torch.device(DEVICE).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


@contextmanager
def timed(label):
    synchronize()
    t0 = time.perf_counter()
    yield
    synchronize()
    print(f"[time] {label}: {time.perf_counter() - t0:.2f}s", flush=True)
