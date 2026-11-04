# d3-turbo-quickstart

Quickstarts for [`vllm-sr/d3-turbo`](https://huggingface.co/vllm-sr/d3-turbo) (26B) and [`vllm-sr/d3-flash`](https://huggingface.co/vllm-sr/d3-flash). The scripts auto-select the device (CUDA, Apple MPS, or CPU) and time each example.

## Usage

```bash
uv run quickstart.py        # d3-turbo
uv run quickstart-flash.py  # d3-flash
```

Both scripts share the device selection and timing helpers in `bench.py`. Each example runs 3 times by default and reports the mean and fastest run; set `BENCH_REPEATS` to change that (for example, `BENCH_REPEATS=1`).

On Linux, `flash-linear-attention` is installed automatically through the `sys_platform == 'linux'` marker in `pyproject.toml`.

## Timings

bf16, 26B model.

### Apple Silicon (MPS)

| Step | Time |
|---|---:|
| Model load | 49.30 s |
| Asset download | 1.54 s |
| Text (2 questions) | 72.21 s |
| Box (4 questions, 1 image) | 27.34 s |
| Images in options (3 images) | 19.32 s |
| Video track (11 s clip, 2 boxes/s) | 222.74 s |
| **Total inference** | **341.61 s** |

### NVIDIA DGX Spark (CUDA)

| Step | Mean (3 runs) | Fastest |
|---|---:|---:|
| Model load (once) | 310.03 s | — |
| Asset download (once) | 1.44 s | — |
| Text (2 questions) | 1.13 s | 0.55 s |
| Box (4 questions, 1 image) | 8.87 s | 8.79 s |
| Images in options (3 images) | 6.56 s | 6.54 s |
| Video track (11 s clip, 2 boxes/s) | 71.77 s | 71.67 s |
| **Total inference** | **88.33 s** | — |

Notes:

- The Apple Silicon table was captured with one run per example. The DGX Spark table shows the mean and fastest of 3 runs.
- `text` is the first inference call, so it includes one-time kernel compilation/warmup (2.30 s on the first DGX Spark run).
- Model load on DGX Spark includes first-time CUDA initialization and loading the 52 GB checkpoint.
- The video example produces 22 boxes over an 11-second clip.
- macOS logs fallbacks for `causal_conv1d` and `flash-linear-attention`; Linux + CUDA uses the optimized kernels (`causal_conv1d` still falls back on the DGX Spark run).
- The Apple Silicon run output is in [`run.log`](run.log).
