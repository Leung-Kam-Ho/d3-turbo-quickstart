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

### d3-turbo

bf16, 26B model.

#### Apple Silicon (M3 Ultra, MPS)

| Step | Time |
|---|---:|
| Model load | 49.30 s |
| Asset download | 1.54 s |
| Text (2 questions) | 72.21 s |
| Box (4 questions, 1 image) | 27.34 s |
| Images in options (3 images) | 19.32 s |
| Video track (11 s clip, 2 boxes/s) | 222.74 s |
| **Total inference** | **341.61 s** |

#### NVIDIA DGX Spark (CUDA)

| Step | Mean (3 runs) | Fastest |
|---|---:|---:|
| Model load (once) | 310.03 s | — |
| Asset download (once) | 1.44 s | — |
| Text (2 questions) | 1.13 s | 0.55 s |
| Box (4 questions, 1 image) | 8.87 s | 8.79 s |
| Images in options (3 images) | 6.56 s | 6.54 s |
| Video track (11 s clip, 2 boxes/s) | 71.77 s | 71.67 s |
| **Total inference** | **88.33 s** | — |

### d3-flash

#### Apple Silicon (M3 Ultra, MPS)

| Step | Mean (3 runs) | Fastest |
|---|---:|---:|
| Model load (once) | 4.55 s | — |
| Asset download (once) | 0.45 s | — |
| Text | 0.45 s | 0.41 s |
| Text + image | 4.08 s | 4.05 s |
| Text + video | 2.54 s | 2.23 s |
| **Total inference** | **7.07 s** | — |

#### NVIDIA DGX Spark (CUDA)

| Step | Mean (3 runs) | Fastest |
|---|---:|---:|
| Model load (once) | 109.74 s | — |
| Asset download (once) | 0.48 s | — |
| Text | 0.55 s | 0.15 s |
| Text + image | 1.76 s | 1.72 s |
| Text + video | 0.97 s | 0.94 s |
| **Total inference** | **3.28 s** | — |

Notes:

- The Apple Silicon results were measured on an M3 Ultra.
- The d3-turbo Apple Silicon table was captured with one run per example. The other tables show the mean and fastest of 3 runs.
- `text` is the first inference call, so it includes one-time kernel compilation/warmup (2.30 s for d3-turbo and 1.35 s for d3-flash on their first DGX Spark run; 0.54 s for d3-flash on Apple Silicon).
- Model load on DGX Spark includes first-time CUDA initialization and loading the checkpoint.
- The d3-turbo video example produces 22 boxes over an 11-second clip.
- macOS logs fallbacks for `causal_conv1d` and `flash-linear-attention`; Linux + CUDA uses the optimized kernels (`causal_conv1d` still falls back on the DGX Spark runs).
- The Apple Silicon run output is in [`run.log`](run.log).
