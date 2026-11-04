# d3-turbo-quickstart

Quickstart for [`vllm-sr/d3-turbo`](https://huggingface.co/vllm-sr/d3-turbo), a 26B multimodal decision model. The script auto-selects the device (CUDA, Apple MPS, or CPU) and prints the time used by each example.

## Usage

```bash
uv run quickstart.py        # d3-turbo
uv run quickstart-flash.py  # d3-flash
```

Both scripts share the device selection and timing helpers in `bench.py`.

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

| Step | Time |
|---|---:|
| Model load | 339.85 s |
| Asset download | 1.48 s |
| Text (2 questions) | 23.72 s |
| Box (4 questions, 1 image) | 19.11 s |
| Images in options (3 images) | 13.33 s |
| Video track (11 s clip, 2 boxes/s) | 79.57 s |
| **Total inference** | **135.73 s** |

Notes:

- `text` is the first inference call, so it includes one-time kernel compilation/warmup.
- Model load on DGX Spark includes first-time CUDA initialization and loading the 52 GB checkpoint.
- The video example produces 22 boxes over an 11-second clip.
- macOS logs fallbacks for `causal_conv1d` and `flash-linear-attention`; Linux + CUDA uses the optimized kernels (`causal_conv1d` still falls back on the DGX Spark run).
- The Apple Silicon run output is in [`run.log`](run.log).
