# decision-3.0-playground

Quickstart for [`vllm-sr/d3-turbo`](https://huggingface.co/vllm-sr/d3-turbo), a 26B multimodal decision model. The script auto-selects the device (CUDA, Apple MPS, or CPU) and prints the time used by each example.

## Usage

```bash
uv run quickstart.py
```

On Linux, `flash-linear-attention` is installed automatically through the `sys_platform == 'linux'` marker in `pyproject.toml`.

## Timings

Apple Silicon (MPS), bf16, 26B model:

| Step | Time |
|---|---:|
| Model load | 49.30 s |
| Asset download | 1.54 s |
| Text (2 questions) | 72.21 s |
| Box (4 questions, 1 image) | 27.34 s |
| Images in options (3 images) | 19.32 s |
| Video track (11 s clip, 2 boxes/s) | 222.74 s |
| **Total inference** | **341.61 s** |

Notes:

- `text` is the first inference call, so it includes one-time MPS kernel compilation.
- The video example produces 22 boxes over an 11-second clip.
- macOS logs fallbacks for `causal_conv1d` and `flash-linear-attention`; Linux + CUDA uses the optimized kernels.
- Full run output is in [`run.log`](run.log).
