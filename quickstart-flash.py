import json

from bench import DEVICE, timed
from huggingface_hub import hf_hub_download
from transformers import AutoModel

print(f"[device] {DEVICE}", flush=True)

with timed("model load"):
    model = AutoModel.from_pretrained(
        "vllm-sr/d3-flash", trust_remote_code=True, device=DEVICE
    )


def asset(name):
    return hf_hub_download("vllm-sr/d3-flash", f"assets/{name}")


ASSETS = ["example-receipt.png", "example-video.mp4"]

# Pre-fetch assets so per-example timings measure inference only.
with timed("asset download"):
    for name in ASSETS:
        asset(name)


# Text
with timed("text"):
    result = model.system_one(
        state="The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
        questions={
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "returns": "Refunds, replacements and damaged deliveries",
                    "billing": "Payments, invoices and charges",
                    "technical": "Product setup and faults"
                }
            },
            "receipt": {
                "type": "noul",
                "instructions": "Does the customer have a receipt?"
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this request?",
                "criteria": [
                    "Routine",
                    "Soon",
                    "Today"
                ]
            }
        },
    )
print(json.dumps(result["answers"], indent=2))

# Text and an image (or several)
with timed("text + image"):
    result = model.system_one(
        state="The customer says the blender arrived cracked and attached the receipt.",
        images=[asset("example-receipt.png")],  # local paths, http(s) URLs, PIL images or base64 data URLs
        questions={
            "route": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "returns": "Refunds, replacements and damaged deliveries",
                    "billing": "Payments, invoices and charges",
                    "technical": "Product setup and faults"
                }
            },
            "on_receipt": {
                "type": "noul",
                "instructions": "Does the receipt list the blender?"
            },
            "payment": {
                "type": "choice",
                "instructions": "How was the order paid?",
                "criteria": {
                    "card": None,
                    "cash": None,
                    "gift card": None
                }
            }
        },
    )
print(json.dumps(result["answers"], indent=2))

# Text and a video (or several)
with timed("text + video"):
    result = model.system_one(
        state="A short clip from a test camera.",
        videos=[asset("example-video.mp4")],  # local paths, http(s) URLs, base64 data URLs or frame arrays
        questions={
            "direction": {
                "type": "choice",
                "instructions": "Which way does the square move?",
                "criteria": {
                    "right": "From left to right",
                    "left": "From right to left",
                    "still": "It does not move"
                }
            },
            "color_change": {
                "type": "noul",
                "instructions": "Does the square change color?"
            }
        },
    )
print(json.dumps(result["answers"], indent=2))

# Or as a pipeline:
# transformers.pipeline("decision", model="vllm-sr/d3-flash", trust_remote_code=True)(state=..., questions=..., images=..., videos=...)
