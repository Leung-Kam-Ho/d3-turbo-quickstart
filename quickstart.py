from bench import DEVICE, timed
from huggingface_hub import hf_hub_download
from transformers import AutoModel

print(f"[device] {DEVICE}", flush=True)


with timed("model load"):
    model = AutoModel.from_pretrained(
        "vllm-sr/d3-turbo", trust_remote_code=True, device=DEVICE
    )


def asset(name):
    return hf_hub_download("vllm-sr/d3-turbo", f"assets/{name}")


ASSETS = [
    "example-frisbee.jpg",
    "example-panda.jpg",
    "example-option-a.jpg",
    "example-option-b.jpg",
    "example-option-c.jpg",
    "example-hummingbird.mp4",
]

# Pre-fetch assets so per-example timings measure inference only.
with timed("asset download"):
    for name in ASSETS:
        asset(name)


# Text: several questions, one call
with timed("text"):
    print(model.system_one(
        state="Ticket: the invoice total is wrong after the plan change.",
        questions={
            "route": {"type": "choice", "instructions": "Which team handles this?",
                      "criteria": {"billing": "Billing", "tech": "Technical support"}},
            "urgent": {"type": "noul", "instructions": "Is the customer asking for an urgent fix?"},
        },
    ))

# Box: where is it in the image? Four questions, one read of the photo
with timed("box (image)"):
    print(model.system_one(
        state="A photo from a dog festival.",
        images=[asset("example-frisbee.jpg")],  # local paths, http(s) URLs, PIL images or base64-encoded images
        questions={
            "frisbee": {"type": "box", "instructions": "The frisbee"},
            "tail": {"type": "box", "instructions": "The dog's tail"},
            "cone": {"type": "box", "instructions": "The orange cone"},
            "camera": {"type": "box", "instructions": "The camera on the tripod"},
        },
    ))

# Images in options: each option is an image that only this question sees
with timed("images in options"):
    print(model.system_one(
        state="This animal: <animal>",
        questions={"pick": {"type": "choice", "instructions": "Which picture shows the same kind of animal?",
                            "criteria": {"a": "<a>", "b": "<b>", "c": "<c>"}}},
        assets={"<animal>": asset("example-panda.jpg"), "<a>": asset("example-option-a.jpg"),
                "<b>": asset("example-option-b.jpg"), "<c>": asset("example-option-c.jpg")},
    ))

# Video: a track over the whole clip, two boxes per second
with timed("video (track)"):
    print(model.system_one(
        state="A bird at a feeder.",
        videos=[asset("example-hummingbird.mp4")],  # local paths, http(s) URLs, base64-encoded videos or frame arrays
        questions={"target": {"type": "track", "instructions": "The hummingbird", "rate": 2}},
    ))
