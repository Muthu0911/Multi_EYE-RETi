# Retinal screening API

This service uses the upstream repository's actual `best_efficientnet_b0.pth`
checkpoint. It is an EfficientNet-B0 with a five-output sigmoid multilabel head.
The default preprocessing is resize to 224x224 followed by ImageNet
normalization, matching the upstream demo.

Configure the checkpoint without committing it:

```text
HF_MODEL_REPO=your-account/retinal-efficientnet
HF_MODEL_FILENAME=best_efficientnet_b0.pth
HF_TOKEN=...                  # only for a private Hugging Face repository
```

`MODEL_PATH` can point to a local checkpoint for development. `MODEL_URL` is
also supported for an externally hosted public file. The process loads the
model once at startup, and `/predict` performs real inference in memory.

