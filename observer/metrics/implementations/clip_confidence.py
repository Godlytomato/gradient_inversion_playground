"""Correct-label CLIP softmax confidence on raw reconstructed pixels.

The batch score is the minimum confidence if every top-1 prediction matches
its ground-truth label, and zero otherwise. This is a semantic recovery metric;
the uncalibrated softmax score is not a probability guarantee.
"""
from functools import lru_cache
import math

import torch


def validate_clip_parameters(parameters):
    temperature = parameters.get("temperature", 1.0)
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("CLIP temperature must be finite and positive")
    template = parameters.get("prompt_template", "a photo of a {label}")
    if not isinstance(template, str) or "{label}" not in template:
        raise ValueError("CLIP prompt_template must contain {label}")
    try:
        template.format(label="example")
    except (KeyError, ValueError, IndexError, AttributeError) as error:
        raise ValueError("Invalid CLIP prompt_template") from error
    names = parameters.get("class_names")
    if names is not None and (
        not isinstance(names, (list, tuple)) or len(names) < 2
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("CLIP class_names must contain at least two unique, nonempty names")
    if {"labels", "data_def", "reconstruction", "ground_truth"} & parameters.keys():
        raise ValueError("CLIP images, labels and dataset definition come from runtime information")


class ClipClassifier:
    def __init__(self, model_name, device, download_root):
        try:
            import clip
        except ImportError as error:
            raise ImportError(
                "CLIP metrics require OpenAI CLIP; install requirements-clip.txt"
            ) from error
        if not hasattr(clip, "load") or not hasattr(clip, "tokenize"):
            raise ImportError("Install OpenAI CLIP using requirements-clip.txt (the 'clip' package is unrelated)")
        self.device = torch.device(device)
        # Loading a pretrained evaluator must not change attack random streams.
        devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                   if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            self.model, self.preprocess = clip.load(
                model_name, device=self.device, jit=False, download_root=download_root
            )
        self.model.eval().requires_grad_(False)
        self.tokenize = clip.tokenize
        self.text_features = {}

    @torch.inference_mode()
    def probabilities(self, images, class_names, prompt_template, temperature):
        from torchvision.transforms.functional import to_pil_image

        prompts = tuple(prompt_template.format(label=name) for name in class_names)
        if prompts not in self.text_features:
            features = self.model.encode_text(self.tokenize(prompts).to(self.device)).float()
            self.text_features[prompts] = torch.nn.functional.normalize(features, dim=-1)
        pixels = images.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
        if pixels.shape[1] == 1:
            pixels = pixels.expand(-1, 3, -1, -1)
        inputs = torch.stack([self.preprocess(to_pil_image(image)) for image in pixels]).to(self.device)
        features = torch.nn.functional.normalize(self.model.encode_image(inputs).float(), dim=-1)
        # Use the checkpoint's learned scale, not an arbitrary cosine multiplier.
        logits = self.model.logit_scale.exp().float() * features @ self.text_features[prompts].T
        return (logits / temperature).softmax(dim=-1)


@lru_cache(maxsize=2)
def load_classifier(model_name, device, download_root):
    return ClipClassifier(model_name, device, download_root)


def clip_confidence(
    reconstruction, ground_truth=None, *, labels, data_def,
    model_name="ViT-B/32", temperature=1.0,
    prompt_template="a photo of a {label}", class_names=None,
    device=None, download_root="checkpoint/clip",
) -> float:
    validate_clip_parameters(dict(temperature=temperature, prompt_template=prompt_template,
                                  class_names=class_names))
    names = tuple(class_names if class_names is not None else (data_def.class_names or ()))
    if len(names) != data_def.class_count or len(names) < 2:
        raise ValueError("CLIP requires one class name per dataset label, in label-index order")
    if (not isinstance(reconstruction, torch.Tensor) or reconstruction.ndim != 4
            or reconstruction.shape[0] < 1 or reconstruction.shape[1] not in {1, 3}):
        raise ValueError("CLIP requires a nonempty NCHW image batch with 1 or 3 channels")
    if not torch.isfinite(reconstruction).all():
        raise ValueError("CLIP requires finite reconstructed pixels")
    labels = torch.as_tensor(labels)
    if (labels.shape != (len(reconstruction),) or labels.dtype == torch.bool
            or labels.is_floating_point() or labels.is_complex()
            or (labels < 0).any() or (labels >= len(names)).any()):
        raise ValueError("CLIP requires one valid integer ground-truth label per image")
    classifier = load_classifier(model_name, str(device or reconstruction.device), download_root)
    probabilities = classifier.probabilities(reconstruction, names, prompt_template, temperature)
    if not torch.isfinite(probabilities).all():
        raise ValueError("CLIP produced nonfinite probabilities")
    confidence, predicted = probabilities.max(dim=-1)
    if not torch.equal(predicted, labels.to(device=predicted.device, dtype=torch.long)):
        return 0.0
    return float(confidence.min().item())
