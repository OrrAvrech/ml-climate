import torch
from transformers import ViTFeatureExtractor, AutoImageProcessor


class ViTPreTrained(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.processor = AutoImageProcessor.from_pretrained(self.name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(x, return_tensors="pt")
        model = ViTFeatureExtractor.from_pretrained(self.name)
        outputs = model(**inputs)
        return outputs
