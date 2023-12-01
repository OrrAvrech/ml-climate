import torch
from transformers import ViTModel, AutoImageProcessor


class ViTPreTrained(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.processor = AutoImageProcessor.from_pretrained(self.name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(x, return_tensors="pt")
        model = ViTModel.from_pretrained(self.name)
        outputs = model(**inputs)
        embedding = outputs.pooler_output
        return embedding
