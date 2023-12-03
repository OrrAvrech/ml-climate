import torch
from transformers import AutoImageProcessor, AutoModel


class BasePreTrained(torch.nn.Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        self.name = name
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(x, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.pooler_output
        return embedding
