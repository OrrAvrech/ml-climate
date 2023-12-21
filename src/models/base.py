import torch
from torch import nn
from typing import Optional
from transformers import AutoImageProcessor, AutoModel
from scale_mae import mae_vit_large_patch16


class BasePreTrained(nn.Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        self.name = name
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.name)
        self.backbone = AutoModel.from_pretrained(self.name).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(x, return_tensors="pt")
        inputs = inputs.to(self.device)
        x = self.backbone(**inputs)
        x = x.pooler_output
        embedding = x.view(x.size(0), -1)
        return embedding


class ViTNonEpisodic(BasePreTrained):
    def __init__(self, name: str, device: torch.device, num_classes: Optional[int] = None):
        super().__init__(name=name, device=device)
        self.num_classes = num_classes
        self.classifier = nn.Linear(768, num_classes).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.processor(x, return_tensors="pt")
        inputs = inputs.to(self.device)
        x = self.backbone(**inputs)
        x = x.pooler_output
        embedding = x.view(x.size(0), -1)
        logit = self.classifier(embedding)
        return embedding, logit


class ScaleMAE(nn.Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        self.weights_path = name
        self.device = device
        model = mae_vit_large_patch16(fixed_output_size=0, independent_fcn_head=True, fcn_dim=512,
                                      fcn_layers=2, decoder_depth=3, absolute_scale=True)
        state_dict = torch.load(self.weights_path)
        model.load_state_dict(state_dict["model"])
        self.model = model.to(device)

    def forward(self, x: torch.Tensor, eval_scale: int = 256, eval_res: float = 1.0) -> torch.Tensor:
        data = x.to(self.device)
        data = data.to(torch.float32)
        data = torch.nn.functional.interpolate(
            data, (eval_scale, eval_scale), mode="area"
        )
        input_res = torch.ones(len(data)).float().to(data.device) * eval_res
        embeddings = self.model(data, knn_feats=True, input_res=input_res)
        return embeddings
