import torch
from models import register, AdaptiveModel

@register("pretrained")
class Pretrained(AdaptiveModel):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)
        return outputs
