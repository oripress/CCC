import torch
import torch.nn.functional as F

from models import register, AdaptiveModel
from models.functional import configure_model, collect_params


@register("rpl")
class RPL(AdaptiveModel):
    def __init__(self, model, lr=0.0005, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        model = configure_model(model)
        params, param_names = collect_params(model)
        self.model = model
        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)

    def forward(self, x):
        outputs = self.model(x)
        labels = outputs.argmax(dim=1)
        loss = gce(outputs, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs


def gce(logits, target, q=0.8):
    """Generalized cross entropy.

    Reference: https://arxiv.org/abs/1805.07836
    """
    probs = F.softmax(logits, dim=1)
    probs_with_correct_idx = probs.index_select(-1, target).diag()
    loss = (1.0 - probs_with_correct_idx ** q) / q
    return loss.mean()
