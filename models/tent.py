# original code copied from: https://github.com/DequanWang/tent
import torch
from models import register, AdaptiveModel
from models.functional import configure_model, collect_params, softmax_entropy


@register("tent")
class Tent(AdaptiveModel):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, lr=0.00025, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        model = configure_model(model)
        params, param_names = collect_params(model)
        self.model = model
        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)

    def forward(self, x):
        outputs = self.model(x)
        # adapt
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs
