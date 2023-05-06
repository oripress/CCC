import torch
import torch.jit
import torch.nn.functional as F

from models import register, AdaptiveModel
from models.functional import configure_model, collect_params


@register("conjugate")
class ConjugateLoss(AdaptiveModel):
    def __init__(self, model, lr=0.001, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        model = configure_model(model)
        params, param_names = collect_params(model)
        self.model = model
        self.optimizer = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)

    def forward(self, x):
        eps = 8.0
        outputs = self.model(x)
        softmax_prob = F.softmax(outputs, dim=1)
        smax_inp = softmax_prob

        eye = torch.eye(1000).to(outputs.device)
        eye = eye.reshape((1, 1000, 1000))
        eye = eye.repeat(outputs.shape[0], 1, 1)
        t2 = eps * torch.diag_embed(smax_inp)
        smax_inp = torch.unsqueeze(smax_inp, 2)
        t3 = eps * torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
        matrix = eye + t2 - t3
        y_star = torch.linalg.solve(matrix, smax_inp)
        y_star = torch.squeeze(y_star)

        pseudo_prob = y_star
        loss = torch.logsumexp(outputs, dim=1) - (
            pseudo_prob * outputs - eps * pseudo_prob * (1 - softmax_prob)
        ).sum(dim=1)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return outputs
