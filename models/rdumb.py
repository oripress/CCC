# original code copied from https://github.com/mr-eggplant/EATA
"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight.
"""
import math

import torch
import torch.jit
import torch.nn.functional as F

from models import AdaptiveModel, functional, register


@register("rdumb")
class RDumb(AdaptiveModel):
    """Our model is ETA with Resetting
    """
    def __init__(self, model):
        super().__init__(model)
        self.num_samples_update_1 = (
            0  # number of samples after First filtering, exclude unreliable samples
        )
        self.num_samples_update_2 = 0  # number of samples after Second filtering, exclude both unreliable and redundant samples
        self.e_margin = math.log(1000) / 2 - 1  # hyper-parameter E_0 (Eqn. 3)
        self.d_margin = (
            0.05  # hyper-parameter \epsilon for consine simlarity thresholding (Eqn. 5)
        )

        self.current_model_probs = (
            None  # the moving average of probability vector (Eqn. 4)
        )

        params, param_names = functional.collect_params(model)
        model = functional.configure_model(model)

        self.model = model
        self.optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)

        self.model_state, self.optimizer_state = functional.copy_model_and_optimizer(
            self.model, self.optimizer
        )
        self.total_steps = 0

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward(
        self,
        x,
    ):
        if self.total_steps % 1000 == 0:
            functional.load_model_and_optimizer(
                self.model, self.optimizer, self.model_state, self.optimizer_state
            )
            self.current_model_probs = None

        # forward
        outputs = self.model(x)
        # adapt
        entropys = functional.softmax_entropy(outputs)
        # filter unreliable samples
        filter_ids_1 = torch.where(entropys < self.e_margin)
        ids1 = filter_ids_1
        ids2 = torch.where(ids1[0] > -0.1)
        entropys = entropys[filter_ids_1]
        self.ent = entropys.size(0)
        # filter redundant samples
        if self.current_model_probs is not None:
            cosine_similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(dim=0),
                outputs[filter_ids_1].softmax(1).detach(),
                dim=1,
            )
            filter_ids_2 = torch.where(torch.abs(cosine_similarities) < self.d_margin)
            self.div = filter_ids_2[0].size(0)
            entropys = entropys[filter_ids_2]
            ids2 = filter_ids_2
            updated_probs = self.update_model_probs(
                self.current_model_probs,
                outputs[filter_ids_1][filter_ids_2].softmax(1).detach(),
            )
        else:
            updated_probs = self.update_model_probs(
                self.current_model_probs, outputs[filter_ids_1].softmax(1)
            )
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)

        if x[ids1][ids2].size(0) != 0:
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        self.num_samples_update_2 += entropys.size(0)
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.current_model_probs = updated_probs
        self.total_steps += 1

        return outputs

    def update_model_probs(self, current_model_probs, new_probs):
        if current_model_probs is None:
            if new_probs.size(0) == 0:
                return None
            else:
                with torch.no_grad():
                    return new_probs.mean(0)
        else:
            if new_probs.size(0) == 0:
                with torch.no_grad():
                    return current_model_probs
            else:
                with torch.no_grad():
                    return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
