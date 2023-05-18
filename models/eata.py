# original code copied from https://github.com/mr-eggplant/EATA
"""
Copyright to EATA ICML 2022 Authors, 2022.03.20
Based on Tent ICLR 2021 Spotlight.
"""
import math

import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from models import AdaptiveModel, register


@register("eata")
@register("eta")
class EATA(AdaptiveModel):
    """EATA adapts a model by entropy minimization during testing.
    Once EATAed, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, loader, eta=False):
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

        if eta == False:
            self.fisher_alpha = 2000.0  # trade-off \beta for two losses (Eqn. 8)
            self.fishers = self.get_fisher_vectors(
                model, 2000, loader
            )  # fisher regularizer items for anti-forgetting, need to be calculated pre model adaptation (Eqn. 9)
        else:
            self.fisher_alpha = 0
            self.fishers = None

        params, param_names = collect_params(model)
        model = configure_model(model)
        self.model = model
        self.optimizer = torch.optim.SGD(params, 0.00025, momentum=0.9)

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward(
        self,
        x,
    ):
        # forward
        outputs = self.model(x)
        # adapt
        entropys = softmax_entropy(outputs)
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
        # implementation version 1, compute loss, all samples backward (some unselected are masked)
        entropys = entropys.mul(coeff)  # reweight entropy losses for diff. samples
        loss = entropys.mean(0)
        """
        # implementation version 2, compute loss, forward all batch, forward and backward selected samples again.
        # if x[ids1][ids2].size(0) != 0:
        #     loss = softmax_entropy(model(x[ids1][ids2])).mul(coeff).mean(0) # reweight entropy losses for diff. samples
        """
        if self.fishers is not None:
            ewc_loss = 0
            for name, param in self.model.named_parameters():
                if name in self.fishers:
                    ewc_loss += (
                        self.fisher_alpha
                        * (
                            self.fishers[name][0] * (param - self.fishers[name][1]) ** 2
                        ).sum()
                    )
            loss += ewc_loss
        if x[ids1][ids2].size(0) != 0:
            self.backward_step = x[ids1][ids2].size(0)
            loss.backward()
            self.optimizer.step()
        else:
            self.backward_step = 0
        self.optimizer.zero_grad(set_to_none=True)

        self.num_samples_update_2 += entropys.size(0)
        self.num_samples_update_1 += filter_ids_1[0].size(0)
        self.current_model_probs = updated_probs

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

    def get_fisher_vectors(self, model, fisher_amount, loader):
        model = configure_model(model)
        params, param_names = collect_params(model)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(loader, start=1):
            images = images.cuda(non_blocking=True)
            outputs = model(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = (
                            param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        )
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ * 64 >= fisher_amount:
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
                    break
            ewc_optimizer.zero_grad()
        del ewc_optimizer
        del loader

        return fishers


from models.functional import collect_params, configure_model, softmax_entropy
