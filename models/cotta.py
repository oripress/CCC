from copy import deepcopy

import torch
import torch.jit

import PIL
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
import torchvision.transforms.functional as F
from numpy import random

from models import register, AdaptiveModel
from models.functional import configure_model, collect_params


@register("cotta")
class CoTTA(AdaptiveModel):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, model, lr=0.01, momentum=0.9):
        super().__init__(model)
        self.lr = lr
        self.momentum = momentum
        model = configure_model(model)
        params, param_names = collect_params(model)
        self.model = model

        optimizer = torch.optim.SGD(
            params, lr=self.lr, momentum=self.momentum, nesterov=True
        )
        self.optimizer = optimizer

        (
            self.model_state,
            self.optimizer_state,
            self.model_ema,
            self.model_anchor,
        ) = self.copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms()

    def forward(self, x):
        outputs = self.model(x)
        self.model.cpu()
        self.model_ema.train()
        # Teacher Prediction
        self.model_anchor.cuda()
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(x), dim=1).max(1)[0]
        self.model_anchor.cpu()
        self.model_ema.cuda()
        standard_ema = self.model_ema(x)
        # Augmentation-averaged Prediction
        N = 32
        outputs_emas = []
        to_aug = anchor_prob.mean(0) < 0.1
        if to_aug:
            for i in range(N):
                outputs_ = self.model_ema(self.transform(x)).detach()
                outputs_emas.append(outputs_)
        # Threshold choice discussed in supplementary
        self.model_ema.cpu()
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema
        # Augmentation-averaged Prediction
        # Student update
        self.model.cuda()
        loss = (softmax_entropy(outputs, outputs_ema.detach())).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # Teacher update
        self.model_ema.cuda()
        self.model_ema = update_ema_variables(
            ema_model=self.model_ema, model=self.model, alpha_teacher=0.999
        )
        self.model_ema.cpu()
        # Stochastic restore
        if True:
            for nm, m in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ["weight", "bias"] and p.requires_grad:
                        mask = (torch.rand(p.shape) < 0.001).float().cuda()
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (
                                1.0 - mask
                            )

        return outputs_ema

    def copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        model_anchor = deepcopy(model)
        optimizer_state = deepcopy(optimizer.state_dict())
        ema_model = deepcopy(model)
        for param in ema_model.parameters():
            param.detach_()
        return model_state, optimizer_state, ema_model, model_anchor


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + "(min_val={0}, max_val={1})".format(
            self.min_val, self.max_val
        )


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor))
            )

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor))
            )

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor))
            )

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = (
                    torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                )
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = (
                    torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                )
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = (
                    torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                )
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(
                    1e-8, 1.0
                )  # to fix Nan values in gradients, which happens when applying gamma
                # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "brightness={0}".format(self.brightness)
        format_string += ", contrast={0}".format(self.contrast)
        format_string += ", saturation={0}".format(self.saturation)
        format_string += ", hue={0})".format(self.hue)
        format_string += ", gamma={0})".format(self.gamma)
        return format_string


def get_tta_transforms(gaussian_std: float = 0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose(
        [
            Clip(0.0, 1.0),
            ColorJitterPro(
                brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                gamma=[0.85, 1.15] if soft else [0.7, 1.3],
            ),
            transforms.Pad(padding=int(n_pixels / 2), padding_mode="edge"),
            transforms.RandomAffine(
                degrees=[-8, 8] if soft else [-15, 15],
                translate=(1 / 16, 1 / 16),
                scale=(0.95, 1.05) if soft else (0.9, 1.1),
                shear=None,
                interpolation=PIL.Image.BILINEAR,
                fill=None,
            ),
            transforms.GaussianBlur(
                kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]
            ),
            transforms.CenterCrop(size=n_pixels),
            transforms.RandomHorizontalFlip(p=p_hflip),
            GaussianNoise(0, gaussian_std),
            Clip(clip_min, clip_max),
        ]
    )
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):  # , iteration):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = (
            alpha_teacher * ema_param[:].data[:]
            + (1 - alpha_teacher) * param[:].data[:]
        )
    return ema_model


def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)