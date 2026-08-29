import argparse
import glob
import os

import torch
import torchvision.models as models
import torchvision.transforms as trn
import webdataset as wds

from ccc_config import baseline_name, dataset_name, stream_config
from models import registery


def identity(x):
    return x


def get_webds_loader(dset_path):
    shards = sorted(glob.glob(os.path.join(dset_path, "serial_*.tar")))
    if not shards:
        raise FileNotFoundError("No CCC shards found in {}".format(dset_path))

    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preproc = trn.Compose(
        [
            trn.ToTensor(),
            normalize,
        ]
    )
    dataset = (
        wds.WebDataset(shards, shardshuffle=False)
        .decode("pil")
        .to_tuple("input.jpg", "output.cls")
        .map_tuple(preproc, identity)
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=64)
    return dataloader


def test(model, dset_path, file_name=None):
    total_seen_so_far = 0
    dataset_loader = get_webds_loader(dset_path)

    for i, (images, labels) in enumerate(dataset_loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        output = model(images)

        num_images_in_batch = images.size(0)
        total_seen_so_far += num_images_in_batch

        vals, pred = (output).max(dim=1, keepdim=True)
        correct_this_batch = pred.eq(labels.view_as(pred)).sum().item()

        with open(file_name, "a+") as f:
            f.write(
                ("acc_{:.10f}\n").format(
                    float(100 * correct_this_batch) / images.size(0)
                )
            )
        if total_seen_so_far >= 7500000:
            return


def evaluate(args):
    torch.manual_seed(42)
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise ValueError("CUDA not available")
    device = torch.device("cuda")
    baseline = baseline_name(args.baseline)
    exp_name = "ccc_{}".format(baseline)

    os.makedirs(os.path.join(args.logs, exp_name), exist_ok=True)

    speed, cur_seed = stream_config(args.processind)

    file_name = os.path.join(
        args.logs,
        exp_name,
        "model_{}_baseline_{}_transition+speed_{}_seed_{}.txt".format(
            str(args.mode), baseline, str(speed), str(cur_seed)
        ),
    )

    dset_name = dataset_name(args.baseline, speed, cur_seed)
    dset_path = os.path.join(args.dset, dset_name)

    model = models.resnet50(pretrained=True)
    model.to(device)
    model = torch.nn.parallel.DataParallel(model)

    assert args.mode in registery.get_options()
    if args.mode == "eta" or args.mode == "eata":
        loader = get_webds_loader(dset_path)
        model = registery.init(args.mode, model, loader, args.mode == "eta")
    else:
        model = registery.init(args.mode, model)

    test(model, dset_path, file_name=file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="tent", choices=registery.get_options()
    )
    parser.add_argument("--processind", type=int, choices=range(9), default=0)
    parser.add_argument("--baseline", type=float, choices=(0, 20, 40), default=20)
    parser.add_argument("--logs", type=str, required=True)
    parser.add_argument("--dset", type=str, required=True)
    args = parser.parse_args()

    evaluate(args)
