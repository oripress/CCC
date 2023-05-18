import argparse
import io
import os
import os.path
import pickle
import random
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
import torch.utils.data as data
import torchvision.transforms as trn
import webdataset as wds
from PIL import Image

from make_imagenet_c import noise_dict

IMGS_PER = 25000


def path_to_dataset(path, root):
    dir_list = []

    for i in range(len(path)):
        dir_list.append(
            os.path.join(
                root,
                "s1_" + str(float(path[i][0]) / 4) + "s2_" + str(float(path[i][1]) / 4),
            )
        )

    return dir_list


def collect(log_path):
    sevs = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.5,
        4.75,
        5.0,
    ]

    results = np.zeros((len(sevs), len(sevs)))
    total = 0

    with open(log_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.split("_")
        if len(line) < 5:  # not sure why this happens, but sometimes lines are empty
            continue
        s1 = line[1][:-2]
        s2 = line[2]
        acc = line[4]

        s1 = sevs.index(float(s1))
        s2 = sevs.index(float(s2))

        results[s1][s2] = float(acc)
        total += 1

    return results


def find_path(arr, target_val):
    cur_max = 99999999999
    cost_dict = {}
    path_dict = {}
    for i in range(1, arr.shape[0]):
        cost_dict, path_dict = traverse_graph(
            cost_dict, path_dict, arr, i, 0, target_val
        )

    for i in range(1, arr.shape[0]):
        cur_cost = abs(cost_dict[(i, 0)] / len(path_dict[(i, 0)]) - target_val)
        if cur_cost < cur_max:
            cur_max = cur_cost
            cur_path = path_dict[(i, 0)]

    return cur_path


def traverse_graph(cost_dict, path_dict, arr, i, j, target_val):
    if j >= arr.shape[1]:
        if (i, j) not in cost_dict.keys():
            cost_dict[(i, j)] = 9999999999999
            path_dict[(i, j)] = [9999999999999]
        return cost_dict, path_dict

    if i == 0:
        if (i, j) not in cost_dict.keys():
            cost_dict[(i, j)] = arr[i][j]
            path_dict[(i, j)] = [(i, j)]
        return cost_dict, path_dict

    if (i - 1, j) not in cost_dict.keys():
        cost_dict, path_dict = traverse_graph(
            cost_dict, path_dict, arr, i - 1, j, target_val
        )
    if (i, j + 1) not in cost_dict.keys():
        cost_dict, path_dict = traverse_graph(
            cost_dict, path_dict, arr, i, j + 1, target_val
        )

    if abs(
        ((cost_dict[(i - 1, j)] + arr[i][j]) / (len(path_dict[i - 1, j]) + 1))
        - target_val
    ) < abs(
        ((cost_dict[(i, j + 1)] + arr[i][j]) / (len(path_dict[i, j + 1]) + 1))
        - target_val
    ):
        cost_dict[(i, j)] = cost_dict[(i - 1, j)] + arr[i][j]
        path_dict[(i, j)] = [(i, j)] + path_dict[(i - 1, j)]
    else:
        cost_dict[(i, j)] = cost_dict[(i, j + 1)] + arr[i][j]
        path_dict[(i, j)] = [(i, j)] + path_dict[(i, j + 1)]

    return cost_dict, path_dict


def write_dataset(dataset, root):
    sink = wds.TarWriter(os.path.join(root + ".tar"))

    for index in range(len(dataset)):
        input, output, info = dataset[index]
        sink.write(
            {
                "__key__": "sample_" + str(index),
                "info": info,
                "input.jpg": input,
                "output.cls": output,
            }
        )

    sink.close()


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def GenerateDataset(
    data_root,
    destination_folder,
    speed,
    seed,
    baseline,
    serial_ind=1,
    totalprocesses=1,
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cutoff = 7500000  # total amount of images to be generated
    os.makedirs(destination_folder, exist_ok=True)
    final_path = os.path.join(
        destination_folder,
        "baseline_{}_transition+speed_{}_seed_{}".format(
            str(baseline), str(speed), str(seed)
        ),
    )
    os.makedirs(final_path, exist_ok=True)

    get_frost_images("./")
    walk_dict = {}

    # The baselines used for CCC-[Easy, Medium, Hard] respectively. Note that this is the baseline accuracy on the calibration set. On cropped data, this accuracy goes down a bit
    # For each baseline accuracy, we use noises such that there isn't a lot of variance (i.e., Brightness, even at severity 5, is far from 40% baseline accuracy,
    # therefore we do not use it.
    if baseline >= 40:
        noises = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "contrast",
            "elastic",
            "pixelate",
            "jpeg",
        ]
    elif baseline >= 20:
        noises = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "contrast",
            "elastic",
            "pixelate",
        ]
    else:
        noises = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "contrast",
        ]

    pickle_path = os.path.join(destination_folder, "ccc_accuracy_matrix.pickle")
    if not os.path.exists(pickle_path):
        url = "https://nc.mlcloud.uni-tuebingen.de/index.php/s/izTMnXkaHoNBZT4/download/ccc_accuracy_matrix.pickle"
        accuracy_matrix = pd.read_pickle(url)
        os.makedirs(destination_folder, exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(accuracy_matrix, f)
    else:
        with open(pickle_path, "rb") as f:
            accuracy_matrix = pickle.load(f)

    for noise1 in noises:
        for noise2 in noises:
            if noise1 == noise2:
                continue
            noise1 = noise1.lower().replace(" ", "_")
            noise2 = noise2.lower().replace(" ", "_")

            current_accuracy_matrix = accuracy_matrix["n1_" + noise1 + "_n2_" + noise2]
            walk = find_path(current_accuracy_matrix, baseline)
            walk_dict[(noise1, noise2)] = walk

    noise1 = random.choice(noises)
    noise2 = random.choice(noises)

    while noise2 == noise1:
        noise2 = random.choice(noises)

    walk = walk_dict[(noise1, noise2)]
    data_path = os.path.join(data_root, "n1_" + noise1 + "_n2_" + noise2)
    walk_datasets = path_to_dataset(walk, data_path)
    walk_ind = 0

    current_datasets = []
    current_datasets_len = 0
    serial = 0
    total_generated_images = 0

    while True:
        path_split = walk_datasets[walk_ind].split("/")
        noise_names = path_split[-2]
        severities = path_split[-1]

        noises_split = noise_names.split("_")
        noises_split = [noise for noise in noises_split if noise not in ["n1", "n2"]]

        if len(noises_split) == 2:
            n1 = noises_split[0]
            n2 = noises_split[1]
        if len(noises_split) == 3:
            question = noises_split[1]
            if question in ["blur", "noise"]:
                n1 = noises_split[0] + "_" + noises_split[1]
                n2 = noises_split[2]
            else:
                n1 = noises_split[0]
                n2 = noises_split[1] + "_" + noises_split[2]
        if len(noises_split) == 4:
            n1 = noises_split[0] + "_" + noises_split[1]
            n2 = noises_split[2] + "_" + noises_split[3]

        severities_split = severities.split("_")
        s1 = float(severities_split[1][:-2])
        s2 = float(severities_split[2])

        if serial % totalprocesses == serial_ind:
            current_destination_path = os.path.join(
                final_path, "serial_" + str(serial).zfill(5)
            )

            if os.path.exists(os.path.join(current_destination_path + ".tar")):
                if (
                    isValid(
                        os.path.join(current_destination_path + ".tar"),
                        total_images=(IMGS_PER + (IMGS_PER % speed)),
                    )
                    == False
                ):
                    os.remove(os.path.join(current_destination_path + ".tar"))
                else:
                    current_datasets_len += speed
                    if current_datasets_len >= IMGS_PER:
                        total_generated_images += current_datasets_len
                        serial += 1
                        current_datasets = []
                        current_datasets_len = 0
            else:
                cur_data = ApplyTransforms(data_root, n1, n2, s1, s2, speed)
                assert len(cur_data) == speed
                current_datasets.append(cur_data)
                current_datasets_len += len(cur_data)

                if current_datasets_len >= IMGS_PER:
                    concatenated_datasets = torch.utils.data.ConcatDataset(
                        current_datasets
                    )
                    write_dataset(concatenated_datasets, current_destination_path)
                    total_generated_images += current_datasets_len
                    serial += 1
                    current_datasets = []
                    current_datasets_len = 0

        else:
            current_datasets_len += speed
            if current_datasets_len >= IMGS_PER:
                total_generated_images += current_datasets_len
                serial += 1
                current_datasets = []
                current_datasets_len = 0

        if total_generated_images >= cutoff:
            return

        if walk_ind == len(walk) - 1:
            noise1 = noise2

            while noise1 == noise2:
                noise2 = random.choice(noises)
                noise2 = noise2.lower().replace(" ", "_")

            walk = walk_dict[(noise1, noise2)]
            data_path = os.path.join(data_root, "n1_" + noise1 + "_n2_" + noise2)
            walk_datasets = path_to_dataset(walk, data_path)
            walk_ind = 0
        else:
            walk_ind += 1


class ApplyTransforms(data.Dataset):
    def __init__(self, data_root, n1, n2, s1, s2, freq, use_trn2=False):
        self.data_root = data_root
        d = noise_dict()

        self.n1 = d[n1]
        self.n2 = d[n2]
        self.s1 = s1
        self.s2 = s2
        self.use_trn2 = use_trn2

        self.n1_name = n1
        self.n2_name = n2

        self.trn1 = trn.Compose(
            [trn.RandomHorizontalFlip(), trn.RandomResizedCrop(224)]
        )

        all_paths = []

        for path, dirs, files in os.walk(self.data_root):
            for name in files:
                all_paths.append(os.path.join(path, name))

        np.random.shuffle(all_paths)
        self.paths = all_paths
        self.paths = self.paths[:freq]
        all_classes = os.listdir(data_root)

        target_list = []
        for cur_path in self.paths:
            cur_class = cur_path.split("/")[-2]
            cur_class = all_classes.index(cur_class)
            target_list.append(cur_class)

        self.targets = target_list

    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]
        img = pil_loader(path)

        img = self.trn1(img)
        if self.s1 > 0:
            img = self.n1(img, self.s1)
            img = Image.fromarray(np.uint8(img))
        if self.s2 > 0:
            img = self.n2(img, self.s2)

        if self.s2 > 0:
            img = Image.fromarray(np.uint8(img))
        output = io.BytesIO()
        img.save(output, format="JPEG", quality=85, optimize=True)
        corrupted_img = output.getvalue()
        info = (
            "++dir++"
            + str(path.split("/")[-2])
            + "++name++"
            + str(path.split("/")[-1])
            + "++n1++"
            + self.n1_name
            + "++s1++"
            + str(self.s1)
            + "++n2++"
            + self.n2_name
            + "++s2++"
            + str(self.s2)
        )
        return corrupted_img, target, info

    def __len__(self):
        return len(self.paths)


def identity(x):
    return x


def isValid(url, total_images):
    ind = (url).split("_")[-1].replace(".tar", "")
    url = os.path.join(url[:-16], "serial_{{{}..{}}}.tar".format(ind, ind))
    normalize = trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    preproc = trn.Compose(
        [
            trn.ToTensor(),
            normalize,
        ]
    )

    dataset = (
        wds.WebDataset(url)
        .decode("pil")
        .to_tuple("input.jpg", "output.cls")
        .map_tuple(preproc, identity)
    )

    loader = iter(torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=50))
    i = 0
    while True:
        try:
            img, label = next(loader)
        except:
            break
        i += 1
    if i * 50 < total_images:
        return False
    return True


def get_frost_images(data_dir):
    """
    Downloads frost images from the ImageNet-C repo.
    Parameters
    ----------
    data_dir : str
        where the images will be saved
    target_dir : str
    """
    url = "https://raw.githubusercontent.com/hendrycks/robustness/master/ImageNet-C/create_c/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    frost_images_path = os.path.join(data_dir, "frost")
    frost_images = {
        "frost1.png",
        "frost2.png",
        "frost3.png",
        "frost4.jpg",
        "frost5.jpg",
        "frost6.jpg",
    }
    if not os.path.exists(frost_images_path):
        os.mkdir(frost_images_path)

    for image_name in frost_images.difference(set(os.listdir(frost_images_path))):
        response = requests.get(url + image_name)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(frost_images_path, image_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=float)
    parser.add_argument("--processind", type=int)
    parser.add_argument("--totalprocesses", type=int)
    parser.add_argument("--imagenetval", type=str)
    parser.add_argument("--dest", type=str)
    args = parser.parse_args()

    speed = [1000, 2000, 5000][
        (args.processind % 9) % 3
    ]  # transition speeds used the paper
    seed = [43, 44, 45][
        int((args.processind % 9) / 3)
    ]  # random seeds used in the paper
    serial_ind = int(args.processind / 9)

    GenerateDataset(
        data_root=args.imagenetval,
        destination_folder=args.dest,
        speed=speed,
        seed=seed,
        baseline=args.baseline,
        serial_ind=serial_ind,
        totalprocesses=int(args.totalprocesses / 9),
    )
