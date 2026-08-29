import math


SPEEDS = (1000, 2000, 5000)
SEEDS = (43, 44, 45)


def baseline_name(baseline):
    value = float(baseline)
    if not math.isfinite(value):
        raise ValueError("baseline must be finite")
    if value.is_integer():
        return str(int(value))
    return format(value, "g")


def dataset_name(baseline, speed, seed):
    return "baseline_{}_transition+speed_{}_seed_{}".format(
        baseline_name(baseline), speed, seed
    )


def stream_config(process_index):
    index = process_index % 9
    return SPEEDS[index % len(SPEEDS)], SEEDS[index // len(SPEEDS)]
