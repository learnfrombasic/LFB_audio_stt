import random
import string
from datetime import datetime

import torch



def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_run_name(
    model_name: str,
    dset_name: str,
    lr: float,
    batch_size: int,
    experiment_type: str = "",
    random_suffix: bool = True,
    random_suffix_len: int = 6,
) -> str:
    """Generate a unique run name with timestamp."""
    now = datetime.now().strftime("%m%d-%H%M")
    rand_suffx = (
        "".join(
            random.Random().choices(
                string.ascii_letters + string.digits, k=random_suffix_len
            )
        )
        if random_suffix
        else ""
    )
    return f"{model_name}_{dset_name}_{experiment_type}_lr{lr}_bs{batch_size}_{now}_{rand_suffx}"
