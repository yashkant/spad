import torch 
import os
import glob
import numpy as np
import unicodedata
import re

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, verbose=False, inference_run=False):
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd["state_dict"] if "state_dict" in sd else sd
    config.model.params.ckpt_path = None
    model = instantiate_from_config(config.model)
    model.reinit_ema()

    print(f"Loading model from {ckpt}")
    m, u = model.load_state_dict(sd, strict=False)
    
    return model


def read_ckpt(opt, return_configs=False):
    if os.path.isdir(opt.ckpt):
        # recursively read all files in the folder
        last_ckpt = None
        for root, dirs, files in os.walk(opt.ckpt):
            for file in files:
                if file.endswith("last.ckpt"):
                    if last_ckpt is None:
                        last_ckpt = os.path.join(root, file)
                    else:
                        # compare the timestamp
                        last_ckpt_time = os.path.getmtime(last_ckpt)
                        current_ckpt_time = os.path.getmtime(os.path.join(root, file))
                        if current_ckpt_time > last_ckpt_time:
                            last_ckpt = os.path.join(root, file)
        opt.ckpt = last_ckpt

    if return_configs:
        # build model and exp config
        path_splits = opt.ckpt.split("/")
        path_splits = path_splits[:path_splits.index("checkpoints")]
        exp_dir = "/".join(path_splits)
        exp_configs = glob.glob(os.path.join(exp_dir, "configs", "*.yaml"))
        assert len(exp_configs) == 2, "should have 2 configs, one for model, one for exp"
        model_config = exp_configs[0] if "project" in exp_configs[0] else exp_configs[1]
        model_config = OmegaConf.load(model_config)

        lightning_config = exp_configs[0] if "lightning" in exp_configs[0] else exp_configs[1]
        lightning_config = OmegaConf.load(lightning_config) 

        return opt, model_config, lightning_config

    return opt


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

