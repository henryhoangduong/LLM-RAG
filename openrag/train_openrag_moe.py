import os
from os.path import exists, join, isdir
import gc
import json
import math
import random
import copy
from copy import deepcopy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any

import torch
from torch import nn
from torch.utils.data import Dataset
import bitsandbytes as bnb

import transformers
from transformers import Trainer, BitsAndBytesConfig, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from datasets import load_dataset
import GPUtil

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
context_markups = []


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora_r: int = field(
        default=64, metadata={"help": "Rank of the LoRA update matrices"}
    )
    lora_alpha: int = field(default=16, metadata={"help": "Scaling factor for LoRA"})
