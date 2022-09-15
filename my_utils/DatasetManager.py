import random, os
import numpy as np
import torch

from transformers import AutoTokenizer, AutoConfig, XLMRobertaAdapterModel
from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm
from torch import nn
import copy
from transformers import AdapterConfig
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from datasets import load_metric
import numpy as np
from transformers.adapters.composition import Stack
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
from my_utils.utils import *
from my_utils.config import *


class DatasetManager:

    def __init__(self):

        self.task = None
    
    def 