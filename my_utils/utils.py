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

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    label_list = id_2_label

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_labels, true_predictions

def seed_everything(seed: int):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:200"
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--model', help='model to be used for training', default = 'projection')
    parser.add_argument('--train', help='set this flag to train the model from scratch', action='store_true')
    parser.add_argument("--seed", default=123, required=False)
    parser.add_argument("--epochs", default=p["epochs"], required=False)
    parser.add_argument('--checkpoint', help='if you want to start from a checkpoint. This model must match the model passed with --model',required=False)
    parser.add_argument('--src_lang', nargs='+', help='list of languages which you want to be addded in the train data', required=True)
    parser.add_argument('--target', nargs='+', help='The target language', required=True)
    parser.add_argument('--subspace_dir', help='The directory containing subspace vectors', required=False, default=p["subspace_dir"])
    parser.add_argument('--layers_c', nargs='+', help='list of layers to add converters on', required=False, default=["6"])
    parser.add_argument('--layers_p', nargs='+', help='list of layers to activate projections on', required=False, default=["6"])
    parser.add_argument("--proj_prob", nargs='+', required=False, default=p["proj_prob"])
    parser.add_argument("--test", help="path to model to be used for testing", required=False)

    # args = parser.parse_args()

    return parser