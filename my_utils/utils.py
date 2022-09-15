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
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
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
    parser.add_argument('--baseline', help='set this flag to train the baseline madx', action='store_true')
    parser.add_argument("--seed", default=123, required=False)
    parser.add_argument("--epochs", default=p["epochs"], required=False)
    parser.add_argument("--sig_center", default=6000, required=False)
    parser.add_argument('--checkpoint', help='if you want to start from a checkpoint. This model must match the model passed with --model', required=False)
    parser.add_argument('--src_lang', nargs='+', help='list of languages which you want to be addded in the train data', required=True)
    parser.add_argument('--target', nargs='+', help='The target language', required=True)
    parser.add_argument('--subspace_dir', help='The directory containing subspace vectors', required=False, default=p["subspace_dir"])
    parser.add_argument('--layers_c', nargs='+', help='list of layers to add converters on', required=False, default=["6"])
    parser.add_argument('--layers_p', nargs='+', help='list of layers to activate projections on', required=False, default=None)
    parser.add_argument("--proj_prob", nargs='+', required=False, default=p["proj_prob"])
    parser.add_argument("--test", help="path to model to be used for testing", required=False)
    parser.add_argument("--policy", nargs='+', help="path to model to be used for testing", required=True)
    parser.add_argument('--save', help='set this flag to save the model after training', action='store_true')
    parser.add_argument("--mean_loss", nargs='+', help="Layers for which mean loss is to be activated", required=False, default=[])
    parser.add_argument("--mean_l_alpha", default=0.1, required=False)
    parser.add_argument("--task", default="NER", required=False)

    return parser

def get_processed_dataset(task_name, tokenizer, src_lang, tgt_lang):

    if task_name.upper() == "XNLI":

        train_dataset = load_dataset(
            "xnli",
            src_lang,
            split="train",
        )
        train_label_list = train_dataset.features["label"].names

        eval_dataset = load_dataset(
            "xnli",
            tgt_lang,
            split="validation",
        )
        eval_label_list = eval_dataset.features["label"].names

        test_dataset = load_dataset(
            "xnli",
            tgt_lang,
            split="test",
        )
        test_label_list = test_dataset.features["label"].names

        # Labels
        num_labels = len(train_label_list)

        padding = False

        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        # data_collator = None

        def preprocess_function(examples):
            # Tokenize the texts
            return tokenizer(
                examples["premise"],
                examples["hypothesis"],
                padding=padding,
                # max_length=data_args.max_seq_length,
                truncation=True,
            )

        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset",
            remove_columns=["hypothesis", "premise"],
        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset",
            remove_columns=["hypothesis", "premise"],
        )
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset",
            remove_columns=["hypothesis", "premise"],
        )

        train_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])
        eval_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=16,
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=16
        )

        test_dataloader = DataLoader(
            test_dataset, collate_fn=data_collator, batch_size=16
        )

        print(train_dataset)

        # for index in random.sample(range(len(train_dataset)), 3):
        #     print(f"Sample {index} of the training set: {train_dataset[index]}.")

        return train_dataloader, eval_dataloader, test_dataloader

    elif task_name.upper() == "NER":

        dataset_src = load_dataset('wikiann', src_lang)
        dataset_tgt = load_dataset('wikiann', tgt_lang)

        # # This method is adapted from the huggingface transformers run_ner.py example script
        # # Tokenize all texts and align the labels with them.

        def tokenize_and_align_labels(examples):
            text_column_name = "tokens"
            label_column_name = "ner_tags"
            tokenized_inputs = tokenizer(
                examples[text_column_name],
                padding=False,
                truncation=True,
                # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                is_split_into_words=True,
            )
            labels = []
            for i, label in enumerate(examples[label_column_name]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.  
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx

                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        dataset_src = dataset_src.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_src["train"].column_names,
        )
        dataset_tgt = dataset_tgt.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset_tgt["train"].column_names,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer,)

        train_dataloader = DataLoader(
            dataset_src["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=16,
        )
        eval_dataloader = DataLoader(
            dataset_tgt["validation"], collate_fn=data_collator, batch_size=16
        )

        test_dataloader = DataLoader(
            dataset_tgt["test"], collate_fn=data_collator, batch_size=16
        )

        return train_dataloader, eval_dataloader, test_dataloader