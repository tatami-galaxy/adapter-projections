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
from my_utils.utils import *
from my_utils.config import *
from my_utils.logger import Logger
from my_utils.colors import bcolors

torch.cuda.empty_cache()
torch.cuda.synchronize()

parser = get_arg_parser()
args = parser.parse_args()

seed = int(args.seed)
seed_everything(seed)

device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")

if args.layers_c is not None:
    converter_active_layers = [int(s) for s in args.layers_c]

proj_active_layers = [int(s) for s in args.layers_p]
layer_probs = [float(p) for p in args.proj_prob]
#The labels for the NER task and the dictionaries to map the to ids or 
#the other way around
model_name = "xlm-roberta-base"

if len(args.src_lang) == 1:
  src_lang = args.src_lang[0]       # en
else:
  src_lang = args.src_lang          

if len(args.target) == 1:
  tgt_lang = args.target[0]         # ja
else:
  tgt_lang = args.target

num_train_epochs = int(args.epochs)

logger = Logger(p["logs_dir"] + "/" + model_name + "p_l_" + "_".join([str(i) for i in proj_active_layers]) + "_p_l_" + "_".join([str(pr) for pr in layer_probs]) +"_src_" + src_lang + "_tgt_" + tgt_lang + "_eps_" + str(num_train_epochs) + "_seed_" + str(seed) + ".log")

logger.write("Seed: " + str(seed))

logger.draw_line()
logger.write("Loading model " + model_name, bcolors.OKCYAN)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if args.train:
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
    model = XLMRobertaAdapterModel.from_pretrained(model_name, config=config)

    logger.write("Loading projections")
    # model.load_adapter_projections(['de', 'hi', 'is', 'es', 'id', 'ja', 'ta', 'th'], 0.9, './subspace/subspace_cache')
    model.load_adapter_projections([tgt_lang], 0.9, './subspace/subspace_cache')
    logger.write("Loaded projections sucessfully!", bcolors.OKGREEN)

    # Load the language adapters
    logger.write("Loading language adapters for: src - " + src_lang + " tgt - " + tgt_lang)
    lang_adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=2)
    model.load_adapter(src_lang+"/wiki@ukp", config=lang_adapter_config) # leave_out=[11])
    model.load_adapter(tgt_lang+"/wiki@ukp", config=lang_adapter_config) # leave_out=[11])

    # model.add_converters(converter_active_layers)

    # Add a new task adapter
    logger.write("Loading task adapters")
    model.add_adapter("wikiann")
    logger.write("Loaded all adapters successfully!", bcolors.OKGREEN)

    model.add_tagging_head("wikiann", num_labels=len(labels))

    model.train_adapter(["wikiann"])

    optimizer = AdamW(model.parameters(), lr=1e-4)

elif args.test:
    config = AutoConfig.from_pretrained(model_name, num_labels=len(labels), label2id=label_2_id, id2label=id_2_label)
    model = torch.load(args.test, config=config)
    best_model = model
else:
    print("Either train or test must be set to true")

logger.write("Loaded model " + model_name + " successfully!", bcolors.OKGREEN)

logger.write("Loading and processing dataset", bcolors.OKCYAN)

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

num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

metric = load_metric("seqeval")

logger.draw_line()
logger.write("Training started", bcolors.OKCYAN)


if args.train:
    progress_bar = tqdm(range(num_training_steps))

    best_model = None
    min_val_loss = 1e7

    model.to(device)

    for epoch in range(num_train_epochs):

        # Unfreeze and activate stack setup
        model.active_adapters = Stack(src_lang, "wikiann")

        for i, prob in zip(proj_active_layers, layer_probs):
            model.activate_adapter_projection_stack('wikiann', i, tgt_lang, prob)

        # for i in converter_active_layers:
        #   model.activate_converter_stack('wikiann', i)

        val_loss = 0
        # Training
        model.train()
        for batch in train_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            del labels
            del inputs

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


        # Evaluation
        model.eval()
        # model.disable_adapter_projection_stack()
        model.active_adapters = Stack(tgt_lang, "wikiann")
        # parallel_net.active_adapters = Stack(tgt_lang, "wikiann")
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=inputs, labels=labels)
                val_loss += outputs.loss.item()

                del labels
                del inputs

                predictions = outputs.logits.argmax(dim=2)
                labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                #predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                #labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                #predictions_gathered = accelerator.gather(predictions)
                #labels_gathered = accelerator.gather(labels)

                true_predictions, true_labels = postprocess(predictions, labels)
                metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        logger.draw_line()
        logger.write(">>> Epoch: " + str(epoch), bcolors.OKCYAN)
        logger.write(
        str(
            f"epoch {epoch}: " +
            str({
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            }),
        )
        )

        net_val_loss = val_loss/len(eval_dataloader)

        print(net_val_loss)

        test_loss = 0
        # model.disable_adapter_projection_stack()
        model.active_adapters = Stack(tgt_lang, "wikiann")

        for batch in test_dataloader:
            with torch.no_grad():
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=inputs, labels=labels)
                test_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=2)
            labels = batch["labels"]

                # Necessary to pad predictions and labels for being gathered
                #predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                #labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

                #predictions_gathered = accelerator.gather(predictions)
                #labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess(predictions, labels)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print(f"Testepoch {epoch}:", {key: results[f"overall_{key}"] for key in ["precision", "recall", "f1", "accuracy"]},)
        print(test_loss/len(test_dataloader))

        if min_val_loss > net_val_loss or best_model is None:
            best_model = model
            torch.save(best_model, p["save_dir"] + "/" + model_name + "p_l_" + "_".join([str(i) for i in proj_active_layers]) + "_src_" + src_lang + "_tgt_" + tgt_lang + "_eps_" + str(num_train_epochs) + "_seed_" + str(seed) + "_baseline.pt")
            min_val_loss = net_val_loss

    logger.draw_line()


    best_model = torch.load(p["save_dir"] + "/" + model_name + "p_l_" + "_".join([str(i) for i in proj_active_layers]) + "_src_" + src_lang + "_tgt_" + tgt_lang + "_eps_" + str(num_train_epochs) + "_seed_" + str(seed) + "_baseline.pt")


if args.test:
    test_loss = 0
    best_model.disable_adapter_projection_stack()
    best_model.active_adapters = Stack(tgt_lang, "wikiann")

    for batch in test_dataloader:
        with torch.no_grad():
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = best_model(input_ids=inputs, labels=labels)
            test_loss += outputs.loss.item()

        predictions = outputs.logits.argmax(dim=2)
        labels = batch["labels"]

            # Necessary to pad predictions and labels for being gathered
            #predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            #labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            #predictions_gathered = accelerator.gather(predictions)
            #labels_gathered = accelerator.gather(labels)

        true_predictions, true_labels = postprocess(predictions, labels)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(f"epoch {epoch}:", {key: results[f"overall_{key}"] for key in ["precision", "recall", "f1", "accuracy"]},)
    print(test_loss/len(test_dataloader))