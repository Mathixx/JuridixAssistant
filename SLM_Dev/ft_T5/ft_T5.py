#!/usr/bin/env python
"""
t5_fine_tuning.py

This script fine-tunes a T5 model on a custom dataset provided as a CSV file.
It is based on your notebook's cells and uses PyTorch Lightning for training.
The CSV file should have the following columns:
    - initial_query : The raw input query.
    - type          : The document type.
    - new_query     : The improved/reformulated query.

The target text is constructed as:
    "type: {doc_type} ; new_query: {new_query} </s>"

After training, the model is saved and (optionally) evaluated.
"""

import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from tqdm import tqdm, trange
import textwrap

# -----------------------------------------------------------------------------
# Set random seed for reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------------------------------------------------------------
# Define the LightningModule for T5 Fine-Tuning (Cell "T5FineTuner")
# -----------------------------------------------------------------------------
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        """
        hparams: a Namespace or dict with hyperparameters.
        Expected keys include:
          - model_name_or_path, tokenizer_name_or_path, weight_decay,
          - learning_rate, adam_epsilon, warmup_steps, train_batch_size,
          - eval_batch_size, num_train_epochs, gradient_accumulation_steps,
          - n_gpu, fp_16, opt_level, max_seq_length, output_dir, etc.
        """
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = T5ForConditionalGeneration.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=hparams.cache_dir if hasattr(hparams, "cache_dir") else None
        )
        self.tokenizer = T5Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path,
            cache_dir=hparams.cache_dir if hasattr(hparams, "cache_dir") else None
        )

    def is_logger(self):
        return self.trainer.proc_rank <= 0

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        # Set pad tokens to -100 to ignore in loss calculation.
        lm_labels = batch["target_ids"].clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        # Prepare optimizer and learning rate scheduler.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # This method supports gradient accumulation, etc.
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True, num_workers=4)
        # Compute total training steps for scheduler.
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(self.tokenizer, type_path="val", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)

# -----------------------------------------------------------------------------
# Define Logging Callback (for logging validation/test results)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

# -----------------------------------------------------------------------------
# Define the Dataset Class for Query Data (CSV based) (Cells from "QueryDataset")
# -----------------------------------------------------------------------------
class QueryDataset(Dataset):
    """
    Dataset class to read a CSV file with columns:
       - initial_query
       - type
       - new_query
    It tokenizes both the input query and the target sequence.
    The target is formatted as:
       "type: {doc_type} ; new_query: {new_query} </s>"
    """
    def __init__(self, tokenizer, path_to_csv, max_len=512, target_max_len=128):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(path_to_csv)
        self.max_len = max_len
        self.target_max_len = target_max_len
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

    def _build(self):
        for _, row in self.data.iterrows():
            initial_query = row['initial_query'].strip()
            doc_type = row['type'].strip()
            new_query = row['new_query'].strip()

            input_text = initial_query + " </s>"
            target_text = f"type: {doc_type} ; new_query: {new_query} </s>"

            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.target_max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

# -----------------------------------------------------------------------------
# Define get_dataset function used by the model (for train/val splits)
# -----------------------------------------------------------------------------
def get_dataset(tokenizer, type_path, args):
    # Here type_path is used if you want to select different splits.
    # For this example, we assume the same CSV is used (modify as needed).
    return QueryDataset(tokenizer=tokenizer, path_to_csv=args.path_to_csv, max_len=args.max_seq_length, target_max_len=args.target_max_len)

# -----------------------------------------------------------------------------
# Define utility function to extract fields from generated text (for evaluation)
# -----------------------------------------------------------------------------
def extract_fields(text):
    """
    Extracts document type and new_query from a formatted string.
    Expected format: "type: {doc_type} ; new_query: {new_query}"
    Returns a tuple (doc_type, new_query) or (None, None) if extraction fails.
    """
    pattern = r"^type:\s*([^\s;]+)\s*;\s*new_query:\s*(.+)$"
    match = re.match(pattern, text.lower().strip())
    if match:
        doc_type = match.group(1).strip()
        new_query = match.group(2).strip()
        return doc_type, new_query
    return None, None

# -----------------------------------------------------------------------------
# Evaluation function to generate predictions and compute metrics
# -----------------------------------------------------------------------------
def evaluate_model(model, tokenizer, dataset, device, batch_size=32, max_gen_len=128):
    """
    Generates predictions on the given dataset and computes overall and per-category accuracy.
    Also plots a confusion matrix.
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    model.model.eval()
    outputs = []
    targets = []

    for batch in tqdm(loader, desc="Evaluating"):
        outs = model.model.generate(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device),
            max_length=max_gen_len
        )
        decoded_outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        decoded_targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(decoded_outputs)
        targets.extend(decoded_targets)

    # Print a few examples
    print("\nSample Predictions:")
    for i in range(min(5, len(outputs))):
        print("Input:", tokenizer.decode(loader.dataset.inputs[i]["input_ids"].squeeze(), skip_special_tokens=True))
        print("Prediction:", outputs[i])
        print("Target:", targets[i])
        print("-" * 50)

    # Calculate overall accuracy
    correct_classification = 0
    total_samples = len(outputs)
    for pred_text, tgt_text in zip(outputs, targets):
        pred_type, _ = extract_fields(pred_text)
        tgt_type, _ = extract_fields(tgt_text)
        if pred_type is None or tgt_type is None:
            continue
        if pred_type == tgt_type:
            correct_classification += 1
    overall_accuracy = correct_classification / total_samples * 100
    print(f"Overall Classification Accuracy: {overall_accuracy:.2f}%")

    # Per-category accuracy
    per_category_correct = defaultdict(int)
    per_category_total = defaultdict(int)
    for pred_text, tgt_text in zip(outputs, targets):
        pred_type, _ = extract_fields(pred_text)
        tgt_type, _ = extract_fields(tgt_text)
        if pred_type is None or tgt_type is None:
            continue
        per_category_total[tgt_type] += 1
        if pred_type == tgt_type:
            per_category_correct[tgt_type] += 1
    print("Per-Category Accuracy:")
    for doc_type in per_category_total:
        cat_accuracy = per_category_correct[doc_type] / per_category_total[doc_type] * 100
        print(f"  {doc_type}: {cat_accuracy:.2f}% ({per_category_correct[doc_type]}/{per_category_total[doc_type]})")

    # Confusion matrix plot
    true_labels = []
    pred_labels = []
    for pred_text, tgt_text in zip(outputs, targets):
        pred_type, _ = extract_fields(pred_text)
        tgt_type, _ = extract_fields(tgt_text)
        if pred_type is None or tgt_type is None:
            continue
        true_labels.append(tgt_type)
        pred_labels.append(pred_type)
    classes = sorted(list(set(true_labels + pred_labels)))
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    return outputs, targets

# -----------------------------------------------------------------------------
# Main function to set up hyperparameters, initialize model, trainer, and run training/evaluation
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5 on a custom CSV dataset using PyTorch Lightning")
    parser.add_argument("--path_to_csv", type=str, default="FILL INNNNN", help="Path to the CSV file for training")
    parser.add_argument("--output_dir", type=str, default="/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/ft_T5/fineTunedT5s", help="Directory to save the fine-tuned model checkpoints")
    parser.add_argument("--model_name_or_path", type=str, default="t5-large", help="Pre-trained T5 model name or path")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="t5-large", help="Pre-trained T5 tokenizer name or path")
    parser.add_argument("--cache_dir", type=str, default="", help="Cache directory for model/tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum input sequence length")
    parser.add_argument("--target_max_len", type=int, default=128, help="Maximum target sequence length")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--fp_16", action="store_true", help="Enable 16-bit (mixed) precision training")
    parser.add_argument("--opt_level", type=str, default="O1", help="Apex AMP optimization level")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation after training")
    args = parser.parse_args()

    # Update hyperparameters dictionary (args_dict equivalent)
    args_dict = {
        "path_to_csv": args.path_to_csv,
        "output_dir": args.output_dir,
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "cache_dir": args.cache_dir,
        "max_seq_length": args.max_seq_length,
        "target_max_len": args.target_max_len,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_epsilon": args.adam_epsilon,
        "warmup_steps": args.warmup_steps,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "n_gpu": args.n_gpu,
        "fp_16": args.fp_16,
        "opt_level": args.opt_level,
        "max_grad_norm": args.max_grad_norm,
        "output_dir": args.output_dir,
        "seed": 42,
    }
    args = argparse.Namespace(**args_dict)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize model (LightningModule)
    model = T5FineTuner(args)
    model.to(device)

    # Define checkpoint callback for saving best models.
    checkpoint_callback = ModelCheckpoint(
        filepath=args.output_dir,
        prefix="checkpoint",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
    )

    # Training parameters for PyTorch Lightning Trainer.
    train_params = {
        "accumulate_grad_batches": args.gradient_accumulation_steps,
        "gpus": args.n_gpu,
        "max_epochs": args.num_train_epochs,
        "early_stop_callback": False,
        "precision": 16 if args.fp_16 else 32,
        "amp_level": args.opt_level,
        "gradient_clip_val": args.max_grad_norm,
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [LoggingCallback()],
    }

    # Initialize trainer.
    trainer = pl.Trainer(**train_params)

    # Start training.
    trainer.fit(model)

    # Save the final model.
    final_output_dir = os.path.join(args.output_dir, "final_model")
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
    # Save using the underlying Hugging Face model.
    model.model.save_pretrained(final_output_dir)
    model.tokenizer.save_pretrained(final_output_dir)
    print(f"Model and tokenizer saved to {final_output_dir}")

    # Optionally run evaluation on test dataset.
    if args.do_eval:
        # For evaluation, you can specify a test CSV file (set path_to_test_csv below).
        parser.add_argument("--path_to_test_csv", type=str, default="FILL THIS IS AS WELLLLLL", help="Path to the CSV file for testing")
        test_csv = args.path_to_csv  # Replace with args.path_to_test_csv if different.
        test_dataset = QueryDataset(model.tokenizer, path_to_csv=test_csv, max_len=args.max_seq_length, target_max_len=args.target_max_len)
        evaluate_model(model, model.tokenizer, test_dataset, device, batch_size=args.eval_batch_size)

if __name__ == "__main__":
    main()
    
# python t5_fine_tuning.py --path_to_csv /path/to/your/data.csv --do_eval
