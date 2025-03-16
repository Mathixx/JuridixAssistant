#!/usr/bin/env python
"""
t5_fine_tuning.py

This script fine-tunes a T5 model on a custom dataset provided as a CSV file.
It was created by reviewing your notebook cell by cell to preserve the original training pipeline.
The CSV file must have the following columns:
  - initial_query : The raw input query.
  - type          : The document type.
  - new_query     : The reformulated query.

The target for fine-tuning is constructed as:
    "type: {doc_type} ; new_query: {new_query} </s>"

After training, the script saves the fine-tuned model and tokenizer.
"""

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

# =============================================================================
# Section 1: Dataset Definition (Notebook Cells 1-2)
# =============================================================================
class QueryDataset(Dataset):
    """
    Dataset class for T5 fine-tuning.
    
    Expects a CSV file with columns:
      - initial_query: The raw query.
      - type: The document type.
      - new_query: The improved query.
    
    The target is built in the format:
        "type: {doc_type} ; new_query: {new_query} </s>"
    
    Both input and target are tokenized using the provided tokenizer.
    """
    def __init__(self, tokenizer, csv_file, max_len=512, target_max_len=128):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_file)
        self.max_len = max_len
        self.target_max_len = target_max_len
        self.inputs = []
        self.targets = []
        self._build_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Squeeze to remove extra dimensions added by the tokenizer.
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

    def _build_dataset(self):
        for _, row in self.data.iterrows():
            # Extract and clean the query and target components.
            initial_query = row['initial_query'].strip()
            doc_type = row['type'].strip()
            new_query = row['new_query'].strip()

            # Construct input and target text.
            input_text = initial_query + " </s>"
            target_text = f"type: {doc_type} ; new_query: {new_query} </s>"

            # Tokenize the input query.
            tokenized_input = self.tokenizer.batch_encode_plus(
                [input_text],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            # Tokenize the target text.
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.target_max_len,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

# =============================================================================
# Section 2: Training Pipeline (Notebook Cells 3-5)
# =============================================================================
def train_model(model, tokenizer, dataset, device, epochs=3, batch_size=8, learning_rate=5e-5):
    """
    Fine-tunes the T5 model on the provided dataset.
    
    Args:
      model: T5ForConditionalGeneration instance.
      tokenizer: T5Tokenizer instance.
      dataset: QueryDataset instance.
      device: torch.device ("cuda" or "cpu").
      epochs: Number of training epochs.
      batch_size: Batch size.
      learning_rate: Learning rate for the optimizer.
    
    Returns:
      The fine-tuned model.
    """
    # Create DataLoader for training.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize the optimizer and learning rate scheduler.
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        # Iterate over batches.
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['source_ids'].to(device)
            attention_mask = batch['source_mask'].to(device)
            labels = batch['target_ids'].to(device)
            
            # Forward pass; model returns loss when labels are provided.
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Average loss for epoch {epoch+1}: {avg_loss:.4f}")
    return model

# =============================================================================
# Section 3: Evaluation (Optional, Notebook Cell 6)
# =============================================================================
def evaluate_model(model, tokenizer, dataset, device, batch_size=8, max_gen_len=64):
    """
    Generates predictions for the validation dataset and prints some statistics.
    
    This function demonstrates how to run inference and extract the classification
    field from the generated output.
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    model.eval()
    outputs = []
    targets = []

    # Generate predictions for each batch.
    for batch in tqdm(loader, desc="Evaluating"):
        outs = model.generate(
            input_ids=batch['source_ids'].to(device),
            attention_mask=batch['source_mask'].to(device),
            max_length=max_gen_len
        )
        decoded_outs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        decoded_targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(decoded_outs)
        targets.extend(decoded_targets)
    
    # Print a few examples for inspection.
    print("\nSample Predictions:")
    for i in range(min(5, len(outputs))):
        print(f"Input: {tokenizer.decode(batch['source_ids'][0], skip_special_tokens=True)}")
        print(f"Prediction: {outputs[i]}")
        print(f"Target: {targets[i]}")
        print("-"*40)

    return outputs, targets

# =============================================================================
# Section 4: Main Function (Notebook Final Cells)
# =============================================================================
def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Fine-tune T5 on a custom CSV dataset.")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the CSV file containing training data.")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="Pre-trained T5 model name (e.g., t5-small, t5-base).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model and tokenizer.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size.")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum length for input queries.")
    parser.add_argument("--target_max_len", type=int, default=128,
                        help="Maximum length for target sequences.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Run evaluation after training.")
    args = parser.parse_args()

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load tokenizer and model.
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)

    # Load dataset.
    print("Loading dataset from:", args.csv_file)
    dataset = QueryDataset(tokenizer, args.csv_file, max_len=args.max_len, target_max_len=args.target_max_len)
    
    # Train the model.
    print("Starting training...")
    model = train_model(model, tokenizer, dataset, device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate)
    
    # Optionally evaluate the model.
    if args.do_eval:
        print("Evaluating model...")
        evaluate_model(model, tokenizer, dataset, device, batch_size=args.batch_size)
    
    # Save model and tokenizer.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()


# TO RUN:
"""
python t5_fine_tuning.py --csv_file path/to/your/data.csv --output_dir path/to/save/model --epochs 3 --batch_size 8 --do_eval
"""
