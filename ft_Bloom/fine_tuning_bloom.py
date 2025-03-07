# Import dependencies 

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import nltk
nltk.download('punkt', quiet=True)

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    AutoModelForCausalLM,
    GPT2Tokenizer, 
    GPT2Model
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
from transformers import AdamW
from tqdm import tqdm

def get_model_size(model):
    """
    Calculates the size of a model in MB before loading it to GPU.
    """
    num_params = sum(p.numel() for p in model.parameters())  # Total parameters
    size_in_bytes = num_params * 4  # Each parameter is 4 bytes (float32)
    size_in_mb = size_in_bytes / (1024 ** 2)  # Convert bytes to MB
    size_in_gb = size_in_mb / 1024  # Convert MB to GB
    return size_in_mb, size_in_gb

# Preparing the dataset : 

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# -------------------------------
# 1. Load and Format the Legal Corpus with a Sliding Window over Sentences
# -------------------------------
def load_and_chunk_dataset(file_path, window_size_sentences=5, stride_sentences=3):
    """
    Reads a text file, splits it into sentences using NLTK, and creates overlapping chunks 
    (i.e. paragraphs) using a sliding window approach.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()
    
    sentences = nltk.sent_tokenize(full_text)
    print(f"Total sentences found: {len(sentences)}")
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + window_size_sentences])
        if chunk:
            chunks.append(chunk)
        if i + window_size_sentences >= len(sentences):
            break
        i += stride_sentences
    
    print(f"Total chunks created: {len(chunks)}")
    
    data = {"text": chunks}
    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    return dataset

def load_and_chunk_dataset_from_csv(file_path, column="cours", window_size_sentences=5, stride_sentences=3, seed=42):
    """
    Reads a CSV file, extracts text from the specified column (default: "cours"),
    splits the combined text into sentences using NLTK, and creates overlapping chunks
    (paragraphs) using a sliding window approach.
    
    Parameters:
        file_path (str): Path to the CSV file.
        column (str): Column name containing text to be chunked.
        window_size_sentences (int): Number of sentences per chunk.
        stride_sentences (int): Number of sentences to skip for each new chunk.
        seed (int): Random seed for train-test split.
    
    Returns:
        dataset: A HuggingFace Dataset with a train/test split, where each sample has a "text" field.
    """
    # Load CSV file
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    
    # Concatenate all text from the specified column into a single string.
    all_text = " ".join(df[column].dropna().tolist()).strip()
    
    # Tokenize text into sentences using NLTK.
    sentences = nltk.sent_tokenize(all_text, language="french")
    print(f"Total sentences found: {len(sentences)}")
    
    # Create overlapping chunks using a sliding window.
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i: i + window_size_sentences])
        if chunk:
            chunks.append(chunk)
        if i + window_size_sentences >= len(sentences):
            break
        i += stride_sentences
    print(f"Total chunks created: {len(chunks)}")
    
    # Create a dictionary and then a HuggingFace Dataset.
    data = {"text": chunks}
    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    return dataset


print("Loading and chunking the dataset...")
# Specify your file path here
dataset = load_and_chunk_dataset_from_csv(
    "/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/data/Juridix/coursDDAformaté.csv",
    window_size_sentences=3,
    stride_sentences=2
)
print("Train dataset length:", len(dataset["train"]))
print("Validation dataset length:", len(dataset["test"]))

# Importing the model and tokenizer : 
model_name = "bigscience/bloom-1b1"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/models_cache")

# -------------------------------
# 2. Unified Tokenization Function
# -------------------------------

def preprocess_and_tokenize_function(examples):
    """
    Tokenizes input text while handling truncation and padding.
    
    - Truncates to a max length of 512 tokens.
    - Applies padding to ensure uniform sequence size.
    - Converts input text into model-friendly input tensors.
    
    Args:
        examples (dict): Dictionary containing the "text" field.

    Returns:
        dict: Tokenized examples with `input_ids` and `attention_mask`.
    """
    return tokenizer(
        examples["text"],
        max_length=200,  # Enforces fixed sequence length
        padding="max_length",  # Pads to exactly 128 tokens
        truncation=True,  # Truncates longer texts to 128 tokens
        return_attention_mask=True,  # Returns attention mask for padding
        return_tensors="pt"  # Returns PyTorch tensors
    )

tokenized_dataset = dataset.map(preprocess_and_tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal Language Modeling
    pad_to_multiple_of=200  # Ensures consistent padding across batches
)

# -------------------------------
# 5. Load Pretrained Model with LoRA
# -------------------------------
# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/models_cache")
print("Base model loaded")

lora_config = LoraConfig(
    r=16,                      # Lower rank might be more stable for your model
    lora_alpha=8,             # Adjusted scaling factor
    target_modules=["query", "key", "value", "dense"],
    lora_dropout=0.05,        # Slightly lower dropout if overfitting isn’t a concern
    bias="none"
)

# Apply LoRA adaptation
model = get_peft_model(model, lora_config)
print("LoRA adaptation applied")
model.print_trainable_parameters()

model_size_mb, model_size_gb = get_model_size(model)
print(f"Model size: {model_size_mb:.2f} MB ({model_size_gb:.2f} GB)")


torch.cuda.empty_cache()
print("CUDA MEMORY has been freed")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on device: {device}")

# Print number of trainable parameters
model.print_trainable_parameters()

# -------------------------------
# 6.1 Define Training Parameters
# -------------------------------

# Set up directories
output_dir = "/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/fine_tuning_bloom_v5"  
os.makedirs(output_dir, exist_ok=True)

# Training hyperparameters
num_epochs = 5
batch_size = 8  # Reduce if running out of memory
learning_rate = 1e-5
weight_decay = 0.01
save_steps = 500  # Save model every N steps
fp16 = True  # Enable mixed precision training

# model.train()

# -------------------------------
# 1. Define TrainingArguments
# -------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,                      # output directory for checkpoints & final model
    num_train_epochs=num_epochs,                # total number of training epochs
    per_device_train_batch_size=batch_size,     # batch size per device during training
    learning_rate=learning_rate,                # initial learning rate
    weight_decay=weight_decay,                  # weight decay if applicable
    save_steps=save_steps,                      # save checkpoint every N steps
    fp16=fp16,                                  # enable mixed precision training
    logging_steps=100,                          # log training info every 100 steps
    save_total_limit=2,                         # limit the total number of checkpoints
    report_to="none"                            # disable integration with experiment trackers
)

# -------------------------------
# 2. Initialize Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,   # your custom data collator
)

# -------------------------------
# 3. Resume from a Checkpoint if Available
# -------------------------------
# If you have a checkpoint, you can resume training as follows:
# trainer.train(resume_from_checkpoint="/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/fine_tuning_bloom_v4/checkpoint-1470")
trainer.train()

# -------------------------------
# 4. Save the Final Model & Tokenizer
# -------------------------------
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Training completed. Model saved at {output_dir}")