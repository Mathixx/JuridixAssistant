import os
import math
import random
import numpy as np
import torch
import nltk
nltk.download('punkt', quiet=True)

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print("Code started")
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
    split_ds = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset, val_dataset = split_ds["train"], split_ds["test"]
    return train_dataset, val_dataset

# Specify your file path here
train_dataset, val_dataset = load_and_chunk_dataset(
    "/users/eleves-a/2022/amine.chraibi/Desktop/alurix/legifrance.gouv.fr.txt",
    window_size_sentences=5,
    stride_sentences=3
)
print("Train dataset length:", len(train_dataset))
print("Validation dataset length:", len(val_dataset))

# -------------------------------
# 2. On-the-Fly Tokenization with Transform
# -------------------------------
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def transform(example):
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    return tokenized

# Apply tokenization transformation
train_dataset = train_dataset.with_transform(transform)
val_dataset = val_dataset.with_transform(transform)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# -------------------------------
# 3. (Previously) Fine-Tuning with LoRA and MLM was Performed
# -------------------------------
# We assume your fine-tuned model is saved at the following path:
finetuned_model_dir = "/users/eleves-a/2022/amine.chraibi/lora_finetuned_legifrance_camembert_mlm"

# -------------------------------
# 4. Define Evaluation Training Arguments
# -------------------------------
# Setting prediction_loss_only=True avoids concatenating logits.
eval_training_args = TrainingArguments(
    output_dir="./results_ddp_eval",
    per_device_eval_batch_size=1,       # Minimal batch size to save memory
    evaluation_strategy="steps",          # (Arbitrary; we call evaluate() once)
    eval_steps=500,                       # Not used since we'll call evaluate() once
    fp16=False,                       # Use full precision (FP32) for evaluation!
    remove_unused_columns=False,
    prediction_loss_only=True,            # Only return loss (no logits) to avoid OOM
    eval_accumulation_steps=1             # Process one batch at a time
)

# -------------------------------
# 5. Define a Utility Function for Evaluation
# -------------------------------
def evaluate_model(model, description):
    trainer = Trainer(
        model=model,
        args=eval_training_args,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    # When prediction_loss_only=True, evaluate() returns only the loss.
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")
    if eval_loss is not None:
        try:
            perplexity = math.exp(eval_loss) if eval_loss < 10 else float("inf")
        except OverflowError:
            perplexity = float("inf")
        print(f"{description} Evaluation on Validation Set:")
        print(f"Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
    else:
        print(f"No evaluation loss returned for {description}.")

# -------------------------------
# 6. Load and Evaluate the Fine-Tuned Model and Baselines (Run only on Master Node)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluate Fine-Tuned Model
loaded_model = AutoModelForMaskedLM.from_pretrained(finetuned_model_dir)
loaded_model.to(device)
evaluate_model(loaded_model, "Fine-Tuned Model")

# Baseline 1: maastrichtlawtech/legal-camembert-base
baseline_model_1 = AutoModelForMaskedLM.from_pretrained("maastrichtlawtech/legal-camembert-base")
baseline_model_1.to(device)
evaluate_model(baseline_model_1, "Baseline Model 1 (maastrichtlawtech/legal-camembert-base)")

# Baseline 2: camembert-base (original pre-trained model)
baseline_model_2 = AutoModelForMaskedLM.from_pretrained("camembert-base")
baseline_model_2.to(device)
evaluate_model(baseline_model_2, "Baseline Model 2 (camembert-base)")