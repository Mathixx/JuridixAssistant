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

train_dataset = train_dataset.with_transform(transform)
val_dataset = val_dataset.with_transform(transform)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# -------------------------------
# 3. Fine-Tuning with LoRA and Masked Language Modeling
# -------------------------------
model = AutoModelForMaskedLM.from_pretrained(model_name)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model is on device:", device)
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# -------------------------------
# 4. Training Setup (No Evaluation During Training)
# -------------------------------
# Here we disable evaluation by setting evaluation_strategy to "no". We also remove callbacks
# and compute_metrics to avoid any evaluation-related GPU memory usage.
training_args = TrainingArguments(
    output_dir="./results_ddp",
    num_train_epochs=3,
    per_device_train_batch_size=16,   # Reduced batch size to help with memory
    evaluation_strategy="no",          # Disable evaluation during training
    save_strategy="steps",
    save_steps=500,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=2,
    fp16=True,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# -------------------------------
# 5. Training and Saving with Automatic Checkpoint Resumption
# -------------------------------
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if last_checkpoint is not None:
    print("Resuming training from checkpoint:", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

trainer.save_model("./lora_finetuned_legifrance_camembert_mlm")

# -------------------------------
# 6. Evaluation on the Validation Set (Run only on Master Node)
# -------------------------------
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    # Create new training arguments for evaluation with a very small eval batch size to avoid OOM.
    eval_training_args = TrainingArguments(
        output_dir="./results_ddp_eval",
        per_device_eval_batch_size=1,  # Use a minimal batch size for evaluation
        evaluation_strategy="no",
        fp16=True,
        remove_unused_columns=False
    )
    
    # Create a new Trainer for evaluation that includes our evaluation dataset.
    eval_trainer = Trainer(
        model=model,
        args=eval_training_args,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # Perform evaluation and compute loss, then compute perplexity.
    eval_results = eval_trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", None)
    if eval_loss is not None:
        perplexity = math.exp(eval_loss) if eval_loss < 10 else float("inf")
        print("Fine-Tuned Model Evaluation on Legifrance Held-Out Data:")
        print(f"Eval Loss: {eval_loss:.4f}, Perplexity: {perplexity:.4f}")
    else:
        print("No evaluation loss returned.")

    # Optionally, evaluate the baseline model similarly:
    baseline_model_name = "maastrichtlawtech/legal-camembert-base"
    baseline_model = AutoModelForMaskedLM.from_pretrained(baseline_model_name)
    baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model_name)
    baseline_model.to(device)
    
    baseline_data_collator = DataCollatorForLanguageModeling(
        tokenizer=baseline_tokenizer, mlm=True, mlm_probability=0.15
    )
    
    baseline_eval_trainer = Trainer(
        model=baseline_model,
        args=eval_training_args,
        eval_dataset=val_dataset,
        data_collator=baseline_data_collator
    )
    
    baseline_eval_results = baseline_eval_trainer.evaluate()
    baseline_eval_loss = baseline_eval_results.get("eval_loss", None)
    if baseline_eval_loss is not None:
        baseline_perplexity = math.exp(baseline_eval_loss) if baseline_eval_loss < 10 else float("inf")
        print("Baseline Model Evaluation on Legifrance Held-Out Data:")
        print(f"Eval Loss: {baseline_eval_loss:.4f}, Perplexity: {baseline_perplexity:.4f}")
    else:
        print("No baseline evaluation loss returned.")