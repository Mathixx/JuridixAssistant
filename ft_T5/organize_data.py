import pandas as pd
import json
import os

# Define file paths
json_file = "ft_T5/data/reformulation_dataset_intermediate_5350.json"
train_csv_file = "ft_T5/train_dataset.csv"
val_csv_file = "ft_T5/val_dataset.csv"
test_csv_file = "ft_T5/test_dataset.csv"

# Load JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert JSON data into a DataFrame
df = pd.DataFrame([
    {
        "initial_query": entry["input"],
        "type": entry["output"]["type"],
        "new_query": entry["output"]["reformulated_query"]
    }
    for entry in data
])

# Print basic information about the dataset
print("Total examples:", len(df))
print("Examples per class:\n", df["type"].value_counts())

# We'll perform a stratified split per class:
#   - 10% for test,
#   - From the remaining 90%, 20/90 (~22.22%) for validation (i.e., overall 20%)
#   - The rest (~70%) for training.

train_df_list = []
val_df_list = []
test_df_list = []

# Group by the 'type' column to maintain class proportions
for doc_type, group in df.groupby("type"):
    # Sample 10% for test set
    test_group = group.sample(frac=0.1, random_state=42)
    
    # Remove test_group from the group to get the remaining data (90%)
    remaining = group.drop(test_group.index)
    
    # From the remaining, sample 20/90 (~22.22%) for validation
    val_group = remaining.sample(frac=0.1, random_state=42)
    
    # The rest becomes the training set
    train_group = remaining.drop(val_group.index)
    
    # Append each split to its respective list
    test_df_list.append(test_group)
    val_df_list.append(val_group)
    train_df_list.append(train_group)

# Concatenate groups to form complete DataFrames for each split
train_df = pd.concat(train_df_list).reset_index(drop=True)
val_df = pd.concat(val_df_list).reset_index(drop=True)
test_df = pd.concat(test_df_list).reset_index(drop=True)

# Verify proportions per class in each dataset
print("Train set examples per class:\n", train_df["type"].value_counts())
print("Validation set examples per class:\n", val_df["type"].value_counts())
print("Test set examples per class:\n", test_df["type"].value_counts())

# Save DataFrames to CSV files
train_df.to_csv(train_csv_file, index=False, encoding="utf-8")
val_df.to_csv(val_csv_file, index=False, encoding="utf-8")
test_df.to_csv(test_csv_file, index=False, encoding="utf-8")

print(f"Train set saved to {os.path.abspath(train_csv_file)}")
print(f"Validation set saved to {os.path.abspath(val_csv_file)}")
print(f"Test set saved to {os.path.abspath(test_csv_file)}")
