import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------
# 10. Test Final Model and Save Results
# -------------------------------

# # Define the directory where the model and tokenizer were saved
output_dir = "/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/fine_tuning_bloom_v5"

# # Define the model name (used in the results file)
model_name = "ft BLOOM 1B1 5 epochs cleaned data"

# # Load the fine-tuned model and tokenizer from the output directory
model = AutoModelForCausalLM.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)

# model_name_ = "bigscience/bloom-1b1"

# tokenizer = AutoTokenizer.from_pretrained(model_name_, cache_dir="/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/models_cache")
# model = AutoModelForCausalLM.from_pretrained(model_name_, cache_dir="/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/models_cache")

# Move the model to the appropriate device and set to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define a set of French legal prompts for testing
french_law_prompts = [
    "Quelles sont les conditions pour obtenir la nationalité française ? Réponse : ",
    "Expliquez le concept de responsabilité civile en droit français. Réponse : ",
    "Quels sont les droits du locataire en France ? Réponse : ",
    "Décrivez le processus de divorce en droit français. Réponse : ",
    "Quelles sont les sanctions pour fraude fiscale en France ? Réponse : ",
    "En quoi consiste la protection des consommateurs en droit français ? Réponse : ",
    "Comment fonctionne le droit du travail en France ? Réponse : ",
    "Quels sont les principes fondamentaux du droit administratif français ? Réponse : ",
    "Quels sont les droits et obligations des employeurs en France ? Réponse : ",
    "Expliquez la notion de contrat en droit français. Réponse : ",
]

# Define a prompt prefix to steer the model (you can adjust this)
prompt_prefix = (
    "Vous êtes un expert en droit français. "
    "Veuillez répondre de manière claire, précise et concise en français sans ajouter d'exemple. \n\n"
)

output_file = "results.txt"
with open(output_file, "a", encoding="utf-8") as f:
    f.write(f"Model Name: {model_name} \n")
    f.write("=" * 80 + "\n\n")
    print(f"Model Name: RAW BLOOM 1B1")

    for prompt in french_law_prompts:
        # Prepend the context prompt.
        full_prompt = prompt_prefix + prompt
        print("Generating response for prompt:", full_prompt)
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_length=250,          # You can adjust the max_length
            temperature=0.3,         # Lower temperature for more deterministic output
            top_k=50,                # Use top_k sampling
            top_p=0.95,              # Use nucleus sampling
            repetition_penalty=1.1,  # Slight repetition penalty
            do_sample=True,          # Enable sampling
        )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # Optionally, post-process the generated text to remove the prompt.
        generated_answer = generated_text[len(full_prompt):].strip()
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Réponse: {generated_answer}\n")
        f.write("-" * 80 + "\n\n")
        print("Response generated!")
