import json
import matplotlib.pyplot as plt

def save_loss_plot(json_file_path, output_image_path):
    """
    Reads a JSON file containing training logs, extracts the loss progression over epochs,
    creates a plot, and saves the plot to the specified output path.
    
    Parameters:
        json_file_path (str): The path to the JSON file.
        output_image_path (str): The path where the plot image will be saved.
    """
    # Load JSON data from file
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    # Extract log history
    log_history = data.get("log_history", [])
    
    # Extract epoch and loss values, ensuring both keys exist in each entry
    epochs = [entry["epoch"] for entry in log_history if "epoch" in entry and "loss" in entry]
    losses = [entry["loss"] for entry in log_history if "epoch" in entry and "loss" in entry]
    
    # Create a sorted progression by epoch if necessary
    progression = sorted(zip(epochs, losses), key=lambda x: x[0])
    epochs, losses = zip(*progression)  # unzip the sorted progression
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', linestyle='-', label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Progression Over Epochs")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the specified file
    plt.savefig(output_image_path)
    plt.close()  # close the plot to free memory

# Example usage:
if __name__ == "__main__":
    json_path = "/users/eleves-b/2022/mathias.perez/Desktop/JuridixAssistant/fine_tuning_bloom/checkpoint-2756945/trainer_state.json"   # Replace with your JSON file path
    image_path = "loss_progression_final.png"        # Desired output image file
    save_loss_plot(json_path, image_path)
    print(f"Plot saved to {image_path}")
