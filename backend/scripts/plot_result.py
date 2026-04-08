import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD THE DATA
# This assumes your training loop from Day 3 created 'training_logs.csv'
try:
    data = pd.read_csv('day5_augmented_logs.csv')
    print("Successfully loaded training logs!")
except FileNotFoundError:
    print("Error: 'training_logs.csv' not found. Please finish training first.")
    exit()

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
epochs = range(1, len(data) + 1)

# 2. GENERATE THE LOSS GRAPH (Decreasing)
plt.figure(figsize=(10, 5))
plt.plot(epochs, data['loss'], 'bo-', label='Training Loss')
plt.plot(epochs, data['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss (Decreasing)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_graph.png', dpi=300) # Save high-res for paper
print("- Saved loss_graph.png")

# 3. GENERATE THE ACCURACY GRAPH (Increasing)
plt.figure(figsize=(10, 5))
plt.plot(epochs, data['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs, data['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy (Increasing)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_graph.png', dpi=300)
print("- Saved accuracy_graph.png")

# 4. GENERATE THE AUC GRAPH (Most important for your Paper!)
if 'auc' in data.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, data['auc'], 'bo-', label='Training AUC')
    plt.plot(epochs, data['val_auc'], 'ro-', label='Validation AUC')
    plt.title('Training and Validation AUC (Area Under Curve)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig('auc_graph.png', dpi=300)
    print("- Saved auc_graph.png")

plt.show()