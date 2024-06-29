import epochs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helper_funcs import print_and_calc_metrics
epochs = np.array([0, 1, 2, 3, 4])
loss = np.array([0.6229950, 0.5394838, 0.4632986, 0.3917421, 0.3262813])
accuracy = np.array([0.029035409426498616, 0.029035409426498616, 0.029035409426498616, 0.029035409426498616, 0.029035409426498616])
precision = np.array([0.014517704713249308, 0.014517704713249308, 0.014517704713249308, 0.014517704713249308, 0.014517704713249308])
recall = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
f1 = np.array([0.0282161421857005, 0.0282161421857005, 0.0282161421857005, 0.0282161421857005, 0.0282161421857005])
uac = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
matrix = np.array([[209154, 166190], [30869, 400785]])

# Create a DataFrame
df = pd.DataFrame({
    'Epochs': epochs,
    'Loss': loss,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'UAC': uac,
})

fig, ax = plt.subplots(figsize=(20, 10)) # set size frame
ax.axis('off')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Create the table
tab = pd.plotting.table(ax, df, loc='center', cellLoc = 'center')

# Adjust the size of the table
tab.scale(1.5, 1.5)  # Increase the size of the table

tab.auto_set_font_size(False)
tab.set_fontsize(10)

# Print the DataFrame
print(df)

plt.tight_layout()
plt.savefig('dataframe.jpg')

# Assuming epochs, loss, accuracy_train, precision_train, recall_train, f1_train, uac_train are defined
# Plotting Loss vs. Epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, marker='o', label='Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plotting Accuracy, Precision, Recall, F1 Score, and UAC vs. Epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, accuracy, marker='o', label='Accuracy')
plt.plot(epochs, precision, marker='o', label='Precision')
plt.plot(epochs, recall, marker='o', label='Recall')
plt.plot(epochs, f1, marker='o', label='F1 Score')
plt.plot(epochs, uac, marker='o', label='UAC')
plt.title('Training Metrics vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.grid(True)

# Confusion matrices for specific epochs (10, 50, 100)
conf_matrices = {
    10: np.array([[209154, 166190], [30869, 400785]]),
    50: np.array([[346959, 28385], [33, 431621]]),
    100: np.array([[349143, 26201], [39, 431615]])
}

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

for i, (epoch, matrix) in enumerate(conf_matrices.items()):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axs[i])
    axs[i].set_title(f'Confusion Matrix at Epoch {epoch}')
    axs[i].set_xlabel('Predicted')
    axs[i].set_ylabel('Actual')

plt.show()
