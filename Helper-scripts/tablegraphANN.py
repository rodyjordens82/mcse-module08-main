import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

epochs = np.array([10, 20, 30, 40, 50, 60, 70 , 80, 90, 100, 110, 120, 130, 140])
accuracy = np.array([0.9631676072849007, 0.9740246141920832, 0.9944197216534877, 0.994442152346324, 0.994442336204462, 0.9944425200626, 0.9944426119916691, 0.994442428133531, 0.994442428133531, 0.9944443586439801, 0.9944443586439801, 0.9944443586439801, 0.994444542502118, 0.994446381083498])
precision = np.array([0.8482506287727167, 0.9868529871099989, 0.9970728710284689, 0.9970858388895905, 0.9970816669538975, 0.9970732289974222, 0.9970718543140903, 0.9970703375690754, 0.9970674935637434, 0.9970628009207617, 0.9970613791040391, 0.9970613791040391, 0.9970586303257456, 0.9970652652120747])
recall = np.array([0.5000135361866179, 0.6473949092935294, 0.9242886105547823, 0.9245919018602897, 0.924597998273317, 0.9246076952674112, 0.9246101433770805, 0.9246088477388088, 0.9246112481261866, 0.924642255136529, 0.9246434553302179, 0.9246434553302179, 0.9246483515495563, 0.9246685090944069])
f1 = np.array([0.49064658555983565, 0.7210198702602241, 0.9575796562298327, 0.9577634835061517, 0.9577653206571195, 0.9577674876945659, 0.9577683512415933, 0.957766954068624, 0.9577671740175722, 0.9577834382622183, 0.9577835481889838, 0.9577835481889838, 0.9577852750641047, 0.9577999053453268])
auc = np.array([0.5000135361866179, 0.6473949092935294, 0.9242886105547823, 0.9245919018602896, 0.924597998273317, 0.9246076952674113, 0.9246101433770805, 0.9246088477388088, 0.9246112481261866, 0.9246422551365289, 0.9246434553302179, 0.9246434553302179, 0.9246483515495563, 0.9246685090944069])
matrix = np.array([[209154, 166190], [30869, 400785]])

# Create a DataFrame
df = pd.DataFrame({
    'Epochs': epochs,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1,
    'AUC': auc,
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

plt.figure(figsize=(10, 6))
plt.scatter(df['Epochs'], df['Accuracy'], label='Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plotting Accuracy, Precision, Recall, F1 Score, and UAC vs. Epochs
plt.figure(figsize=(12, 8))
plt.plot(epochs, accuracy, marker='o', label='Accuracy')
plt.plot(epochs, precision, marker='o', label='Precision')
plt.plot(epochs, recall, marker='o', label='Recall')
plt.plot(epochs, f1, marker='o', label='F1 Score')
plt.plot(epochs, auc, marker='o', label='AUC')
plt.title('ANN training')
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
