import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Define the file path using pathlib
file_path = Path(__file__).parent / 'evaluation_metrics.txt'

# Load the file and parse JSON lines
data = []
with file_path.open('r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Convert to DataFrame
df = pd.DataFrame(data)

# Figure 1
plt.figure(figsize=(8, 6))
plt.plot(df['iteration'], df['eval_loss'], label='Eval Loss')
plt.plot(df['iteration'], df['eval_loss_all'], label='Eval Loss (All)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()
plt.grid(True)
plt.ylim(0, 0.1)
# Figure 2
plt.figure(figsize=(8, 6))
plt.plot(df['iteration'], df['eval_acc'], label='Eval Accuracy')
plt.plot(df['iteration'], df['eval_acc_all'], label='Eval Accuracy (All)')
plt.plot(df['iteration'], df['eval_miou'], label='Eval mIoU')
plt.plot(df['iteration'], df['eval_miou_all'], label='Eval mIoU (All)')
plt.xlabel('Iteration')
plt.ylabel('Metrics')
plt.title('Accuracy and mIoU over Iterations')
plt.legend()
plt.grid(True)
plt.ylim(80, 100)
# Show both figures at once
plt.show()
