import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Load data from file
file_path = Path(__file__).parent / 'evaluation_metrics.txt'
data = []
with file_path.open('r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)

# Plot Eval Loss separately
plt.figure(figsize=(12, 6))
plt.plot(df['iteration'], df['eval_loss'], label='Eval Loss', color='red')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Evaluation Loss over Iterations')
plt.legend()
plt.grid(True)
plt.show(block=False)  # Non-blocking show

# Plot Eval mIoU and Eval Accuracy separately
plt.figure(figsize=(12, 6))
plt.plot(df['iteration'], df['eval_miou'], label='Eval mIoU', color='blue')
plt.plot(df['iteration'], df['eval_acc'], label='Eval Accuracy', color='green')
plt.xlabel('Iteration')
plt.ylabel('Metrics')
plt.title('Evaluation mIoU and Accuracy over Iterations')
plt.legend()
plt.grid(True)
plt.show(block=False)  # Non-blocking show

# Keep the figures open until user closes them manually
input("Press Enter to close the plots...")
