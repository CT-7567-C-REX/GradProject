# to get graph run this file alone. 

import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

file_path = Path(__file__).parent / 'evaluation_metrics.txt' 

data = []
with file_path.open('r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))

plt.plot(df['iteration'], df['eval_loss'], label='Eval Loss')
plt.plot(df['iteration'], df['eval_miou'], label='Eval mIoU')
plt.plot(df['iteration'], df['eval_acc'], label='Eval Accuracy')

plt.xlabel('Iteration')
plt.ylabel('Metrics')
plt.title('Evaluation Metrics over Iterations')
plt.legend()
plt.grid(True)
plt.show()
