import math
import pandas as pd

df = pd.read_csv("lightning_logs/version_1/metrics.csv")

epoch_index = 0
for val_loss in df["val_loss"]:
    if not math.isnan(val_loss):
        print(f"Epoch {epoch_index}, Loss: {val_loss}")
        epoch_index += 1
