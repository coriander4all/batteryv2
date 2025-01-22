import matplotlib.pyplot as plt
import pandas as pd

path = "experiments/1/"
file = "metrics.csv"
data = pd.read_csv(path + file)

plt.plot(data["train_loss"], label="Train Loss")
plt.plot(data["val_loss"], label="Validation Loss")
plt.legend()
plt.title("MSE Loss")
plt.savefig(path + "loss.png")
