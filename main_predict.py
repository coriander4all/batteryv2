import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from src.dataset import SOHDataset
from src.dataloader import LSTM_SOHPredictor
from src.explainer import LSTMExplainer
import numpy as np

from hyperparams import *

# paths
test_csv = "data/processed2/test.csv"
model_weights_path = "experiments/1/model.pth"

scaler = MinMaxScaler()
test_dataset = SOHDataset(test_csv, seq_length, pred_length, scaler, "test")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LSTM_SOHPredictor(input_size, hidden_size, num_layers, pred_length)
model.load_state_dict(torch.load(model_weights_path, weights_only=True))
model.eval()

def plot_curve_predictions(id: int, true_values, predicted_values, masks):
    samples_path = "predictions"
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)
    

    mask = masks[id]
    actual = true_values[id][mask]
    pred = predicted_values[id][mask]
    
    
    print(f"\nSequence {id}:")
    print(f"Number of valid points: {sum(mask)}")
    print(f"Total sequence length: {len(mask)}")
    print(f"First few actual values: {actual[:5]}")
    print(f"First few predicted values: {pred[:5]}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="True values", marker='o', markersize=2)
    plt.plot(pred, label="Predicted values", marker='x', markersize=2)
    plt.legend()
    plt.title(f'SOH Prediction - Sequence {id}\n(Valid points: {sum(mask)})')
    plt.xlabel('Cycles')
    plt.ylabel('SOH')
    plt.grid(True)
    plt.savefig(f"{samples_path}/predictions{id}.png")
    plt.close()

def main():
    # Load multiple batches to see more variety
    all_predictions = []
    all_targets = []
    all_masks = []
    
    
    for i, (test_seq, test_target, test_mask) in enumerate(test_loader):
        if i >= 1:  #examples from one batch
            break
            
        with torch.no_grad():
            predicted = model(test_seq)
        

        all_predictions.append(predicted.numpy())
        all_targets.append(test_target.numpy())
        all_masks.append(test_mask.numpy())
    

    predicted = np.concatenate(all_predictions)
    true_values = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)
    
    print(f"\nTotal sequences loaded: {len(true_values)}")
    print(f"Prediction shape: {predicted.shape}")
    print(f"True values shape: {true_values.shape}")
    print(f"Masks shape: {masks.shape}")

    for i in range(30): 
        plot_curve_predictions(i, true_values, predicted, masks)

    print("\nshap analysis...")
    
    # shap form 50 tests
    background_data = torch.stack([test_dataset[i][0] for i in range(min(50, len(test_dataset)))])


    explainer = LSTMExplainer(model, background_data)

    feature_names = [
        "Current", "Voltage", "Temperature", "Capacity",
        "Resistance", "Power", "Energy", "Efficiency",
        "Cycle_time", "Charge_time", "Discharge_time",
        "Rest_time", "SOH"
    ]

    # A sequence to eplain
    test_seq, _, _ = next(iter(test_loader))
    sequence_to_explain = test_seq[0:1]

    print("Calculating SHAP values...")
    shap_values = explainer.explain_prediction(
        sequence_to_explain,
        feature_names=feature_names
    )
    print("\nSHAP: shap_summary.png and shap_summary_bar.png")

if __name__ == "__main__":
    main()