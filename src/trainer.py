import torch
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train(
    num_epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    save_path="experiments",
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + "/metrics.csv", "w") as file:
        file.write("epoch,train_loss,val_loss,duration\n")
        
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for seq_input, seq_target, target_mask in progress_bar:
                optimizer.zero_grad()
                outputs = model(seq_input)
                loss = compute_loss(outputs, seq_target, target_mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        val_loss = validate(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        end_time = time.time()
        duration = end_time - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss:.8f}, "
            f"Validation Loss: {val_loss:.8f}, "
            f"Duration: {duration:.2f}s"
        )

        # log metrics
        with open(save_path + "/metrics.csv", "a") as file:
            file.write(f"{epoch + 1},{train_loss},{val_loss},{duration}\n")

        #plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_path}/loss_curve.png")
        plt.close()

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path + "/model.pth")
            print(f"Model saved at epoch {epoch + 1} with Val Loss: {val_loss:.4f}")


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq_input, seq_target, target_mask in val_loader:
            outputs = model(seq_input)
            loss = compute_loss(outputs, seq_target, target_mask)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def eval(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for seq_input, seq_target, target_mask in test_loader:
            outputs = model(seq_input)
            loss = compute_loss(outputs, seq_target, target_mask)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")


def compute_loss(predictions, targets, masks):
    #only compute loss on non-padded elements
    masked_predictions = predictions[masks]
    masked_targets = targets[masks]
    return F.mse_loss(masked_predictions, masked_targets)
