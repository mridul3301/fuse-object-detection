import config
import torch
import torch.optim as optim

from tqdm import tqdm
import time

from model import YOLOv3
from loss import YoloLoss
from utils import *

# Instantiate the model
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

# Compile the model
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
loss_fn = YoloLoss()

# Scaler
scaler = torch.cuda.amp.GradScaler()

# Train-Test Loader
train_loader, test_loader = get_loaders(
    train_csv_path='/kaggle/input/pascalvoc-yolo/test.csv', test_csv_path='/kaggle/input/pascalvoc-yolo/test.csv'
)

# Anchors
scaled_anchors = (
    torch.tensor(config.ANCHORS) * torch.tensor([13,26,52]).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(config.DEVICE)



###################### TRAINING #############################

history_loss = [] # To plot the epoch vs. loss

for epoch in tqdm(range(config.NUM_EPOCHS), desc="Epochs"):
    model.train()

    losses = []

    start_time = time.time() # Start time of the epoch

    for batch_idx, (x,y) in enumerate(train_loader):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (y[0].to(config.DEVICE),
                    y[1].to(config.DEVICE),
                    y[2].to(config.DEVICE))

        # context manager is used in PyTorch to automatically handle mixed-precision computations on CUDA-enabled GPUs
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    end_time = time.time()  # End time of the epoch
    epoch_duration = end_time - start_time  # Duration of the epoch
        
    history_loss.append(sum(losses)/len(losses)) # Store for plotting loss 

    if (epoch+1) % 10 == 0:
        # Print the epoch duration
        tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

        # Print the loss and accuracy for training and validation data
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], "
            f"Loss: {sum(losses)/len(losses):.4f}")

        # save the model after every 10 epoch
        torch.save(model.state_dict(), f'/kaggle/working/Yolov3_epoch{epoch+1}.pth')



################ PLOT LOSSES ####################
import matplotlib.pyplot as plt
epochs = range(1, len(history_loss)+1)

# Plot losses
plt.plot(epochs, history_loss)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training Loss")
plt.show()