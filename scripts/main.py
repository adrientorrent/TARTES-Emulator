import time
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchinfo import summary
from utils.get_parquet_files import split_train_test_val
from dataset import TartesDataset
from model import TartesEmulator
from training import train

t0 = time.time()

# Datasets
# train_files, test_files, val_files = split_train_test_val(0.7, 0.2, 0.1)
# train_dataset = TartesDataset(parquet_files_groupby_year=train_files)
# test_dataset = TartesDataset(parquet_files_groupby_year=test_files)
# val_dataset = TartesDataset(parquet_files_groupby_year=val_files)

# Dataloaders
# batch_size = 16
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
# test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size)

# 1st batch
# for X_snowpack, X_sun, y in train_dataloader:
#     print("--- BATCH INFO ---")
#     print(f"SHAPES | X_snowpack: {X_snowpack.shape}, X_sun: {X_sun.shape}, y: {y.shape}")
#     print(f"SAMPLE 1 | {X_snowpack[0], X_sun[0], y[0]}")
#     print(f"SAMPLE 2 | {X_snowpack[1], X_sun[1], y[1]}")
#     print(f"SAMPLE 3 | {X_snowpack[2], X_sun[2], y[2]}")
#     break

# Model
tartes_model = TartesEmulator()
summary(
    tartes_model,
    input_size=[(1, 5, 50, 50), (1, 3)],
    col_names=["kernel_size", "input_size", "output_size", "num_params"]
)

# Loss, optimizer and metric
loss_fn = MSELoss()
optimizer = Adam(tartes_model.parameters())
metric = MeanSquaredError()

# Device
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"The following experiments will be launched on {DEVICE}")
tartes_model = tartes_model.to(DEVICE)
metric = metric.to(DEVICE)

# Forward pass
# for X_snowpack, X_sun, y in train_dataloader:
#     X_snowpack, X_sun = X_snowpack.to(DEVICE), X_sun.to(DEVICE)
#     y_hat = tartes_model(X_snowpack, X_sun)
#     print("--- FORWARD PASS ---")
#     print(f"TARTES ALBEDO: {y}")
#     print(f"MODEL PREDICTION: {y_hat}")
#     break

# Training
# epochs = 1
# for ep in range(epochs):
#     model = train(
#         train_dataloader=train_dataloader, 
#         test_dataloader=test_dataloader, 
#         model=tartes_model,
#         loss_function=loss_fn, 
#         optimizer=optimizer,
#         metric=metric,
#         device=DEVICE,
#         epoch=ep
#     )

# Save model parameters
PATH = "/home/torrenta/TARTES-Emulator/data/models/tartes-model.pt"
torch.save(tartes_model.state_dict(), PATH)

# Save full model ?

t1 = time.time()
print(f"Run time: {(t1 - t0):.2f} secondes")
