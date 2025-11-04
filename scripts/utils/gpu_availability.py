import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Les calculs seront lancés sur {DEVICE}")
