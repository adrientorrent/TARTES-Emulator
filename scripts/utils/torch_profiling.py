#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity


class RandomImageDataset(Dataset):

    def __init__(self, n_samples: int = 1024, n_classes: int = 10) -> None:
        super().__init__()
        self.x = torch.randn(n_samples, 3, 32, 32)
        self.y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class EasyCNN(nn.Module):

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset = RandomImageDataset()
loader = DataLoader(dataset, batch_size=64)

model = EasyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

logdir = "./logs"

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA if torch.cuda.is_available()
        else ProfilerActivity.CPU
    ],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
    record_shapes=False,
    profile_memory=False,
    with_stack=False
) as prof:

    model.train()
    for step, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with record_function("forward"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        with record_function("backward"):
            loss.backward()
        optimizer.step()
        prof.step()
        if step % 2 == 0:
            print(f"step={step}, loss={loss.item():.4f}")

print(f"Logs dans {logdir}")
