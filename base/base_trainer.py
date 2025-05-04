import torch
from torch.utils.data import DataLoader
from utils.logger import get_logger

class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader: DataLoader, val_loader: DataLoader = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = get_logger("Trainer")

    def train(self, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            for batch in self.train_loader:
                inputs, labels = batch
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.logger.info(f"Loss: {loss.item():.4f}")
