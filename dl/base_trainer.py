import torch
from torch.utils.data import DataLoader
from .utils.logger import get_logger

class BaseTrainer:
    """
    BaseTrainer is a utility class for training deep learning models. It handles
    the training loop, logging, and optionally supports validation.

    Attributes:
        model: The model to be trained.
        optimizer: The optimizer used for training.
        loss_fn: The loss function used for optimization.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader, optional): DataLoader for the validation dataset.
        logger: Logger instance for logging training progress.
    """

    def __init__(self, model, optimizer, loss_fn, train_loader: DataLoader, val_loader: DataLoader = None):
        """
        Initialize the BaseTrainer with model, optimizer, loss function, and data loaders.

        Args:
            model: The model to be trained.
            optimizer: The optimizer used for training.
            loss_fn: The loss function used for optimization.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader, optional): DataLoader for the validation dataset. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = get_logger("Trainer")

    def train(self, epochs: int):
        """
        Perform the training loop for a specified number of epochs.

        Args:
            epochs (int): The number of epochs to train the model.
        """
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
