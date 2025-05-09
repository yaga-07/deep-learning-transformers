from dl import BaseTrainer
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from dl.utils.training_report import TrainingRunReport

class BertMLMTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, report_path="bert_training_report.md"):
        super().__init__(model, optimizer, loss_fn, train_loader, val_loader)
        self.report = TrainingRunReport(model_name=model.__class__.__name__, output_path=report_path)

    def train(self, num_epochs, device):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
        self.model.to(device)
        self.model.train()

        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0
            for batch in tqdm(self.train_loader, desc="Training"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            logging.info(f"Training Loss: {avg_loss:.4f}")

            avg_val_loss = None
            if self.val_loader:
                avg_val_loss = self.validate(device)

            # Log metrics to the report
            self.report.log_epoch(epoch + 1, avg_loss, avg_val_loss)

        # Save the Markdown report after training
        self.report.save_report()

    def validate(self, device):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
