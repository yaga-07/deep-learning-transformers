from dl import BaseTrainer
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

class BertTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None):
        super().__init__(model, optimizer, loss_fn, train_loader, val_loader)

    @staticmethod
    def prepare_data(tokenizer_name, dataset_name, batch_size, max_seq_length):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        dataset = load_dataset(dataset_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_seq_length)

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

        train_loader = DataLoader(
            tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator
        )
        val_loader = DataLoader(
            tokenized_datasets["validation"], batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )
        return train_loader, val_loader

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

            if self.val_loader:
                self.validate(device)

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
