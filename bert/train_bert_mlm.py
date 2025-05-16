import argparse
import torch
import yaml
from bert.bert_architecture import BertModel
from bert.bert_config import BertConfig
from bert.bert_trainer import BertMLMTrainer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from termcolor import colored
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from bert.bert_dataloader import JsonlDataset  # add import
from dl.utils.utils import check_device_and_packages


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

def prepare_jsonl_data(tokenizer_name, train_path, val_path, batch_size, max_seq_length):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = JsonlDataset(train_path, tokenizer, max_seq_length)
    val_dataset = JsonlDataset(val_path, tokenizer, max_seq_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(
        description=colored("Train BERT for Masked Language Modeling", "cyan", attrs=["bold"]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_path", type=str, required=True, help=colored("Path to the YAML configuration file", "yellow"))
    args = parser.parse_args()

    # Load configuration from YAML
    with open(args.config_path, "r") as file:
        config_data = yaml.safe_load(file)

    # Device/package check
    device = config_data["training"]["device"]
    check_device_and_packages(device)

    # Configure BERT
    model_config = BertConfig(**config_data["model"])
    model = BertModel(model_config)

    # Prepare data
    if config_data["training"].get("train_jsonl_path", False):
        train_loader, val_loader = prepare_jsonl_data(
            tokenizer_name=config_data["training"]["tokenizer_name"],
            train_path=config_data["training"]["train_jsonl_path"],
            val_path=config_data["training"]["val_jsonl_path"],
            batch_size=config_data["training"]["batch_size"],
            max_seq_length=config_data["training"]["max_seq_length"]
        )
    else:
        train_loader, val_loader = prepare_data(
            tokenizer_name=config_data["training"]["tokenizer_name"],
            dataset_name=config_data["training"]["dataset_name"],
            batch_size=config_data["training"]["batch_size"],
            max_seq_length=config_data["training"]["max_seq_length"]
        )

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=float(config_data["training"]["learning_rate"]))
    loss_fn = CrossEntropyLoss()

    # Initialize trainer
    trainer = BertMLMTrainer(model, optimizer, loss_fn, train_loader, val_loader)

    # Train the model
    trainer.train(num_epochs=config_data["training"]["num_epochs"], device=config_data["training"]["device"])

    # Save the trained model
    if config_data["training"].get("save_model_path", False):
        import os
        save_path = config_data["training"]["save_model_path"]
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        print(colored(f"Saving model to {save_path}", "green"))
        model.save_pretrained(save_path)

if __name__ == "__main__":
    main()
