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

    # Configure BERT
    model_config = BertConfig(**config_data["model"])
    model = BertModel(model_config)

    # Prepare data
    train_loader, val_loader = prepare_data(
        tokenizer_name=config_data["model"]["model_type"],
        dataset_name=config_data["training"]["dataset_name"],
        batch_size=config_data["training"]["batch_size"],
        max_seq_length=config_data["training"]["max_seq_length"]
    )

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=config_data["training"]["learning_rate"])
    loss_fn = CrossEntropyLoss()

    # Initialize trainer
    trainer = BertMLMTrainer(model, optimizer, loss_fn, train_loader, val_loader)

    # Train the model
    trainer.train(num_epochs=config_data["training"]["num_epochs"], device=config_data["training"]["device"])

    # Save the trained model
    model.save_pretrained(config_data["training"]["save_model_path"])

if __name__ == "__main__":
    main()
