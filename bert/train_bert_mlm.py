import argparse
import torch
from bert.bert_architecture import BertModel
from bert.bert_config import BertConfig
from bert.bert_trainer import BertTrainer
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from termcolor import colored

def main():
    parser = argparse.ArgumentParser(
        description=colored("Train BERT for Masked Language Modeling", "cyan", attrs=["bold"]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--tokenizer_name", type=str, required=True, help=colored("Name of the Hugging Face tokenizer", "yellow"))
    parser.add_argument("--dataset_name", type=str, required=True, help=colored("Name of the Hugging Face dataset", "yellow"))
    parser.add_argument("--vocab_size", type=int, required=True, help=colored("Vocabulary size of the tokenizer", "yellow"))
    parser.add_argument("--hidden_size", type=int, default=768, help=colored("Hidden size of the BERT model", "yellow"))
    parser.add_argument("--num_attention_heads", type=int, default=12, help=colored("Number of attention heads", "yellow"))
    parser.add_argument("--num_hidden_layers", type=int, default=12, help=colored("Number of hidden layers", "yellow"))
    parser.add_argument("--intermediate_size", type=int, default=3072, help=colored("Intermediate size of feedforward layers", "yellow"))
    parser.add_argument("--max_seq_length", type=int, default=128, help=colored("Maximum sequence length", "yellow"))
    parser.add_argument("--batch_size", type=int, default=32, help=colored("Batch size for training", "yellow"))
    parser.add_argument("--num_epochs", type=int, default=3, help=colored("Number of training epochs", "yellow"))
    parser.add_argument("--learning_rate", type=float, default=5e-5, help=colored("Learning rate for the optimizer", "yellow"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help=colored("Device to train on", "yellow"))
    args = parser.parse_args()

    # Configure BERT
    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-12
    )
    model = BertModel(config)

    # Prepare data
    train_loader, val_loader = BertTrainer.prepare_data(
        tokenizer_name=args.tokenizer_name,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length
    )

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = CrossEntropyLoss()

    # Initialize trainer
    trainer = BertTrainer(model, optimizer, loss_fn, train_loader, val_loader)

    # Train the model
    trainer.train(num_epochs=args.num_epochs, device=args.device)

if __name__ == "__main__":
    main()
