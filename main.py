import torch
from bert.bert_config import BertConfig
from bert.bert_architecture import BertModel
from bert.bert_trainer import BertTrainer
from utils.logger import RichLogger

logger = RichLogger("Main")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.new_run("Initializing BertConfig...")
    config = BertConfig()

    logger.step(1, "Creating BertModel...")
    model = BertModel(config)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {params}")
    logger.info(f"Model architecture: {model}")

    logger.step(2, "Testing model forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings))
    output = model(input_ids)
    logger.info(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
