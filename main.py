import torch
from bert.bert_config import BertConfig
from bert.bert_architecture import BertModel
from bert.bert_trainer import BertTrainer
from utils.logger import get_logger

logger = get_logger("Main")

def main():
    logger.info("Initializing BertConfig...")
    config = BertConfig()

    logger.info("Creating BertModel...")
    model = BertModel(config)

    logger.info("Testing model forward pass...")
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings))
    output = model(input_ids)
    logger.info(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
