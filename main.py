import yaml
import torch
from data.dataset import get_dataloaders
from models.moe import MoE
from training.trainer import Trainer
from logsys.app_logger import get_logger
from transformers import AutoTokenizer

def main():
    config_path = 'configs/config.yaml'  # Path to your YAML configuration file
    
    # Load the configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize the logger
    logger = get_logger('MoE_Trainer', 'train.log')
    logger.info("Starting training process")
    
    # Set device based on configuration
    if config['device']['use_gpu']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Using GPU for training.")
        else:
            device = torch.device('cpu')
            logger.warning("GPU requested but not available. Using CPU instead.")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for training.")
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Example tokenizer
    
    # Get the data loaders
    train_loader, val_loader = get_dataloaders(config, tokenizer)
    
    # Initialize the model
    model = MoE(
        num_experts=config['model']['num_experts'],
        expert_hidden_size=config['model']['expert_hidden_size'],
        gate_hidden_size=config['model']['gate_hidden_size'],
        input_size=config['data']['max_sequence_length'],
        dropout=config['model']['dropout'],
        device=device
    )
    model.to(device)
    
    # Initialize the trainer
    trainer = Trainer(model, train_loader, val_loader, config, logger, device)
    trainer.train()
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()