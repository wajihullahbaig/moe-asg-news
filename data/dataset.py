import torch
from datasets import load_dataset
from transformers import AutoTokenizer

class AGNewsTinyDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, max_length, split='train', num_samples=100):
        """
        Initializes the AG News Tiny dataset with a tokenizer, maximum sequence length, and a subset size.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            max_length (int): The maximum sequence length.
            split (str): The dataset split to use ('train', 'test').
            num_samples (int): The number of samples to include in the subset.
        """
        # Load the AG News dataset
        self.dataset = load_dataset('ag_news', split=split)
        
        # Create a tiny subset
        self.dataset = self.dataset.select(range(min(num_samples, len(self.dataset))))
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the text and label
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        
        # Text Cleaning
        text = self.clean_text(text)
        
        # Tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        
        return {
            'input_ids': encoding['input_ids'].squeeze().float(),  # Removes the batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)  # Ensure label is a long tensor
        }

    def clean_text(self, text):
        """
        Cleans the input text by lowercasing, removing HTML tags, URLs, and special characters.

        Args:
            text (str): The input text.

        Returns:
            str: The cleaned text.
        """
        # Lowercasing
        text = text.lower()
        
        # Remove HTML tags
        import re
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers (optional)
        # Uncomment the next line if you want to remove special characters and numbers
        # text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        return text

def get_dataloaders(config, tokenizer):
    """
    Creates training and validation data loaders.

    Args:
        config (dict): Configuration dictionary containing dataset parameters.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        tuple: Tuple containing (train_loader, val_loader).
    """
    # Load the dataset
    dataset = AGNewsTinyDataset(
        tokenizer=tokenizer,
        max_length=config['data']['max_sequence_length'],
        split='train',
        num_samples=100  # Adjust the number of samples as needed
    )
    
    # Split the dataset into training and validation sets
    # Here, we'll use a simple 90-10 split for demonstration
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader