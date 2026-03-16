import torch
from torch.utils.data import Dataset, DataLoader

# 1. Our highly engineered, massive, web-scale dataset (just kidding)
# Format: (Prompt, Chosen Response, Rejected Response)
RAW_DATA = [
    ("Help me fix this bug.", "I would be happy to help you look into this issue.", "Fix it yourself, I'm busy."),
    ("What is 2+2?", "The answer to 2+2 is 4.", "Are you serious? It's obviously 4."),
    ("I don't understand this code.", "Let's break it down step by step.", "That's because you are not very smart."),
    ("Review my pull request.", "I will review your PR shortly, thanks!", "Your code is terrible, denied."),
    ("Good morning!", "Good morning! How can I assist you today?", "What's so good about it?"),
]

class DPODataset(Dataset):
    def __init__(self, data, max_length=64):
        self.data = data
        self.max_length = max_length
        
        # 2. Build a character-level vocabulary from scratch
        # We join all text, find unique characters, and sort them
        chars = set()
        for prompt, chosen, rejected in data:
            chars.update(list(prompt + chosen + rejected))
        self.chars = sorted(list(chars))
        
        # Add special tokens for padding and structural separation
        self.chars = ['<PAD>', '<SEP>'] + self.chars
        self.vocab_size = len(self.chars)
        
        # Create mapping dictionaries
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        """Convert a string to a list of integers."""
        tokens = []
        i = 0
        while i < len(text):
            if text.startswith('<SEP>', i):
                tokens.append(self.stoi['<SEP>'])
                i += 5
            elif text.startswith('<PAD>', i):
                tokens.append(self.stoi['<PAD>'])
                i += 5
            else:
                tokens.append(self.stoi[text[i]])
                i += 1
        return tokens

    def decode(self, tokens):
        """Convert an iterable of integers back to a string."""
        return ''.join([self.itos[i] for i in tokens])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt_str, chosen_str, rejected_str = self.data[idx]
        
        # Format the text so the model knows what is the prompt and what is the response
        # Example: "Help me fix this bug.<SEP>I would be happy to help..."
        chosen_full = prompt_str + '<SEP>' + chosen_str
        rejected_full = prompt_str + '<SEP>' + rejected_str
        
        # Encode to integers
        chosen_tokens = self.encode(chosen_full)
        rejected_tokens = self.encode(rejected_full)
        
        # Pad sequences to max_length so we can batch them in tensors
        chosen_padded = chosen_tokens + [self.stoi['<PAD>']] * (self.max_length - len(chosen_tokens))
        rejected_padded = rejected_tokens + [self.stoi['<PAD>']] * (self.max_length - len(rejected_tokens))
        
        # Truncate if they somehow exceed max_length
        chosen_padded = chosen_padded[:self.max_length]
        rejected_padded = rejected_padded[:self.max_length]
        
        return {
            'chosen': torch.tensor(chosen_padded, dtype=torch.long),
            'rejected': torch.tensor(rejected_padded, dtype=torch.long)
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    # Test the dataset and dataloader
    dataset = DPODataset(RAW_DATA)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(f"Vocabulary Size: {dataset.vocab_size}")
    print(f"Characters: {''.join(dataset.chars[2:])}") # Skip special tokens for printing
    
    # Grab one batch
    batch = next(iter(dataloader))
    
    print("\n--- Batch Shape ---")
    print(f"Chosen Tensor Shape: {batch['chosen'].shape}")
    print(f"Rejected Tensor Shape: {batch['rejected'].shape}")
    
    print("\n--- Decoding the first item in the batch ---")
    print("CHOSEN:   ", dataset.decode(batch['chosen'][0].tolist()).replace('<PAD>', ''))
    print("REJECTED: ", dataset.decode(batch['rejected'][0].tolist()).replace('<PAD>', ''))