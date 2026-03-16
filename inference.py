import torch
from model import MiniGPT
from dataset import DPODataset, RAW_DATA

def main():
    # Rebuild dataset to get the exact vocabulary mapping
    dataset = DPODataset(RAW_DATA)
    
    # Load the aligned model
    model = MiniGPT(vocab_size=dataset.vocab_size)
    try:
        model.load_state_dict(torch.load('aligned_micro_gpt.pth'))
        print("Model loaded successfully.\n")
    except FileNotFoundError:
        print("Please run train.py first to generate the aligned model weights.")
        return

    model.eval()

    # The prompt we want to test
    test_prompt = "Help me fix this bug."
    
    # Format the prompt exactly how the model expects to see it before answering
    formatted_prompt = test_prompt + "<SEP>"
    
    # Encode the prompt using the dataset's vocabulary
    input_ids = torch.tensor([dataset.encode(formatted_prompt)], dtype=torch.long)
    
    print(f"User: {test_prompt}")
    print("Generating response...")
    
    # Generate 40 new tokens
    generated_ids = model.generate(input_ids, max_new_tokens=40)
    
    # Decode the output, stripping the original prompt and padding
    output_text = dataset.decode(generated_ids[0].tolist())
    response_only = output_text.split("<SEP>")[-1].replace('<PAD>', '')
    
    print(f"\nAligned Model: {response_only}")

if __name__ == "__main__":
    main()