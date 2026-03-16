import torch
import torch.optim as optim
import copy
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MiniGPT
from dataset import DPODataset, RAW_DATA

# 1. Helper Function: Extract log probabilities of the actual target tokens
def get_logprobs(logits, labels, pad_token_id):
    # logits shape: (batch, seq_len, vocab_size)
    # labels shape: (batch, seq_len)
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather the log probabilities of the specific tokens in the labels
    token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out the padding tokens so they don't affect the loss
    mask = (shift_labels != pad_token_id).float()
    
    # Sum log probs over the sequence
    return (token_log_probs * mask).sum(dim=-1)

# 2. The Core DPO Loss Function
def micro_dpo_loss(policy_chosen_logits, policy_rejected_logits,
                   ref_chosen_logits, ref_rejected_logits,
                   chosen_labels, rejected_labels, pad_token_id, beta=0.1):
    
    policy_chosen_logps = get_logprobs(policy_chosen_logits, chosen_labels, pad_token_id)
    ref_chosen_logps = get_logprobs(ref_chosen_logits, chosen_labels, pad_token_id)

    policy_rejected_logps = get_logprobs(policy_rejected_logits, rejected_labels, pad_token_id)
    ref_rejected_logps = get_logprobs(ref_rejected_logits, rejected_labels, pad_token_id)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    margin = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(margin).mean()

    return loss, margin.mean().item()

# 3. Training Loop
def main():
    dataset = DPODataset(RAW_DATA)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    pad_id = dataset.stoi['<PAD>']

    # Initialize the model we want to train (The Policy)
    policy_model = MiniGPT(vocab_size=dataset.vocab_size)
    optimizer = optim.AdamW(policy_model.parameters(), lr=1e-3)

    # In a real scenario, this model would already be pre-trained on text completion.
    # For this micro-example, we start from scratch, but we STILL freeze a copy to act as the reference.
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval() # Freeze the reference model

    print("Starting microDPO Alignment...")
    epochs = 1000
    
    metrics = {'epochs': [], 'losses': [], 'margins': []}

    for epoch in range(epochs):
        total_loss = 0
        total_margin = 0
        for batch in dataloader:
            chosen = batch['chosen']
            rejected = batch['rejected']
            
            optimizer.zero_grad()
            
            # Forward pass on Policy Model
            policy_chosen_logits = policy_model(chosen)
            policy_rejected_logits = policy_model(rejected)
            
            # Forward pass on Reference Model (No gradients!)
            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen)
                ref_rejected_logits = ref_model(rejected)
                
            loss, margin = micro_dpo_loss(
                policy_chosen_logits, policy_rejected_logits,
                ref_chosen_logits, ref_rejected_logits,
                chosen, rejected, pad_token_id=pad_id, beta=0.1
            )
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_margin += margin
            
        avg_loss = total_loss / len(dataloader)
        avg_margin = total_margin / len(dataloader)
        
        metrics['epochs'].append(epoch)
        metrics['losses'].append(avg_loss)
        metrics['margins'].append(avg_margin)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch} | DPO Loss: {avg_loss:.10f} | Margin: {avg_margin:.10f}")

    import json
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    # Save the aligned model
    torch.save(policy_model.state_dict(), 'aligned_micro_gpt.pth')
    print("Training complete. Model saved to 'aligned_micro_gpt.pth'")

if __name__ == "__main__":
    main()