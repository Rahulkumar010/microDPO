# microDPO 

**The simplest, fastest repository for understanding and training Direct Preference Optimization (DPO) from scratch.**



Currently, if a student or researcher wants to learn how to align a language model (teaching it to prefer a "helpful" response over a "toxic" one), they are forced to navigate massive, abstracted libraries like Hugging Face's `TRL` (Transformer Reinforcement Learning). 

`microDPO` strips away all the production bloat, distributed computing wrappers, and complex YAML configurations. It reduces the "dark art" of AI alignment into a single, highly readable file of pure PyTorch. 

If you can write a `for` loop, you can align a language model.

## The Philosophy
* **Zero Bloat:** No `accelerate`, no `trl`, no `peft` (initially). Just clean, mathematically exact PyTorch.
* **Focus on the Math:** Demystifies the DPO loss function—showing how alignment is fundamentally just binary cross-entropy over log probabilities.
* **Trainable on a Laptop:** Uses a tiny dataset of Shakespearean insults to train a miniature transformer to be "polite" in minutes.

## Why DPO over RLHF?
Traditional Reinforcement Learning from Human Feedback (RLHF) requires training a separate Reward Model and using highly unstable PPO (Proximal Policy Optimization) algorithms. 

DPO proves mathematically that **the language model itself is the reward model**. It bypasses the PPO loop entirely, optimizing the policy directly based on human preferences using this elegant objective:

$$\mathcal{L}_{DPO} = -\log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right)$$

## Quick Start

```bash
git clone [https://github.com/Rahulkumar010/microDPO.git](https://github.com/Rahulkumar010/microDPO.git)
cd microDPO
uv sync --link-mode=copy

```

Train the alignment model on the toy dataset:

```bash
uv run train.py

```

For inference:

```bash
uv run inference.py

```

For visualizing the metrics:

```bash
uv run plot_metrics.py

```

## The Core Code

The entire magic of DPO is contained in just a few lines of code. We calculate how much the model we are training (the **Policy**) likes the "good" answer vs. the "bad" answer, compared to a frozen copy of the original model (the **Reference**).

```python
import torch
import torch.nn.functional as F

def micro_dpo_loss(policy_chosen_logps, policy_rejected_logps, 
                   ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    The absolute minimal implementation of DPO.
    """
    # 1. Calculate the implicit "Rewards"
    # Reward = How much MORE the policy model likes the text compared to the frozen reference model
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # 2. The Core Objective: Maximize the margin between chosen and rejected rewards
    margin = chosen_rewards - rejected_rewards
    
    # 3. Binary Cross Entropy (Log Sigmoid)
    loss = -F.logsigmoid(margin).mean()

    return loss

```

## Repository Structure

*   `model.py`: A bare-bones, miniature GPT architecture (borrowed heavily from the spirit of `nanoGPT`).
*   `dataset.py`: A tiny script that generates synthetic `(prompt, chosen_response, rejected_response)` triplets.
*   `train.py`: The main training loop. It handles the freezing of the Reference Model and the backpropagation of the Policy Model, and saves training metrics to `metrics.json`.
*   `inference.py`: A python script showing the before-and-after of the model's behavior. Watch it go from generating toxic text to perfectly polite responses.
*   `plot_metrics.py`: Visualizes the DPO loss and margin metric over epochs from the metrics output.

## Roadmap & Extensions

While `microDPO` is meant to be minimal, it serves as a perfect foundation for extensions:

* [ ] Add simple LoRA (Low-Rank Adaptation) injection to train slightly larger models without OOM errors.
* [ ] Port the core DPO loop to C or Rust.
* [x] Add a visualization script for tracking the "Margin" metric during training.

## Acknowledgements

Inspired by the pedagogical brilliance of Andrej Karpathy's `nanoGPT` and `micrograd`. The math is directly sourced from the original [Direct Preference Optimization paper](https://arxiv.org/abs/2305.18290) by Rafailov et al.

## License

MIT
