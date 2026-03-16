import json
import matplotlib.pyplot as plt

def main():
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("metrics.json not found. Please run train.py first to generate metrics.")
        return

    epochs = metrics['epochs']
    losses = metrics['losses']
    margins = metrics['margins']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot DPO Loss
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('DPO Loss', color=color)
    ax1.plot(epochs, losses, color=color, label='Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot Margin on the second y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Margin (Chosen - Rejected)', color=color)
    ax2.plot(epochs, margins, color=color, label='Margin', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)

    # Formatting and saving
    fig.tight_layout()
    plt.title('DPO Training Metrics (Loss & Margin)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_path = 'dpo_metrics.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot successfully saved as {output_path}")
    
    # Check if we should display it immediately (useful if running in GUI env)
    # plt.show()

if __name__ == "__main__":
    main()
