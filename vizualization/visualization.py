import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

class Visualizer:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.plots = {}
        self.expert_usage_history = []
        self.train_losses = []
        self.val_losses = []
        self.epochs = []

    def save_scalar(self, tag, value, step, x_label='Step', y_label='Value', title=None):
        if tag not in self.plots:
            self.plots[tag] = {'x': [], 'y': []}
        
        self.plots[tag]['x'].append(step)
        self.plots[tag]['y'].append(value)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.plots[tag]['x'], self.plots[tag]['y'], marker='o', label=tag)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title if title else f"{tag} over Steps")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.log_dir, f"{tag}_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot for {tag} to {plot_path}")

    def save_combined_loss(self, train_loss, val_loss, epoch):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.epochs.append(epoch)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs, self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(self.log_dir, 'combined_loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Saved combined loss plot to {loss_plot_path}")

    def save_expert_usage(self, usage, epoch):
        self.expert_usage_history.append(usage)
        num_experts = len(usage)
        plt.figure(figsize=(10, 5))
        bar_width = 0.8
        plt.bar(range(num_experts), usage, width=bar_width, color='skyblue', edgecolor='black')
        plt.xlabel('Expert Index')
        plt.ylabel('Usage Count')
        plt.title(f'Expert Usage at Epoch {epoch+1}')
        plt.xticks(range(num_experts))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        usage_path = os.path.join(self.log_dir, f'expert_usage_epoch_{epoch+1}.png')
        plt.savefig(usage_path)
        plt.close()
        print(f"Saved expert usage plot to {usage_path}")

    def save_expert_usage_matrix(self):
        if not self.expert_usage_history:
            print("No expert usage data to visualize.")
            return
        usage_matrix = np.array(self.expert_usage_history)
        num_epochs = usage_matrix.shape[0]
        num_experts = usage_matrix.shape[1]
        plt.figure(figsize=(10, 6))
        cax = plt.matshow(usage_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(cax)
        plt.xlabel('Expert Index')
        plt.ylabel('Epoch')
        plt.title('Expert Usage Matrix')
        plt.xticks(range(num_experts))
        plt.yticks(range(num_epochs))
        plt.grid(False)
        matrix_path = os.path.join(self.log_dir, 'expert_usage_matrix.png')
        plt.savefig(matrix_path)
        plt.close()
        print(f"Saved expert usage matrix to {matrix_path}")

    def save_histogram(self, tag, values, step, x_label='Value', y_label='Frequency', title=None):
        plt.figure(figsize=(10, 5))
        plt.hist(values, bins=50, edgecolor='k', alpha=0.7)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title if title else f"Histogram of {tag} at Step {step}")
        plt.grid(True)
        histogram_path = os.path.join(self.log_dir, f"{tag}_histogram_step_{step}.png")
        plt.savefig(histogram_path)
        plt.close()
        print(f"Saved histogram for {tag} to {histogram_path}")