# training/trainer.py
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from vizualization.visualization import Visualizer

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, logger, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.visualizer = Visualizer(config['visualization']['log_dir'])
        self.global_step = 0
        self.expert_usage = []
        self.balance_loss_weight = 0.3  # Weight for the balance loss

    def train(self):
        for epoch in range(self.config['training']['epochs']):
            self.model.train()
            epoch_train_loss = 0.0
            epoch_val_loss = 0.0
            for batch in self.train_loader:
                # Move batch to device
                inputs = batch['input_ids'].to(self.device)
                targets = batch['input_ids'].to(self.device)  # Replace with actual targets
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Calculate balance loss
                balance_loss = self.calculate_balance_loss()
                total_loss = loss + self.balance_loss_weight * balance_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clipping'])
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_train_loss += loss.item()
                
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    self.logger.info(f"Epoch {epoch+1}, Step {self.global_step}, Loss: {loss.item()}, Balance Loss: {balance_loss.item()}")
                    self.visualizer.save_scalar('Training Loss', loss.item(), self.global_step)
                    self.visualizer.save_scalar('Balance Loss', balance_loss.item(), self.global_step)
                self.global_step += 1
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            self.validate(epoch)
            self.save_checkpoint(epoch)
            self.collect_expert_usage(epoch)
            self.visualize_expert_usage(epoch)
            self.visualizer.save_expert_usage_matrix()
            self.visualizer.save_combined_loss(avg_train_loss, epoch_val_loss, epoch)
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input_ids'].to(self.device)
                targets = batch['input_ids'].to(self.device)  # Replace with actual targets
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(self.val_loader)
        self.logger.info(f"Validation Loss at Epoch {epoch+1}: {avg_val_loss}")
        self.visualizer.save_scalar('Validation Loss', avg_val_loss, epoch)
        # Save a histogram of the loss values
        self.visualizer.save_histogram('Validation Loss', [avg_val_loss], epoch)
    
    def calculate_balance_loss(self):
        """
        Calculates the balance loss to ensure each expert is used equally.
        """
        usage = self.model.get_usage_count()
        usage_tensor = torch.tensor(usage, dtype=torch.float, device=self.device)
        mean_usage = torch.mean(usage_tensor)
        balance_loss = torch.mean((usage_tensor - mean_usage) ** 2)
        return balance_loss

    def collect_expert_usage(self, epoch):
        usage = self.model.get_usage_count()
        self.expert_usage.append(usage)
        # Reset usage count for next epoch
        self.model.usage_count.zero_()
    
    def visualize_expert_usage(self, epoch):
        usage = self.expert_usage[-1]
        self.visualizer.save_expert_usage(usage, epoch)
    
    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.config['visualization']['checkpoint_dir']}/model_epoch_{epoch+1}.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")