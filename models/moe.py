# models/moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# models.moe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class MoE(nn.Module):
    def __init__(self, num_experts, expert_hidden_size, gate_hidden_size, input_size, dropout,device):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Expert(input_size, expert_hidden_size, dropout) for _ in range(num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hidden_size),
            nn.ReLU(),
            nn.Linear(gate_hidden_size, num_experts),
            nn.Softmax(dim=1)
        )
        self.device = device
        self.input_size = input_size
        self.usage_count = torch.zeros(num_experts, dtype=torch.long, device=self.device)

    def forward(self, x):
        gates = self.gate(x)  # (batch_size, num_experts)
        expert_outputs = [expert(x) for expert in self.experts]  # List of (batch_size, input_size)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, input_size)
        # Select experts based on gates
        output = torch.sum(gates.unsqueeze(2) * expert_outputs, dim=1)  # (batch_size, input_size)
        # Update usage count
        _, selected_experts = torch.max(gates, dim=1)  # (batch_size,)
        for expert_id in selected_experts:
            self.usage_count[expert_id] += 1
        return output

    def get_usage_count(self):
        return self.usage_count.cpu().numpy()
