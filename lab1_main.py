import torch 
import torch.nn as nn 

# Створення нейронної мережі
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Ініціалізація моделі
model = SimpleNN()

# Огляд структури моделі
print(model)