import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, 3, 3, 1)
      self.conv2 = nn.Conv2d(3, 3, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return F.relu(x)

random_data = torch.rand((1, 3, 21, 21))
my_nn = Net()
print(my_nn)
result = my_nn(random_data)
print(result)
