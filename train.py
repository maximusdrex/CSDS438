import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = nn.Conv2d(3, 3, 3, 1)
      #self.conv2 = nn.Conv2d(3, 3, 3, 1)
      #self.dropout1 = nn.Dropout2d(0.25)
      #self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        #x = F.relu(x)
        #x = self.conv2(x)
        return F.relu(x)

random_data = torch.rand((1, 3, 21, 21))
print(random_data.shape)
in_data = torch.load('processed/00001_00_0.1s.pt').float()
in_data = torch.unsqueeze(torch.transpose(in_data, 0, 2), dim=0)
print(in_data.shape)
my_nn = Net()
print(my_nn)
print(in_data)
result = my_nn(in_data)
print(result.shape)

target = torch.load('processed/00001_00_10s.pt').float()
target = torch.unsqueeze(torch.transpose(target[1:-1, 1:-1, :], 0, 2), dim=0)

criterion = nn.MSELoss()
loss = criterion(result, target)
print(loss)
