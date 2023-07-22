import torch
random_data = torch.rand((1, 3, 21, 21))
print(random_data)
torch.save(random_data, 'random_data.pt')