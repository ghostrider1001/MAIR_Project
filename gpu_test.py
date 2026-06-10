import torch

x = torch.rand(1000,1000).cuda()

print(x.device)