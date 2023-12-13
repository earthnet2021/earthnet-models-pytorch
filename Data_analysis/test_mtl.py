import torch
import torch.nn as nn
import torch.optim as optim

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks):
        super(MultiTaskLoss, self).__init__()
        self.tasks = nn.ModuleList(tasks)
        self.sigma = nn.Parameter(torch.ones(len(tasks)))
        self.mse = nn.MSELoss()

    def forward(self, x, targets):
       l = [self.mse(f(x), y) for y, f in zip(targets, self.tasks)]
       l = 0.5 * torch.Tensor(l) / self.sigma**2
       l = l.sum() + torch.log(self.sigma.prod())
       print("LOSS:", l.item(), "SIGMA: ", self.sigma[0].item(), self.sigma[1].item())
       return l

f1 = nn.Linear(5, 1, bias=False)
f2 = nn.Linear(5, 1, bias=False)

x = torch.randn(100, 5)
y1 = 2* x[:,0] + torch.randn(100, 1)
y2 = x[:,1] + torch.randn(100, 1)

mtl = MultiTaskLoss([f1, f2])

print(list(mtl.parameters()))

optimizer = optim.SGD(mtl.parameters(), lr = 0.01)
optimizer.zero_grad()
for i in range(100):
    mtl(x, [y1, y2]).backward()
    optimizer.step()
