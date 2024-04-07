import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from loss import SupConLoss
import networks
import datasets
from seed import fix

fix(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

model = networks.MLP(in_features=3*28*28, out_features=64, num_layers=3, hidden_dim=[100, 100, 100])
trainset = datasets.CMNIST_ZWDataset(split='train', conflict_ratio=0.5)
train_loader = DataLoader(dataset=trainset, batch_size=256, shuffle=True)
optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_fn = SupConLoss(GSC=True)

model = model.to('cuda')
for epoch in range(5):
    total_loss = 0
    for idx, (X, y, bias, image_path) in enumerate(train_loader):
        X, y = X.to('cuda'), y.to('cuda')
        logits = model(X)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(epoch, total_loss)

torch.save(model.state_dict(), './aux_model.pth')