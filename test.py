import os 
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import networks
import datasets

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

model = networks.MLP(in_features=3*28*28, out_features=10, num_layers=3, hidden_dim=[256, 128, 64])
trainset = datasets.CMNIST_ZWDataset(split='train', conflict_ratio=0.5)
train_loader = DataLoader(dataset=trainset, batch_size=256, shuffle=True)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

model = model.to('cuda')
for epoch in range(100):
    for idx, (X, y, bias, image_path) in enumerate(train_loader):
        X, y = X.to('cuda'), y.to('cuda')
        logits = model(X)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss.item())