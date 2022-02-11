import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
from mlp_mixer_pytorch import MLPMixer

n_epochs = 30
batch_size_train = 52
batch_size_test = 100
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
img_height = 28
img_width = 28


from torchvision.datasets import MNIST

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

trainset = MNIST(root = './data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)

testset = MNIST(root = './data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)


# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLPMixer(
    image_size = 28,
    channels = 1,
    patch_size = 7,
    dim = 14,
    depth = 3,
    num_classes = 10
)

model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

mse = nn.MSELoss()


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size_train = data.shape[0]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pre_out = model(data)
        targ_out = torch.nn.functional.one_hot(target,num_classes=10)
        targ_out = targ_out.view((batch_size_train,10)).float()
        loss = mse(pre_out, targ_out)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss =0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size_test = data.shape[0]
            data, target = data.to(device), target.to(device)
            pre_out = model(data)
            targ_out = torch.nn.functional.one_hot(target,num_classes=10)
            targ_out = targ_out.view((batch_size_test,10)).float()
            test_loss += mse(pre_out, targ_out) # 将一批的损失相加
    
    test_loss /= len(test_loader.dataset)
    print("nTest set: Average loss: {:.4f}".format(test_loss))

    
for epoch in range(n_epochs):               
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
    torch.save(model.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')

