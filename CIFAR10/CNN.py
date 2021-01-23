import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
   [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#Load training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                         shuffle=True, num_workers=0)
#Load test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                        shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
          'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Define deep neural net
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(3, 10, 3, padding=(1,1))
       self.conv2 = nn.Conv2d(10, 50, 3, padding=(1,1))
       self.conv3 = nn.Conv2d(50, 100, 3, padding=(1,1))
       self.pool = nn.MaxPool2d(2, 2)
       self.dropout = nn.Dropout(0.5)
       self.bn1 = nn.BatchNorm2d(10)
       self.bn2 = nn.BatchNorm2d(50)
       self.bn3 = nn.BatchNorm2d(100)
       self.fc1 = nn.Linear(100 * 4 * 4, 120)
       self.fc2 = nn.Linear(120, 84)
       self.fc3 = nn.Linear(84, 10)

   def forward(self, x):
       x = self.pool(F.relu(self.bn1(self.conv1(x))))
       x = self.pool(F.relu(self.bn2(self.conv2(x))))
       x = self.pool(F.relu(self.bn3(self.conv3(x))))
       x = x.view(-1, 100 * 4 * 4)
       x = F.relu(self.fc1(x))
       x = F.relu(self.dropout(self.fc2(x)))
       x = self.fc3(x)
       return x

net = Net()
#Put net in CUDA device
net.to(device)

#Define a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

#Train the network
for epoch in range(10):  # loop over the dataset multiple times

   running_loss = 0.0
   for i, data in enumerate(trainloader, 0):
       # get the inputs; data is a list of [inputs, labels]
       inputs, labels = data[0].to(device), data[1].to(device)

       # zero the parameter gradients
       optimizer.zero_grad()

       # forward + backward + optimize
       outputs = net(inputs)
       loss = criterion(outputs, labels)
       loss.backward()
       optimizer.step()

       # print statistics
       running_loss += loss.item()
       if i % 200 == 199:    # print every 1000 mini-batches
           print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 200))
           running_loss = 0.0
print('Finished Training')

#Training accuracy
correct = 0
total = 0
with torch.no_grad():
   for data in trainloader:
       images, labels = data[0].to(device), data[1].to(device)
       outputs = net(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 50000 training images: %d %%' % (
   100 * correct / total))

#save net
# PATH = './cifar_net.pth'
# torch.save(net.state_dict(), PATH)

#Test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()

#load back in our saved model
# net = Net()
# net.to(device)
# net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
   for data in testloader:
       images, labels = data[0].to(device), data[1].to(device)
       outputs = net(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
   100 * correct / total))

#Show which classes performed well and which classes did not perform well
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
   for data in testloader:
       images, labels = data[0].to(device), data[1].to(device)
       outputs = net(images)
       _, predicted = torch.max(outputs, 1)
       c = (predicted == labels).squeeze()
       for i in range(4):
           label = labels[i]
           class_correct[label] += c[i].item()
           class_total[label] += 1

for i in range(10):
   print('Accuracy of %5s : %2d %%' % (
       classes[i], 100 * class_correct[i] / class_total[i]))
