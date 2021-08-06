import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from CustomizedLinear import CustomizedLinear
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
batch_size=1
img_transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# device = torch.device("cpu")
trainset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

testset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9')


    
#feature mask (deterministic dropout)
class Net(nn.Module):
    def __init__(self,column):
        super(Net, self).__init__()
        self.column = column
        self.column_width = 4
        self.fc1 = nn.Linear(28*28, self.column_width*self.column)
        self.fc2 = nn.Linear(self.column_width*self.column, self.column_width*self.column)
        self.fc3 = nn.Linear(self.column_width*self.column, 10)

        self.mask1 = torch.zeros([1,self.column_width * self.column])
        self.mask1[0,0:self.column_width]=1
        self.mask1 = self.mask1.to(device)

        self.mask2=self.mask1.clone()
        self.mask2[0,0:self.column_width*2]=1
        self.mask2 = self.mask2.to(device)

        self.mask3=self.mask1.clone()
        self.mask3[0,0:self.column_width*3]=1
        self.mask3 = self.mask3.to(device)

        self.mask4=self.mask1.clone()
        self.mask4[0,0:self.column_width*4]=1
        self.mask4 = self.mask4.to(device)
        
        
        # #debugging
        # m1 = self.mask1.numpy()
        # m2 = self.mask2.numpy()
        # m3 = self.mask3.numpy()
        # m4 = self.mask4.numpy()
        # mm = np.concatenate((m4, m3, m2,m1), axis=0)
        
        # pass
        
        # self.mask1[:]=1
        # self.mask2[:]=1
        # self.mask3[:]=1


        
        
        # for i in range(512):
        #     for j in range(512*2):
        #         self.mask[i,j]=1
        # for k in range(1,column-1):
        #     for i in range(512*k,512*(k+1)):
        #         for j in range(512*(k-1),512*(k+1)):
        #             self.mask[i,j]=1
        # for i in range(512*column-1,512*column):
        #     for j in range(512*(k-1),512*k):
        #         self.mask[i,j]=1
        
        # self.c = CustomizedLinear(self.mask, bias=None)

    def mask_update(self):
        for i in range(self.column_width*self.column):
            if self.mask1[0,i] == 1:
                if i >= (column-1)*self.column_width:
                    self.mask1 = torch.zeros([1,self.column_width * self.column])
                    self.mask1[0,0:self.column_width]=1
                    self.mask2 = torch.zeros([1,self.column_width * self.column])
                    self.mask2[0,0:self.column_width*2]=1
                    self.mask3 = torch.zeros([1,self.column_width * self.column])
                    self.mask3[0,0:self.column_width*3]=1
                    self.mask4 = torch.zeros([1,self.column_width * self.column])
                    self.mask4[0,0:self.column_width*4]=1
                    
                else:
                    self.mask1 = torch.zeros([1,self.column_width * self.column])
                    self.mask1[0,i+self.column_width:i+self.column_width*2]=1

                    self.mask2 = torch.zeros([1,self.column_width * self.column])
                    if i >= (self.column-2)*self.column_width:
                        self.mask2[0,i:]=1
                    else:
                        self.mask2[0,i:i+self.column_width*3]=1

                    self.mask3 = torch.zeros([1,self.column_width * self.column])
                    if i >= (self.column-3)*self.column_width:
                        if i <= 1*(self.column_width):
                            self.mask3[0,0:]=1
                        else:
                            self.mask3[0,i-self.column_width:]=1
                    elif i < 1*(self.column_width):
                        self.mask3[0,0:i+self.column_width*4]=1
                    else:
                        self.mask3[0,i-self.column_width:i+self.column_width*4]=1

                    self.mask4 = torch.zeros([1,self.column_width * self.column])
                    if i >= (self.column-4)*self.column_width:
                        if i < 2*(self.column_width):
                            self.mask4[0,0:]=1
                        else:
                            self.mask4[0,i-2*self.column_width:]=1
                    elif i < 2*(self.column_width):
                        self.mask4[0,0:i+self.column_width*5]=1
                    else:
                        self.mask4[0,i-2*self.column_width:i+self.column_width*5]=1

                self.mask1 = self.mask1.to(device)
                self.mask2 = self.mask2.to(device)
                self.mask3 = self.mask3.to(device)
                self.mask4 = self.mask4.to(device)
                
                
                # #debugging
                # m1 = self.mask1.numpy()
                # m2 = self.mask2.numpy()
                # m3 = self.mask3.numpy()
                # m4 = self.mask4.numpy()
                # mm = np.concatenate((m4, m3, m2,m1), axis=0)
                
                return
                    
    def test(self):
        self.mask1 = torch.zeros([1,self.column_width * self.column])
        self.mask1[0,:]=1
        self.mask1 = self.mask1.to(device)

        self.mask2=self.mask1.clone()
        self.mask2[0,:]=1
        self.mask2 = self.mask2.to(device)

        self.mask3=self.mask1.clone()
        self.mask3[0,:]=1
        self.mask3 = self.mask3.to(device)

        self.mask4=self.mask1.clone()
        self.mask4[0,:]=1
        self.mask4 = self.mask4.to(device)
        

            
   

    def forward(self, x):
        x = F.relu(self.fc1(x)*self.mask1)
        x = F.relu(self.fc2(x)*self.mask2)
        x = F.relu(self.fc2(x)*self.mask3)
        x = F.relu(self.fc2(x)*self.mask4)
        x = F.log_softmax(self.fc3(x))
        
    
        #debugging
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x))
        return x
    
column=10
net=Net(column)
net.to(device)

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001)
import math

loss_history=[]
label_history=[]
acc_hist=[]
tot_epoch=10
datasize=2000
for epoch in range(tot_epoch):  # loop over the dataset multiple times
    running_loss = 0.0


    for i, data in enumerate(trainloader, 0):

        if i > (datasize/batch_size):
          break
        # get the inputs
        inputs, labels = data
        label_history.append(labels)

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.view(1,-1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        net.test()
        

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            loss_history.append(running_loss / 100)
            running_loss = 0.0
        net.mask_update()
