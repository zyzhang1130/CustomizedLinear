import sys
sys.modules[__name__].__dict__.clear()


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from CustomizedLinear import CustomizedLinear
import torch.optim as optim
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from torch.backends import cudnn
import random

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
cudnn.benchmark = True

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
    def __init__(self,column,column_width):
        super(Net, self).__init__()
        self.column = column
        self.column_width = column_width
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
        
        self.mask5=self.mask1.clone()
        self.mask5[0,0:self.column_width*5]=1
        self.mask5 = self.mask5.to(device)
        
        
        #different scaling
        
        # self.scaling1 = self.column
        # self.scaling2 = self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]
        # self.scaling3 = self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]
        # self.scaling4 = self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]
        # self.scaling5 = self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]
        
        
        # self.scaling1 = 1.5
        # self.scaling2 = 1.5
        # self.scaling3 = 1.5
        # self.scaling4 = 1.5
        # self.scaling5 = 1.5
        
        self.scaling1 = 2
        self.scaling2 = 2
        self.scaling3 = 2
        self.scaling4 = 2
        self.scaling5 = 2
        
        # self.scaling1 = self.column
        # self.scaling2 = self.column/3
        # self.scaling3 = self.column/5
        # self.scaling4 = self.column/7
        # self.scaling5 = self.column/9
        
        
        # #debugging
        # m1 = self.mask1.numpy()
        # m2 = self.mask2.numpy()
        # m3 = self.mask3.numpy()
        # m4 = self.mask4.numpy()
        # m5 = self.mask5.numpy()
        # mm = np.concatenate((m5, m4, m3, m2,m1), axis=0)
        
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
                    self.mask5 = torch.zeros([1,self.column_width * self.column])
                    self.mask5[0,0:self.column_width*5]=1
                    
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
                        
                        
                    self.mask5 = torch.zeros([1,self.column_width * self.column])
                    if i >= (self.column-5)*self.column_width:
                        if i < 3*(self.column_width):
                            self.mask5[0,0:]=1
                        else:
                            self.mask5[0,i-3*self.column_width:]=1
                    elif i < 3*(self.column_width):
                        self.mask5[0,0:i+self.column_width*6]=1
                    else:
                        self.mask5[0,i-3*self.column_width:i+self.column_width*6]=1

                self.mask1 = self.mask1.to(device)
                self.mask2 = self.mask2.to(device)
                self.mask3 = self.mask3.to(device)
                self.mask4 = self.mask4.to(device)
                self.mask5 = self.mask5.to(device)
                
                # self.scaling1 = self.column
                # self.scaling2 = self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]
                # self.scaling3 = self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]
                # self.scaling4 = self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]
                # self.scaling5 = self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]
                
                
                #debugging
                # m1 = self.mask1.numpy()
                # m2 = self.mask2.numpy()
                # m3 = self.mask3.numpy()
                # m4 = self.mask4.numpy()
                # m5 = self.mask5.numpy()
                # mm = np.concatenate((m5, m4, m3, m2,m1), axis=0)
                
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
        
        self.mask5=self.mask1.clone()
        self.mask5[0,:]=1
        self.mask5 = self.mask5.to(device)
        
        
        #different scaling
        
        # self.scaling1 = self.column
        # self.scaling2 = self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]
        # self.scaling3 = self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]
        # self.scaling4 = self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]
        # self.scaling5 = self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]
        
        # self.scaling1 = 1.5
        # self.scaling2 = 1.5
        # self.scaling3 = 1.5
        # self.scaling4 = 1.5
        # self.scaling5 = 1.5
        
        # self.scaling1 = 1/self.column
        # self.scaling2 = 1/(self.column/3)
        # self.scaling3 = 1/(self.column/5)
        # self.scaling4 = 1/(self.column/7)
        # self.scaling5 = 1/(self.column/9)
    
        self.scaling1 = 2
        self.scaling2 = 2
        self.scaling3 = 2
        self.scaling4 = 2
        self.scaling5 = 2

            
   

    def forward(self, x):
        # x = F.relu(self.fc1(x)*self.mask1*self.column)
        # x = F.relu(self.fc2(x)*self.mask2*(self.column/3))
        # x = F.relu(self.fc2(x)*self.mask3*(self.column/5))
        # x = F.relu(self.fc2(x)*self.mask4*(self.column/7))
        # x = F.relu(self.fc2(x)*self.mask5*(self.column/9))
        # x = F.log_softmax(self.fc3(x))
        
        # x = F.relu(self.fc1(x)*self.mask1*self.column)
        # x = F.relu(self.fc2(x)*self.mask2*(self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]))
        # x = F.relu(self.fc2(x)*self.mask3*(self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]))
        # x = F.relu(self.fc2(x)*self.mask4*(self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]))
        # x = F.relu(self.fc2(x)*self.mask5*(self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]))
        # x = F.log_softmax(self.fc3(x))
        
        x = F.relu(self.fc1(x)*self.mask1*self.scaling1)
        x = F.relu(self.fc2(x)*self.mask2*self.scaling2)
        x = F.relu(self.fc2(x)*self.mask3*self.scaling3)
        x = F.relu(self.fc2(x)*self.mask4*self.scaling4)
        x = F.relu(self.fc2(x)*self.mask5*self.scaling5)
        x = F.log_softmax(self.fc3(x))
        
        
        # x = F.relu(self.fc1(x)*self.mask1)
        # x = F.relu(self.fc2(x)*self.mask2)
        # x = F.relu(self.fc2(x)*self.mask3)
        # x = F.relu(self.fc2(x)*self.mask4)
        # x = F.relu(self.fc2(x)*self.mask5)
        # x = F.log_softmax(self.fc3(x))
        
    
        # #debugging
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x))
        return x
    
    
    
    
#feature mask2 (deterministic dropout)
class Net2(nn.Module):
    def __init__(self,column,column_width):
        super(Net2, self).__init__()
        self.column = column
        self.column_width = column_width
        self.fc1 = nn.Linear(28*28, self.column_width*self.column)
        self.fc2 = nn.Linear(self.column_width*self.column, self.column_width*(self.column+2))
        self.fc3 = nn.Linear(self.column_width*(self.column+2), self.column_width*(self.column+4))
        self.fc4 = nn.Linear(self.column_width*(self.column+4), self.column_width*(self.column+6))
        self.fc5 = nn.Linear(self.column_width*(self.column+6), self.column_width*(self.column+8))
        self.fc6 = nn.Linear(self.column_width*(self.column+8), 10)
        self.bn1 = nn.LayerNorm(self.column_width*self.column)
        self.bn2 = nn.LayerNorm(self.column_width*(self.column+2))
        self.bn3 = nn.LayerNorm(self.column_width*(self.column+4))
        self.bn4 = nn.LayerNorm(self.column_width*(self.column+6))
        self.bn5 = nn.LayerNorm(self.column_width*(self.column+8))

        self.mask1 = torch.zeros([1,self.column_width * self.column])
        self.mask1[0,0:self.column_width]=1
        self.mask1 = self.mask1.to(device)

        self.mask2 = torch.zeros([1,self.column_width * (self.column+2)])
        self.mask2[0,0:self.column_width*3]=1
        self.mask2 = self.mask2.to(device)

        self.mask3 = torch.zeros([1,self.column_width * (self.column+4)])
        self.mask3[0,0:self.column_width*5]=1
        self.mask3 = self.mask3.to(device)

        self.mask4 = torch.zeros([1,self.column_width * (self.column+6)])
        self.mask4[0,0:self.column_width*7]=1
        self.mask4 = self.mask4.to(device)
        
        self.mask5 = torch.zeros([1,self.column_width * (self.column+8)])
        self.mask5[0,0:self.column_width*9]=1
        self.mask5 = self.mask5.to(device)
        
        
        #different scaling
        
        # self.scaling1 = self.column
        # self.scaling2 = self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]
        # self.scaling3 = self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]
        # self.scaling4 = self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]
        # self.scaling5 = self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]
        
        
        # self.scaling1 = 1.5
        # self.scaling2 = 1.5
        # self.scaling3 = 1.5
        # self.scaling4 = 1.5
        # self.scaling5 = 1.5
        
        # self.scaling1 = 1
        # self.scaling2 = 1
        # self.scaling3 = 1
        # self.scaling4 = 1
        # self.scaling5 = 1
        
        # self.scaling1 = 1/self.column
        # self.scaling2 = 1/(self.column/3)
        # self.scaling3 = 1/(self.column/5)
        # self.scaling4 = 1/(self.column/7)
        # self.scaling5 = 1/(self.column/9)
        
        self.scaling1 = self.column
        self.scaling2 = (self.column/3)
        self.scaling3 = (self.column/5)
        self.scaling4 = (self.column/7)
        self.scaling5 = (self.column/9)
        
        # self.scaling1 = self.column
        # self.scaling2 = self.column
        # self.scaling3 = self.column
        # self.scaling4 = self.column
        # self.scaling5 = self.column
        
        
        # #debugging
        # m1 = self.mask1.numpy()
        # m1 = np.concatenate((np.zeros((1,4*self.column_width)),m1,np.zeros((1,4*self.column_width))), axis=1)
        # m2 = self.mask2.numpy()
        # m2 = np.concatenate((np.zeros((1,3*self.column_width)),m2,np.zeros((1,3*self.column_width))), axis=1)
        # m3 = self.mask3.numpy()
        # m3 = np.concatenate((np.zeros((1,2*self.column_width)),m3,np.zeros((1,2*self.column_width))), axis=1)
        # m4 = self.mask4.numpy()
        # m4 = np.concatenate((np.zeros((1,self.column_width)),m4,np.zeros((1,self.column_width))), axis=1)
        # m5 = self.mask5.numpy()
        # mm = np.concatenate((m5, m4, m3, m2,m1), axis=0)
        
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
                    self.mask2 = torch.zeros([1,self.column_width * (self.column+2)])
                    self.mask2[0,0:self.column_width*3]=1
                    self.mask3 = torch.zeros([1,self.column_width * (self.column+4)])
                    self.mask3[0,0:self.column_width*5]=1
                    self.mask4 = torch.zeros([1,self.column_width * (self.column+6)])
                    self.mask4[0,0:self.column_width*7]=1
                    self.mask5 = torch.zeros([1,self.column_width * (self.column+8)])
                    self.mask5[0,0:self.column_width*9]=1
                    
                    
                    
                    
                else:
                    self.mask1 = torch.zeros([1,self.column_width * self.column])
                    self.mask1[0,i+self.column_width:i+self.column_width*2]=1

                    self.mask2 = torch.zeros([1,self.column_width * (self.column+2)])
                    self.mask2[0,i+self.column_width:i+self.column_width*4]=1
                    
                    self.mask3 = torch.zeros([1,self.column_width * (self.column+4)])
                    self.mask3[0,i+self.column_width:i+self.column_width*6]=1
                    
                    self.mask4 = torch.zeros([1,self.column_width * (self.column+6)])
                    self.mask4[0,i+self.column_width:i+self.column_width*8]=1
                    
                    self.mask5 = torch.zeros([1,self.column_width * (self.column+8)])
                    self.mask5[0,i+self.column_width:i+self.column_width*10]=1
                    
                self.mask1 = self.mask1.to(device)
                self.mask2 = self.mask2.to(device)
                self.mask3 = self.mask3.to(device)
                self.mask4 = self.mask4.to(device)
                self.mask5 = self.mask5.to(device)
                
                # self.scaling1 = self.column
                # self.scaling2 = self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]
                # self.scaling3 = self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]
                # self.scaling4 = self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]
                # self.scaling5 = self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]
                
                
                # #debugging
                # m1 = self.mask1.numpy()
                # m1 = np.concatenate((np.zeros((1,4*self.column_width)),m1,np.zeros((1,4*self.column_width))), axis=1)
                # m2 = self.mask2.numpy()
                # m2 = np.concatenate((np.zeros((1,3*self.column_width)),m2,np.zeros((1,3*self.column_width))), axis=1)
                # m3 = self.mask3.numpy()
                # m3 = np.concatenate((np.zeros((1,2*self.column_width)),m3,np.zeros((1,2*self.column_width))), axis=1)
                # m4 = self.mask4.numpy()
                # m4 = np.concatenate((np.zeros((1,self.column_width)),m4,np.zeros((1,self.column_width))), axis=1)
                # m5 = self.mask5.numpy()
                # mm = np.concatenate((m5, m4, m3, m2,m1), axis=0)
                
                return
            
    def train(self):
        self.mask1=self.mask1_cache.clone()
        self.mask2=self.mask2_cache.clone()
        self.mask3=self.mask3_cache.clone()
        self.mask4=self.mask4_cache.clone()
        self.mask5=self.mask5_cache.clone()
        
                    
    def test(self):
        
        self.mask1_cache=self.mask1.clone()
        self.mask2_cache=self.mask2.clone()
        self.mask3_cache=self.mask3.clone()
        self.mask4_cache=self.mask4.clone()
        self.mask5_cache=self.mask5.clone()
        
        
        
        self.mask1 = torch.zeros([1,self.column_width * self.column])
        self.mask1[0,:]=1
        self.mask1 = self.mask1.to(device)

        self.mask2 = torch.zeros([1,self.column_width * (self.column+2)])
        self.mask2[0,:]=1
        self.mask2 = self.mask2.to(device)

        self.mask3 = torch.zeros([1,self.column_width * (self.column+4)])
        self.mask3[0,:]=1
        self.mask3 = self.mask3.to(device)

        self.mask4 = torch.zeros([1,self.column_width * (self.column+6)])
        self.mask4[0,:]=1
        self.mask4 = self.mask4.to(device)
        
        self.mask5 = torch.zeros([1,self.column_width * (self.column+8)])
        self.mask5[0,:]=1
        self.mask5 = self.mask5.to(device)
        
        
        #different scaling
        
        # self.scaling1 = self.column
        # self.scaling2 = self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]
        # self.scaling3 = self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]
        # self.scaling4 = self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]
        # self.scaling5 = self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]
        
        # self.scaling1 = 1.5
        # self.scaling2 = 1.5
        # self.scaling3 = 1.5
        # self.scaling4 = 1.5
        # self.scaling5 = 1.5
        
        # self.scaling1 = 1/self.column
        # self.scaling2 = 1/(self.column/3)
        # self.scaling3 = 1/(self.column/5)
        # self.scaling4 = 1/(self.column/7)
        # self.scaling5 = 1/(self.column/9)
    
        # self.scaling1 = 1
        # self.scaling2 = 1
        # self.scaling3 = 1
        # self.scaling4 = 1
        # self.scaling5 = 1
        # pass

            
   

    def forward(self, x):
        # #unsqueeze for InstanceNorm1d
        # x = torch.unsqueeze(x, 0)
        
        # x = F.relu(self.fc1(x)*self.mask1*self.column)
        # x = F.relu(self.fc2(x)*self.mask2*(self.column/3))
        # x = F.relu(self.fc2(x)*self.mask3*(self.column/5))
        # x = F.relu(self.fc2(x)*self.mask4*(self.column/7))
        # x = F.relu(self.fc2(x)*self.mask5*(self.column/9))
        # x = F.log_softmax(self.fc3(x))
        
        # x = F.relu(self.fc1(x)*self.mask1*self.column)
        # x = F.relu(self.fc2(x)*self.mask2*(self.column*self.column_width/np.count_nonzero(self.mask2, axis=1)[0]))
        # x = F.relu(self.fc2(x)*self.mask3*(self.column*self.column_width/np.count_nonzero(self.mask3, axis=1)[0]))
        # x = F.relu(self.fc2(x)*self.mask4*(self.column*self.column_width/np.count_nonzero(self.mask4, axis=1)[0]))
        # x = F.relu(self.fc2(x)*self.mask5*(self.column*self.column_width/np.count_nonzero(self.mask5, axis=1)[0]))
        # x = F.log_softmax(self.fc3(x))
        
        # x = F.relu(self.bn1(self.fc1(x)*self.mask1*self.scaling1))
        # x = F.relu(self.bn2(self.fc2(x)*self.mask2*self.scaling2))
        # x = F.relu(self.bn3(self.fc3(x)*self.mask3*self.scaling3))
        # x = F.relu(self.bn4(self.fc4(x)*self.mask4*self.scaling4))
        # x = F.relu(self.bn5(self.fc5(x)*self.mask5*self.scaling5))
        # x = F.log_softmax(self.fc6(x))
        
        x = F.relu(self.fc1(x)*self.mask1*self.scaling1)
        x = F.relu(self.fc2(x)*self.mask2*self.scaling2)
        x = F.relu(self.fc3(x)*self.mask3*self.scaling3)
        x = F.relu(self.fc4(x)*self.mask4*self.scaling4)
        x = F.relu(self.fc5(x)*self.mask5*self.scaling5)
        x = F.log_softmax(self.fc6(x))
        
        
        # x = F.relu(self.fc1(x)*self.mask1)
        # x = F.relu(self.fc2(x)*self.mask2)
        # x = F.relu(self.fc2(x)*self.mask3)
        # x = F.relu(self.fc2(x)*self.mask4)
        # x = F.relu(self.fc2(x)*self.mask5)
        # x = F.log_softmax(self.fc3(x))
        
    
        # #debugging
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc2(x))
        # x = F.log_softmax(self.fc3(x))
        
        
        # #for InstanceNorm1d
        # x = torch.squeeze(x)
        # x = torch.unsqueeze(x, 0)
        return x
    
column=10
column_width=64
net=Net2(column,column_width)
net.to(device)

criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.001)
import math

loss_history=[]
label_history=[]
acc_hist=[]
tot_epoch=50
datasize=2000
for epoch in range(tot_epoch):  # loop over the dataset multiple times
    running_loss = 0.0


    for i, data in enumerate(trainloader, 0):

        if i > (datasize/batch_size)-1:
          break
        # get the inputs
        inputs, labels = data
        label_history.append(labels)

        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.view(1,-1))
        # print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            loss_history.append(running_loss / 1000)
            running_loss = 0.0
        net.mask_update()
        
plt.plot(loss_history)
        
net.test()
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
acc=[]
with torch.no_grad():
    for i, data in enumerate(trainloader, 0):
      if i > 5000:
          break
        # get the inputs
      images, labels = data
      outputs = net(images.view(1,-1).to(device))
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels.to(device)).squeeze()
      # print(c,labels)
      
      label = labels[0]
      class_correct[label] += c.item()
      class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    acc.append(class_correct[i] / class_total[i])

print('Average Accuracy: %2d %%' % (
        100 * sum(class_correct) / sum(class_total)))
    



objects = ('0', '1', '2', '3',
           '4', '5', '6', '7', '8', '9')
y_pos = np.arange(len(objects))
performance = acc

plt.figure()
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('accuracy')
# plt.title('Programming language usage')

plt.show()
