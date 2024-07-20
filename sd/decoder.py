import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_Residual(nn.Module) :
    def __init__(self,inChannels,outChannels) :
        super(VAE_Residual,self).__init__()
        self.groupnorm1 = nn.GroupNorm(32,inChannels)
        self.conv1 = nn.conv2d(inChannels,outChannels,kernel_size = 3 , padding = 1)

        self.groupnorm2 = nn.GroupNorm(32,outChannels)
        self.conv2 = nn.Conv2d(outChannels,outChannels,kernel_size=3,padding = 1)
        '''
            Skip connection , take input ,skip some layers and connect it with the output of the last layer
            * If the output and the input are of same channels we can , directly add the residue to the output
            * Or we have to convert the residue to the shape of the input
        '''
        if inChannels == outChannels :
            self.residualLayer = nn.Identity()
        else :
            self.residualLayer = nn.Conv2d(inChannels,outChannels,kernel_size=1,padding = 0)

    def forwaard(self,x : torch.Tensor) -> torch.Tensor :
        
        self.residue = x

        x = self.groupnorm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.groupnorm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + self.residualLayer(self.residue)
