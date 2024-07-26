import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_Attention(nn.Module) :
    def __init__(self,channels:int) :
        super(VAE_Attention,self).__init__()
        '''
            Vannila Transformer - Layer Normalization
                * In case of DNN , each layer of network produce output (0,1), fed into next layer
                * If the output of the previous layer is  not in a range of (0,1) , lets say , it pushes the outptu of the current layer too
                * This pushes the loss function to osscilate
                * Some time very large number , sometimes very small number ....
                * Training becomes very slow
                * So with notmalization we push the data to be distributed around 0 , with variance 1

            Layer Normalization (over all channels of the item)
                * Calculate the mean and variance (row wise)
                * Find the new value from the calculated value of mean and variance

            Batch Normalization (over a particular channel of the item)
            *   Calculate hte mean and varaince (column wise)

            Group Normalization
                * In case of group normalization  , we use the same as the layer nornalization 
                * But we group the channels into groups (n - channels to n/5 groups)
                * We caluclate the mean and varaince over this particular group , so we get 5 mean and varaince instance norms 
                * Beacuse these features are close to each other , and could be relatable so we use group norm over the consecutive features

            Only reason to use is to make the training faster and the values to not osscilate when back propagated
        '''
        self.groupnorm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(1, channels)

    def forward(self,x : torch.Tensor) -> torch.Tensor  :
        '''
            x : (BatchSize , features , height , width)
            Self attention - Between all the pixels of the image
        '''
        residue = x 
        n,c,h,w = x.shape
        x = x.view(n,c,h*w)
        '''
            We transpose the x over the last two dimension , this is due to we want to compute the attention between every pixel
            (BatchSize , features, Height*Width) -> (BatchSize, Height*width, features)
        '''
        x = x.transpose(-1,-2)
        '''
            We calculate the self attention over the x
            (BatchSize, Height*width, features) -> (BatchSize, Height*width, features)
        '''
        x = self.attention(x)
        '''
            We transpose the values back to the original state
            (BatchSize, Height*width, features) -> (BatchSize , features, Height*Width)
        '''
        x = x.transpose(-1,2)
        '''
            Seperate the values of the height and width from the combined values 
        '''
        x = x.view((n,c,h,w))
        '''
        Add the residue to the final computes value of the x after self attention and dividing it into its original format
        '''
        x += residue

        return x

            


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
