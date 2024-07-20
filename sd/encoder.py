import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE_Encoder(nn.Sequential) :
    def __init__(self) :
        super(VAE_Encoder,self).__init__ (
            '''
                In this convolution we convert the image , with 3 channels into representation of 128 features for pixel
                (BatchSize , Channel = 3 , Height , Width) -> (BatchSize , 128 , Height , Width)
                with padding as 1 , the shape of the image , height and width remain constannt
            '''
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            '''
                Next if the residula block , which convert the 128 channels to 128 channels , but not change the size of the image
                Combination of convolutions + normlaizations
                (Batchsize,128,Height,width) -> (Batchsize,128,Height,width) 
            '''
            VAE_Residual(128,128),
            '''
                Again apply another residual block over it 
            '''
            VAE_Residual(128,128),
            '''
                Now this convolution changes the siuze of the image
                * Stide - skips the layers in between
                * kernel - is the convolution layer
                (BatchSize,128,Height,Width) -> (BatchSize , 128,Height/2,Width/2)
            '''
            nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=0),
            '''
                Again 2 more residual block after the convoltuion
                Increate the feature size in this case of the residual
                (BatchSize,128,Height/2 ,Width/2) -> (BatchSize,256,Height/2,width/2)
            '''
            VAE_Residual(128,256),
            '''
                Apply another layer of residual layer over the top of it
                (BatchSize , 256 , Heigth/2 , Width/2) -> (BatchSize , 256 , Height/2 , Width/2) 
            '''
            VAE_Residual(256,256),
            '''
                Now the convolution changes the height again by half same with the width too
                (BatchSize,256,Height/2,width/2) -> (BatchSize , 256 , Heigth/4 , Width/4)
            '''
            nn.Conv3d(256,256,stride = 2 , kernel_size = 3 , padding = 0),
            '''
                Residual layers to increase the feature space of the image
                (BatchSize,256,Height/4,Width/4) -> (BatchSize , 512 , Height/4,Width/4)
            '''
            VAE_Residual(256,512),
            '''
                Another residual but here the features and the dimensions reamin the same
                (BatchSize,512,Height/4,Width/4) -> (BatchSize , 512 , Height/4,Width/4)
            '''
            VAE_Residual(512,512),
           '''
                Now the convolution changes the height again by half same with the width too
                (BatchSize,512,Height/2,width/2) -> (BatchSize , 1024 , Heigth/8 , Width/8)
            '''
            nn.Conv3d(512,512,stride = 2 , kernel_size = 3 , padding = 0),
            '''
                Residual layers to increase the feature space of the image
                (BatchSize,512,Height/8,Width/8) -> (BatchSize , 512, Height/8,Width/8)
                Here the features per pixel remains the same
            '''
            VAE_Residual(512,512),
            '''
                Another residual but here the features and the dimensions reamin the same
                (BatchSize,512,Height/8,Width/8) -> (BatchSize , 512, Height/8,Width/8)
            '''
            VAE_Residual(512,512),
            '''
                Additional residual block 
            '''
            VAE_Residual(512,512),
            '''
                Attention Block  - Self attention over eeach pixel
                    * Way to relate token to each other in the pixel
                    * Each pixel is realted to each other , global context to the pixel
                    * Size os same , and the features are the same too
                (BatchSize,512,Height/8,Width/8) -> (BatchSize , 512,Height/8,Width/8)
            '''
            VAE_Attention(512),
            '''
                Additional layer of residual block , here we are keeping the dimensions same too
                (Batchsize,512,Height/8,width/8) -> (BatchSize,512 , Height/8,Width/8)
            '''
            VAE_Residual(512,512),
            '''
                Group normalization - doesnt change the channels or deimensions of the image
                Groups - 32
                Channels - 512
                (Batchsize,512,Height/8,width/8) -> (BatchSize,512 , Height/8,Width/8)
            '''
            nn.GroupNorm(32,512),
            '''
                Activation function - SilU (Sigmoid Linear unit)
                    * No reason to take it , but it works better
                (Batchsize,512,Height/8,width/8) -> (BatchSize,512 , Height/8,Width/8)
            '''
            nn.SiLU(),
            '''
                Another layer of convolutions - no change in the size of the model 
                * Decrease the number of features in this case
                (BatchSize,512,Heigth/8,width/8) -> (BatchSize , 8 , Height/8,Width/8)
            '''
            nn.Conv3d(512,8,kernel_size = 3,padding = 1),
            '''
                The final layer of convolution , not changes the features or shape of the convolutions in this case
            '''
            nn.Conv3d(8,8,kernel_size = 1,padding = 0)
        )

    def forward(self,x : torch.Tensor ,noise:torch.Tensor) :
        '''
            X - Image we want to encode 
                (BatchSize , 3, Height,Width)
            noise - same size of the output of the encoder 
                (BatchSize,8,Height/8,Width/8)
            
            Run all of these modules sequentially over each module
        '''
        for module in self :
            '''
                If the convolution has padding we have to add a another layer for processing
                Bottom and right of the image , (Padding_Left, Padding_Right,Padding_Top,Padding_Bottom)
            '''
            if getattr(module,'stride',None) == (2,2) :
                # Apply the padding when the stride of the convolution is 2
                ''' 
                    Why to add padding instead we could use the padding in the conv3d ?
                        * Cause the padding in the conv3d , is responsible for adding whole surroundings
                        * But for assymnetric padding we have to define them seperately 
                '''
                x = F.pad(x,(0,1,0,1))
            x = module(x)

        '''
            Mean and variance is what we learn from the variational auto encoder 
                * Each tesnor represent the mean and logVariance seperately
            (BatchSize , 8 , height/8,width/8) -> two tensors of shape 4 (BatchSize , 4 , height/8 , Width/8)
        '''
        mean , logVariance = torch.chunk(x,2,dim = 2)
        '''
            Clamping , Varaice to limit in a range , acceptable for us
                * Here we are converting the varaince to the range of -30 and 20 
            And make the logVaraivce to varaice
            (batchSize,4,Height/8,width/8) -> (batchSize , 4, Height/8,width/8)
        '''
        logVariance = torch.clamp(logVariance,-30.20)
        variance = logVariance.exp()
        '''
            (batchSize,4,Height/8,width/8) -> (batchSize , 4, Height/8,width/8)
        '''
        stdDeviation = variance.sqrt()

        '''
            Sample from the mean and variance we learn , how ?
                * We have the noise N(0,1) , we have to sample from this with given mean and varaice N(mean,varaince)
            z = N(0,1) -> N(mean,variance) by using
                * X = mean + stddeviation * z (sample from this distribution) 
        '''
        x = mean + stdDeviation*noise
        
        #Scale the output by  a constance - even i dont understand why we are doing but it is just a scaling constant
        x *= 0.18215

        return x


        
