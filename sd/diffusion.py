import torch
import torch.nn as nn 
from torch.nn import functional as F
from attention import SelfAttention , CrossAttention

'''
    The Diffusion model is the UNet model that predicts the noise at a time stamp t-1 given teh noise at time t  , 
    * We need to give the unet not only the noisified image , but also at what time stamp does the image correspond to.
    * 
'''

class TimeEmbedding(nn.Module) :
    def __init__(self,nEmbedd:int) :
        super().__init__()
        self.linear1 = nn.Linear(nEmbedd , 4 * nEmbedd)
        self.linear2 = nn.Linear(4*nEmbedd,4*nEmbedd)
    
    def forward(self,x : torch.Tensor) -> torch.Tensor :
        # x : (1,320)

        x = self.linear1(x)

        x = F.silu(x)

        x = self.linear2(x)

        #x : (1,1280)  
        return x
    
class UNET_ResidualBlock(nn.Module) :
    def __init__(self , inChannels : int , outChannels : int, timeEmbedd : 1280) :
        super().__init__()

        self.groupNorm = nn.GroupNorm(32, inChannels)
        self.conv = nn.Conv2d(inChannels,outChannels , kernel_size=3 , padding=1)
        self.linearTime = nn.Linear(timeEmbedd , outChannels)

        self.groupNorm_merged = nn.GroupNorm(32 , outChannels)
        self.conv_merged = nn.Conv2d(outChannels,outChannels , kernel_size=3 , padding = 1)

        if inChannels == outChannels :
            self.residualLayer = nn.Identity()

        else :
            self.residualLayer = nn.Conv2d(inChannels,outChannels, kernel_size=1 , padding=0)

    def forward(self, feature , time) :
        # Feature (BatchSize , inchannels , aHeight , Width)
        # tiume (1,1280)

        # We have time embedding , feature and context at the input of residual
        # Unet learn to recognize noise at a particular time stamp , and to relate it with the time embedding
        #And relate it with context with cross attention

        residue = feature

        feature = self.groupNorm(feature)

        feature = F.siu(feature)

        feature = self.conv(feature)

        time = F.silu(time)

        time = self.linearTime(time)

        merged = feature + time.unsqueeze(-1).unsqeeze(-1)

        merged = self.groupNorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        merged = merged + self.residualLayer(residue)

        return merged
    
class UNET_AttentionBlock(nn.Module) :
    def __init__(self , nHead : int , nEmbedd : int , dContext : 768) :
        super().__init__() 
        channels = nHead * nEmbedd

        self.groupNorm = nn.GroupNorm(32 , channels , eps = 1e-6)
        self.conv_input = nn.Conv2d(channels , channels , kernel_size = 1 , padding = 0)

        self.layerNorm1 = nn.LayerNorm(channels)
        self.attention1 = SelfAttention(nHead , channels , in_proj_bias=False)
        self.layerNorm2 = nn.LayerNorm(channels)
        self.attention2 = CrossAttention(nHead , channels , d_context = 768, in_proj_bias=False)
        self.layerNorm3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels , 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels , channels)

        self.conv_output = nn.Conv2d(channels , channels , kernel_size = 1 ,padding = 0)

    def forward(self , x , context) :
        # x : BatchSiez , features , height , width
        # context : BatcSize , seqlen , dim

        residueLong = x

        x = self.groupNorm(x)

        x = self.conv_input(x)

        n,c,h,w = x.shape

        # (BatchSize , feature ,height , width) -> (BatchSize , feature , height * width)
        x = x.view((n,c,h*w))

        # (BatchSize , feature , height*width ) -> (BatchSize , height * width , features)
        x = x.transpose(-1 , -2)

        # Normalize + selfattention with skip connection

        residueShort = x

        x = self.layerNorm1(x)
        x = self.attention1(x)

        x += residueShort

        # Normalize + cross_attention with skip connection

        residueShort = x

        x = self.layerNorm2(x)
        x = self.attention2(x,context)

        x += residueShort

        # Activation function for geglu activation feed forward
        
        residueShort = x

        x = self.layerNorm3(x)

        # Did not understand leave it , even I didn't understood properly !
        x , gate = self.linear_geglu_1(x).chunk(2 , dim =  -1)

        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residueShort

        # (BatchSize ,  height * width ,feature ) -> (BatcShize ,feature , height * width)

        x = x.transpose(-1,-2)

        # (BatchSize , feature , height*width) -> (BatcSize , feature , height , width)

        x = x.view((n,c,h,w))

        x = self.conv_output(x)

        x += residueLong

        return x


class Upsample(nn.Module) :
    def __init__(self,channels : int) :
        super().__init__()
        self.conv = nn.Conv2d(channels, channels , kernel_size = 3 , padding = 1)

    def forward(self , x : torch.Tensor) :
        # (BatchSize , features , Heigth , Width) -> (BatchSize , features , Heigth*2 , Width*2)
        x = F.interpolate(x , scale_factor=2 , mode = 'nearest')

        x = self.conv(x)

        return x
    
class UNET_OutputLayer(nn.Module) :
    def __init__(self , inChannels : int , outChannels : int) :
        super().__init__()
        self.groupNorm = nn.GroupNorm(32 , inChannels)
        self.conv = nn.Conv2d(inChannels,outChannels , kernel_size=3 , padding=1)

    def forward(self, x : torch.Tensor) :
        #x : (BatchSize , 320 , Height/8 , Width / 8)

        x = self.groupNorm(x)

        x = F.silu()

        x = self.conv(x)

        # (BatchSize , 4 ,Height/8 , Width / 8)
        return x

# This module `Switch Sequential` - given a sequence of layers , it will apply one after the other
class SwitchSequential(nn.Sequential) :
    def forward(self, x : torch.Tensor , context : torch.Tensor , time : torch.Tensor) -> torch.Tensor :
        for layer in self :
            if isinstance(layer,UNET_AttentionBlock) :
                # Applies the cross attention betwenen the latents and the context
                x = layer(x , context)
            elif isinstance(layer,UNET_ResidualBlock) :
                # Residual block computes latent woth the time stamp
                x : layer( x , time)
            else :
                x = layer(x)
            
        return x
    
class UNET(nn.Module) :
    def __init__(self) :
        super().__init__()
        # Encoder decreases the number of channels and the image size , and then we upscale

        # Start with the encoder
        # SwithSequential , given a list of layers it will apply one after the other sequentially !
        self.encoders = nn.ModuleList([
            # (BatchSize , 4 , Heigth/8,Width / 8 )
            SwitchSequential(
                nn.Conv2d(4,320, kernel_Size = 3 , padding = 1),
            ),
            SwitchSequential(
                UNET_ResidualBlock(320,320),
                UNET_AttentionBlock(8,40)
            ),
            SwitchSequential(
                UNET_ResidualBlock(320,320),
                UNET_AttentionBlock(8,40)
            ),

             #(BatchSize , 320 , Heigth/8,Width / 8 ) ->  (BatchSize , 320, Heigth/16,Width / 16 ) 
            SwitchSequential(
                nn.Conv2d(320,320 , kernel_size = 3 ,strid = 2, padding = 1),
            ),
            SwitchSequential(
                UNET_ResidualBlock(320,640),
                UNET_AttentionBlock(8,80)
            ),
            SwitchSequential(
                UNET_ResidualBlock(640,640),
                UNET_AttentionBlock(8,80)
            ),

            #(BatchSize , 640, Heigth/16,Width / 16 ) ->  (BatchSize , 640 , Heigth/32 ,Width / 32) 
            SwitchSequential(
                nn.Conv2d(640,640 , kernel_size = 3 , strid = 2 , padding = 1),
            ),
            SwitchSequential(
                UNET_ResidualBlock(640,1280),
                UNET_AttentionBlock(8,160)
            ),
            SwitchSequential(
                UNET_ResidualBlock(1280,1280),
                UNET_AttentionBlock(8,160)
            ),

            #(BatchSize , 1280 , Heigth/32 ,Width / 32)  -> (BatchSize , 1280, Heigth/64 ,Width / 64) 
            SwitchSequential(
                nn.Conv2d(1280,1280 , kernel_size = 3 , strid = 2 , padding = 1),
            ),

            SwitchSequential(
                UNET_ResidualBlock(1280,1280)
            ),
            #(BatchSize , 1280 , Heigth/64 ,Width / 64) -> (BatchSize , 1280 , Heigth/64 ,Width / 64) 
            SwitchSequential(
                UNET_ResidualBlock(1280,1280)
            )
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),

            UNET_AttentionBlock(8,160),

            UNET_ResidualBlock(1280,1280),
        )

        self.decoder = nn.ModuleList([
            SwitchSequential(
                # The 2560 - is due to 1280 from bottle neck and the remaining 1280 is from the skip connection from encoder
                # (BatchSize , 2560 , Heigth/64 , Width / 64) -> (BatchSize , 1280 , Heigth/64 , Width / 64) 
                UNET_ResidualBlock(2560 , 1280)
            ),
             SwitchSequential(
                UNET_ResidualBlock(2560 , 1280)
            ),
             SwitchSequential(
                UNET_ResidualBlock(2560 , 1280), 
                Upsample(1280),
            ),
            SwitchSequential(
                UNET_ResidualBlock(2560,1280),
                UNET_AttentionBlock(8,160),
            ),
            SwitchSequential(
                UNET_ResidualBlock(2560,1280),
                UNET_AttentionBlock(8,160),
            ),
            SwitchSequential(
                UNET_ResidualBlock(1920,1280),
                UNET_AttentionBlock(8,160),
                Upsample(1280),
            ),
            SwitchSequential(
                UNET_ResidualBlock(1920,640),
                UNET_AttentionBlock(8,80),
            ),
            SwitchSequential(
                UNET_ResidualBlock(1280,640),
                UNET_AttentionBlock(8,80),
            ),
            SwitchSequential(
                UNET_ResidualBlock(960,640),
                UNET_AttentionBlock(8,80),
                Upsample(640),
            ),
            SwitchSequential(
                UNET_ResidualBlock(960,320),
                UNET_AttentionBlock(8,40),
            ),
            SwitchSequential(
                UNET_ResidualBlock(640,320),
                UNET_AttentionBlock(8,80),
            ),
            SwitchSequential(
                UNET_ResidualBlock(640,320),
                UNET_AttentionBlock(8,40),
            ),    
        ])



class Diffusion(nn.Module) :

    def __init__(self) :
        # To understand the timestamp of the image
        self.timeEmbedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)

    # This unet recives the latent and the promt embeddings(Context) and time embedding(tenstot)
    def forward(self,latent : torch.Tensor,context : torch.Tensor , time : torch.Tensor) :
        # latent : batchsize , 4 - channels in the latent, Heigth/8 , width/8
        # Prompt - (BatchSize , seqLen , Dim (728)) - state
        # time - (1,320) - vector of size 320 - same like the sine and cosine


        # (1,320) -> (1,1280)
        time = self.timeEmbedding(time)

        #Unet - latent - to another latent

        # (Batch,4 , Height/8, width/8) -> (Batch, 128, Height/8, width/8) (final output layer)
        # The last layer of unet - we need to go back to same numebr of features , the decoder

        # The unet predicts the noise from the time embedding and the latents , and then remove the noise and again predicst the noise and remove it .....

        output = self.unet(latent,context,time)

        #(Batch, 128, Height/8, width/8) (final output layer) -> (Batch,4 , Height/8, width/8)
        output = self.final(output)

        #(Batch,4 , Height/8, width/8)
        return output
    



       



