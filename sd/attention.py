import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module) :
    def __init__(self , nHeads:int, dEmbedding:int , in_proj_bias : True , out_proj_bias : True) :
        super(SelfAttention,self).__init__() 
        '''
            Bias we applybefore doing attention  - So in_proj_bias
        '''
        self.in_proj = nn.Linear(dEmbedding , 3 * dEmbedding , bias = in_proj_bias)
        self.out_proj = nn.Linear(dEmbedding,dEmbedding,bias = out_proj_bias)
        self.nHeads = nHeads
        self.dHeads = dEmbedding//nHeads

    def forward(self,x : torch.Tensor , casualMask : False) -> torch.Tensor :
        #X : (BatchSize,Seqlen,dim)
        inputShape = x.shape
        batchSize , sequenceLength , dEmbedd = inputShape

        interimshape = (batchSize , sequenceLength,self.nHeads,self.dHeads)

        # (BatchSize , seq_len,dim) -> (BatchSize , Seqlen ,H, Dim/H)
        q,k,v = self.in_proj(x).chunk(3,dim = 1)

        #(BatchSize , Seqlen ,H, Dim/H) -> (BatchSize , H , Seqlen , Dim/H)
        q = q.view(interimshape).transpose(1,2)
        k = k.view(interimshape).transpose(1,2)
        v = v.view(interimshape).transpose(1,2)

        # (BatchSize, H , seqLen , seqLen)

        weight = q @ k.transpose(-1,-2) # @ is used for matrix multiplication

        # Apply mask on the 
        if casualMask :
            mask = torch.ones_like(weight,dtype = torch.bool).triu(1)
            weight.masked_fill(mask,-torch.inf)

        weight /= math.sqrt(self.dHead)

        weight = F.softmax(weight,dim = -1)

        # (BatchSie, h , Seq_len, Seq_len) @ (BatchSize , H , Seq_len , Dim/H) -> (BatchSize , H , seqlen, Dim/h)
        output = weight @ v

        # (BatchSize , H , seqlen, Dim/h) -> (Batsize , seqLen  , H , Dim/H)
        output = output.transpose(1,2)

        output = output.reshape(inputShape)

        output = self.out_proj(output)

        return output
    

class CrossAttention(nn.Module) :
    def __init__(self , nHeads : int ,dEmbedd : int , dCross : int , in_proj_bias : True , out_proj_bias = Tre) :
        # Same ut query from one side and key , value from other side
        super()._init__()

        self.q_proj = nn.Linear(dEmbedd ,dEmbedd , bias = in_proj_bias)
        self.k_proj = nn.Linear(dCross ,dEmbedd , bias = in_proj_bias)     
        self.v_proj = nn.Linear(dCross ,dEmbedd , bias = in_proj_bias)

        self.out_proj = nn.Linear(dEmbedd , dEmbedd , bias = out_proj_bias)
        self.nHeads = nHeads
        self.dHeads = dEmbedd // nHeads

    def forward(self , x ,y ) :
        # X - our latents - the query values
        # X - latent (BatchSize , seqlen_q , dim_q)
        # y - key,value pairs  , which help in udnersatnding teh context format
        # y - (context) - (BatcSize , seqlen_kv , dim_kv) = (BatchSize , 77 , 768)

        inputShape = x.shape
        batchSize , seqLen , dEmbedd = inputShape


        interimShape = (batchSize , -1 , self.nHeads , self.dHeads)

        # Multiply query by wq

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interimShape).transpose(1,2)
        k = k.view(interimShape).transpose(1,2)
        v = v.view(interimShape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)

        weight /= math.sqrt(self.dHeads)

        # In this case we have no casual mask , her we are trying to relate prompt with the pixel , any token watch teh pixel and vice versa
        weight = F.softmax(weight , dim = -1)

        output = weight @ v

        output = output.transpose(1,2).continuous()

        output = output.view(inputShape)

        output = self.out_proj(output)

        return output



        

        
 

