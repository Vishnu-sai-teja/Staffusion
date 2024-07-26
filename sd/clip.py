import torch
import torch.nn as nn
from torch.nn import fucntional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module) :
    # nVocab - length of the size of vocabulary , nTokens - sequence length
    def __init__(self,nVocab : int ,nEmbedd : int ,nTokens : int) :
        super().__init__()

        # nn.Embedding takes the length of the size of vocabulary and the sequence length
        self.tokenEmbedding = nn.Embedding(nVocab,nTokens)
        
        #Add a positional embedding - not sin/cos but rather learn by the model at the time od training
        self.positionalEmbedding = nn.Parameter(torch.zeros(nTokens,nEmbedd))

    def forward(self,tokens) :
        # BatchLength , seq_len -> batchlength , seq_len , embedding_dim
        x = self.tokenEmbedding(tokens)

        x += self.positionalEmbedding

        return x


class CLIPLayer(nn.Module) :
    def __init__(self,nHeads : int,nEmbedd :int) -> None :
        super().__init__()

        self.layerNorm1 = nn.LayerNorm(nEmbedd)

        self.attention = SelfAttention(nHeads,nEmbedd)

        self.layerNorm2 = nn.LayerNorm(nEmbedd)

        self.linear1 = nn.Linear(nEmbedd , 4 * nEmbedd)
        
        self.linear2 = nn.Linear(4 * nEmbedd , nEmbedd)

    def forward(self , x : torch.Tensor) -> torch.Tensor :
        # (BatchSize ,sef_len , dimensiton(nEmbedd))
         
        residual = x

        # Self Attention

        x = self.layerNorm1(x)

        x = self.attention(x , casualMask = True)

        x += residual 

        residual = x

        # Feed forward Network , to be precise

        x = self.layerNorm2(x)

        x = self.linear1(x)

        x *= torch.sigmoid(1.702 * x ) # Gelu activation function - published by the paper

        x = self.linear2(x)

        x += residual

        return x




class CLIP(nn.Module) :
    def __init__(self) :
        super().__init__()
        # (Dimension of the vocabulary , Embedding Size ,sequenceLength ) 
        self.embedding = CLIPEmbedding(48408,768,77)
        
        self.layers = nn.Module(
            # (Number of head of multi head attention , embedding size) for about 12 layers
            CLIPLayer(12,728) for i in range(12)
        )

        self.layerNorm = nn.LayerNorm(768)

    def forward(self, tokens : torch.LongTensor) -> torch.FloatTensor :
        # Toeken to long tensor
        tokens = tokens.type(torch.long)

        # Convert the tokens to embeddings
        # BatchLength , seq_len -> batchlength , seq_len , embedding_dim
        state = self.embedding(tokens)

        for layer in self.layers :
            state = layer(state)

        # (batchlength , seq_len , embedding_dim) 
        output = self.layerNorm(state)

        return output
