import torch.nn as nn
import torch
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self,n_head,d_k,dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V,A_ds):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) 
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn) 
        context = torch.matmul(attn, V)
        return context
    
class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,d_model,d_k,d_v,dropout):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.n_heads,self.d_k,dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)
        nn.init.xavier_normal_(self.fc.weight)
     
    def forward(self, input_Q, input_K, input_V,A_ds):
       
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        context = self.ScaledDotProductAttention(Q, K, V,A_ds)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context) 
        output = self.dropout(output)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ELU(),#nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        nn.init.xavier_normal_(self.fc[0].weight)
        nn.init.xavier_normal_(self.fc[2].weight)
    
    def forward(self, inputs):
        residual = inputs
        output = self.dropout(self.fc(inputs))
        return nn.LayerNorm(self.d_model).cuda()(output + residual) 
    
class EncoderLayer(nn.Module):#n_layers=2,n_heads=5,d_model=self.band_num*2,d_k=8,d_v=8,d_ff=10
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff=10, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dropout = dropout
        self.enc_self_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v,dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff,dropout)

    def forward(self, enc_inputs,A_ds):
        
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,A_ds) 
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs