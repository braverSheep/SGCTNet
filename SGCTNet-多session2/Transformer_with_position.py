import torch.nn as nn
import torch
import numpy as np

# Transformer ²¿·Ö
class ScaledDotProductAttention(nn.Module):
    def __init__(self,n_head,d_k,dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = n_head
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V,A_ds):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        # chan_weighting = A_ds.repeat(self.n_head,1,1)
        # attn = attn*chan_weighting
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context
    
class MultiHeadAttention(nn.Module):#n_layers=2,n_heads=5,d_model=self.band_num*2,d_k=8,d_v=8,d_ff=10
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
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = self.ScaledDotProductAttention(Q, K, V,A_ds)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
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
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.dropout(self.fc(inputs))
        return nn.LayerNorm(self.d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]
    
class EncoderLayer(nn.Module):#n_layers=2,n_heads=5,d_model=self.band_num*2,d_k=8,d_v=8,d_ff=10
    def __init__(self,n_heads,d_model,d_k,d_v,d_ff=10, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dropout = dropout
        self.enc_self_attn = MultiHeadAttention(n_heads,d_model,d_k,d_v,dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model,d_ff,dropout)

    def forward(self, enc_inputs,A_ds):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,A_ds) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs