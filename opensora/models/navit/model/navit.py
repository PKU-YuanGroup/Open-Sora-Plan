import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataset.data_util import image_packaging

# Postition embedding used in the original paper
def fixed_position_encoding(position,hidden_dim):
    assert hidden_dim==(hidden_dim//2)*2
    pe=torch.zeros(*(position.shape),hidden_dim,device=position.device)
    factor=torch.exp(torch.arange(0,hidden_dim,2,device=position.device)*(-math.log(10000)/hidden_dim)).unsqueeze(0).unsqueeze(1)

    pe[:,:,0::2]=torch.sin(position.unsqueeze(-1)*factor)
    pe[:,:,1::2]=torch.cos(position.unsqueeze(-1)*factor)
    return pe

def fourier_position_encoding(xy_coor:torch.Tensor,hidden_dim,coor_weight,element_weight,max_token_lim=10000):
    assert hidden_dim==(hidden_dim//2)*2
    pe=torch.zeros(*(xy_coor.shape),hidden_dim,device=xy_coor.device)

    pe[:,:,1::2]=torch.sin(xy_coor.float()/max_token_lim*coor_weight.unsqueeze(0).unsqueeze(1)*2*math.pi).unsqueeze(-1)*element_weight.unsqueeze(0).unsqueeze(1)
    pe[:,:,0::2]=torch.cos(xy_coor.float()/max_token_lim*coor_weight.unsqueeze(0).unsqueeze(1)*2*math.pi).unsqueeze(-1)*element_weight.unsqueeze(0).unsqueeze(1)
    return pe

# Following the original method, we use q-k normalization
class QKNorm(nn.Module):
    def __init__(self, hidden_dim,num_head):
        super().__init__()
        self.scale = hidden_dim** 0.5
        self.gamma = nn.Parameter(torch.ones(num_head,hidden_dim))

    def forward(self, x):
        return F.normalize(x, dim = -1)* self.scale * self.gamma.unsqueeze(0)

# Following the original method, enable postion embedding type to be of learned,fixed or fourier
class PatchEmbedding(nn.Module):
    def __init__(self,max_img_size, patch_size, emb_dim,pos_emb_type='learned'):
        super().__init__()
        max_c,max_h,max_w=max_img_size
        self.max_w=max_w
        self.max_h=max_h
        self.max_c=max_c
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        assert max_w % patch_size == 0 and max_h % patch_size == 0, 'image width and height must be divisible by patch size'

        token_dim=max_c*patch_size**2
        self.proj=nn.Linear(token_dim,emb_dim)
        self.pos_emb_type=pos_emb_type

        if pos_emb_type=='learned':
            self.pos_emb_x = nn.Embedding(max_h//patch_size,emb_dim)
            self.pos_emb_y = nn.Embedding(max_w//patch_size,emb_dim)
        elif pos_emb_type=='fourier':
            self.coor_weight=nn.Parameter(torch.randn(2))
            self.element_weight=nn.Parameter(torch.randn(emb_dim//2))
        elif pos_emb_type=='fixed':
            self.pos_emb = fixed_position_encoding(torch.arange(max(max_w,max_h)//patch_size,device=self.device()).unsqueeze(0),emb_dim)
        else:    
            raise ValueError('pos_emb_type error')
        
    def forward(self,x,xy_coor):
        x=self.proj(x).reshape(x.shape[0],-1,self.emb_dim)
        if self.pos_emb_type=='fourier':
            x+=fourier_position_encoding(xy_coor,self.emb_dim,self.coor_weight,self.element_weight,max_token_lim=max(self.max_w,self.max_h)//self.patch_size)
        elif self.pos_emb_type=='learned':
            emb_x=self.pos_emb_x(xy_coor[:,:,0])
            emb_y=self.pos_emb_y(xy_coor[:,:,1])
            x=x+emb_x+emb_y
        elif self.pos_emb_type=='fixed':
            x=x+self.pos_emb[xy_coor[:,:,0]]
            x=x+self.pos_emb[xy_coor[:,:,1]]
        return x

class Attention(nn.Module):
    def __init__(self,hidden_dim,kv_dim,q_dim,num_head = 8,dropout_rate = 0.0):
        super().__init__()
        self.num_head = num_head
        self.hidden_dim=hidden_dim
        assert hidden_dim % num_head == 0, 'hidden_dim must be divisible by num_head'
        assert q_dim==hidden_dim, 'q_dim must be equal to hidden_dim'
        self.head_dim=hidden_dim // num_head
        self.kv_proj=nn.Linear(kv_dim,2*hidden_dim)
        self.q_proj=nn.Linear(q_dim,hidden_dim)
        self.q_norm = QKNorm(self.head_dim,num_head)
        self.k_norm = QKNorm(self.head_dim,num_head)
        self.out_proj = nn.Linear(hidden_dim,hidden_dim, bias = False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        x,
        context = None,
        attn_mask = None,
        padding_mask = None
    ):
        if context is None:
            context = x
        assert context.shape[-1]==x.shape[-1], 'context and x must have the same hidden dimension'

        q=self.q_proj(x).reshape(x.shape[0],x.shape[1],self.num_head,-1)
        k,v=self.kv_proj(context).chunk(2,dim=-1)
        k=k.reshape(context.shape[0],context.shape[1],self.num_head,-1)
        v=v.reshape(context.shape[0],context.shape[1],self.num_head,-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_weight=torch.einsum('bqhd,bkhd->bhqk',q,k)
        if attn_mask is not None:
            attn_weight=attn_weight.masked_fill(attn_mask.unsqueeze(-3),float('-inf'))
        if padding_mask is not None:
            attn_weight=attn_weight.masked_fill(padding_mask.unsqueeze(-3),float('-inf'))
        attn=torch.softmax(attn_weight,dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhqk,bkhd->bqhd',attn,v).reshape(x.shape[0],x.shape[1],-1)
        return self.out_proj(out)

class encoder_layer(nn.Module):
    def __init__(self,hidden_dim,kv_dim,q_dim,expand_dim,num_head = 8,dropout_rate = 0.0):
        super().__init__()
        self.attn = Attention(hidden_dim,kv_dim,q_dim,num_head,dropout_rate)
        self.ff1=nn.Linear(hidden_dim,expand_dim)
        self.ff2=nn.Linear(expand_dim,hidden_dim)
        self.ln1=nn.LayerNorm(hidden_dim,elementwise_affine=False)
        self.ln2=nn.LayerNorm(hidden_dim,elementwise_affine=False)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,x,context=None,attn_mask=None,padding_mask=None):
        res=x
        x=self.attn(x,context,attn_mask,padding_mask)
        x=self.ln1(x+res)
        res=x
        x=self.ff1(x)
        x=F.gelu(x)
        x=self.dropout(x)
        x=self.ff2(x)
        x=self.dropout(x)
        x=self.ln2(x+res)

        return x

class encoder(nn.Module):
    def __init__(self,num_layer,hidden_dim,kv_dim,q_dim,expand_dim,num_head = 8,dropout_rate = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layer):
            self.layers.append(encoder_layer(hidden_dim,kv_dim,q_dim,expand_dim,num_head,dropout_rate))

        self.input_ln = nn.LayerNorm(hidden_dim,elementwise_affine=False)

    def forward(self,x,context=None,attn_mask=None,padding_mask=None):
        for i,layer in enumerate(self.layers):
            x=layer(x,context,attn_mask,padding_mask)

        return self.input_ln(x)

class NaViT(nn.Module):
    def __init__(
        self,
        num_layer,
        hidden_dim,
        kv_dim,
        q_dim,
        expand_dim,
        max_img_size, 
        patch_size, 
        num_class,
        num_head = 8,
        dropout_rate = 0.0,
        pos_emb_type='learned',
        max_token_lim = 1024,
    ):
        super().__init__()
        max_c,max_h,max_w=max_img_size
        assert (max_h % patch_size) == 0 and (max_w % patch_size) == 0, f'image width and height must be divisible by patch size {patch_size}'

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.token_dim=max_c*patch_size**2
        self.max_token_lim = max_token_lim
        self.patch_embedding=PatchEmbedding(max_img_size,patch_size,hidden_dim,pos_emb_type)
        self.encoder = encoder(num_layer,hidden_dim,kv_dim,q_dim,expand_dim,num_head,dropout_rate)

        self.pool_q = nn.Parameter(torch.randn(hidden_dim))
        self.pool = Attention(hidden_dim,hidden_dim,hidden_dim,num_head,dropout_rate)

        self.class_logit_proj = nn.Linear(hidden_dim,num_class)

    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        images,
        token_dropout_rate=0.0,
        cls=True    
    ):

        image_packages = image_packaging(
            images,
            patch_size = self.patch_size,
            max_token_lim = self.max_token_lim,
            token_dropout_rate=token_dropout_rate,
        )

        batched_img = []
        batched_pos = []
        batched_idx = []

        batched_len=[]
        max_len=0
        package_size=[]
        for package in image_packages:
            package_size.append(len(package))

            image_seq = torch.torch.zeros((0,self.token_dim), device = self.device(), dtype = torch.float)
            image_pos = torch.torch.zeros((0,), device = self.device(), dtype = torch.long)
            image_idx = torch.torch.zeros((0,), device = self.device(), dtype = torch.long)

            for idx, image in enumerate(package):
                num_patch_h=(image.shape[-2]//self.patch_size)
                num_patch_w=(image.shape[-1]//self.patch_size)

                xy_coor = torch.stack(torch.meshgrid((
                    torch.arange(num_patch_h, device = self.device()),
                    torch.arange(num_patch_w, device = self.device())
                ), indexing = 'ij'), dim = -1)

                xy_coor = xy_coor.reshape(-1,2)
                seq = rearrange(image, 'c (x nx) (y ny) -> (x y) (c nx ny)', nx = self.patch_size, ny = self.patch_size)

                if token_dropout_rate > 0:
                    selected_len = int(seq.shape[0] * (1 - token_dropout_rate))
                    select_indices = torch.randperm(seq.shape[0], device = self.device())[:selected_len]
                    seq = seq[select_indices]
                    xy_coor = xy_coor[select_indices]

                image_seq = torch.cat([image_seq,seq],dim=0)
                image_pos = torch.cat([image_pos,xy_coor],dim=0)
                image_idx = torch.cat((image_idx,torch.full((seq.shape[0],),idx,device=self.device(),dtype=torch.long)))

            batched_img.append(image_seq)
            batched_pos.append(image_pos)
            batched_idx.append(image_idx)

            batched_len.append(image_seq.shape[0])
            if image_seq.shape[0]>max_len:
                max_len=image_seq.shape[0]

        batched_img = nn.utils.rnn.pad_sequence(batched_img,batch_first=True)
        batched_pos = nn.utils.rnn.pad_sequence(batched_pos,batch_first=True)
        batched_idx = nn.utils.rnn.pad_sequence(batched_idx,batch_first=True)

        batched_len = torch.Tensor(batched_len, device = self.device()).long()

        attn_mask = batched_idx.unsqueeze(-1) != batched_idx.unsqueeze(1)
        padding_mask = batched_len.unsqueeze(-1) <= torch.arange(max_len, device = self.device(), dtype = torch.long).unsqueeze(0)
        padding_mask = padding_mask.unsqueeze(1)

        package_size = torch.Tensor(package_size, device = self.device()).long() 

        x = self.patch_embedding(batched_img,batched_pos)        
        x = self.encoder(x,attn_mask=attn_mask,padding_mask=padding_mask)

        max_package_size = package_size.max().item()
        pool_q = self.pool_q.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],max_package_size,1)

        image_id = torch.arange(max_package_size, device = self.device(), dtype = torch.long)

        pool_mask = image_id.unsqueeze(0).unsqueeze(-1) != batched_idx.unsqueeze(-2)

        x = self.pool(pool_q,x,attn_mask=pool_mask,padding_mask=padding_mask)
        image_id = image_id.unsqueeze(0) < package_size.unsqueeze(-1)
        x = x[image_id]
        if cls:
            x = self.class_logit_proj(x)
        return x
    