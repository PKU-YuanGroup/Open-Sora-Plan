# In order to improve model performance, we referred to Latte to design our video NaViT,
#  which requires the input video frame rate to be the same, but the resolution of each 
# frame can be different.

from model.navit import *
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
USE_FLASH = False
try:
    from flash_attn import flash_attn_func
    USE_FLASH = True
except:
    pass

from dataset.data_util import image_packaging

class st_PatchEmbedding(PatchEmbedding):
    def forward(self,x,xy_coor):
        x=self.proj(x).reshape(x.shape[0],x.shape[1],-1,self.emb_dim)
        if self.pos_emb_type=='fourier':
            x+=fourier_position_encoding(xy_coor,self.emb_dim,self.coor_weight,self.element_weight,max_token_lim=max(self.max_w,self.max_h)//self.patch_size).unsqueeze(1)
        elif self.pos_emb_type=='learned':
            emb_x=self.pos_emb_x(xy_coor[:,:,0])
            emb_y=self.pos_emb_y(xy_coor[:,:,1])
            x=x+emb_x.unsqueeze(1)+emb_y.unsqueeze(1)
        elif self.pos_emb_type=='fixed':
            x=x+self.pos_emb[xy_coor[:,:,0]].unsqueeze(1)
            x=x+self.pos_emb[xy_coor[:,:,1]].unsqueeze(1)
        return x

class SpatialAttention(nn.Module):
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

        q=self.q_proj(x).reshape(x.shape[0],x.shape[1],x.shape[2],self.num_head,-1)
        k,v=self.kv_proj(context).chunk(2,dim=-1)
        k=k.reshape(context.shape[0],context.shape[1],x.shape[2],self.num_head,-1)
        v=v.reshape(context.shape[0],context.shape[1],x.shape[2],self.num_head,-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        attn_weight=torch.einsum('btqhd,btkhd->bthqk',q,k)
        if attn_mask is not None:
            attn_weight=attn_weight.masked_fill(attn_mask.unsqueeze(-3).unsqueeze(-4),float('-inf'))
        if padding_mask is not None:
            attn_weight=attn_weight.masked_fill(padding_mask.unsqueeze(-3).unsqueeze(-4),float('-inf'))
        attn=torch.softmax(attn_weight,dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bthqk,btkhd->btqhd',attn,v).reshape(x.shape[0],x.shape[1],x.shape[2],-1)
        return self.out_proj(out)

class TemporalAttention(nn.Module):
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
        context = None
    ):
        if context is None:
            context = x
        assert context.shape[-1]==x.shape[-1], 'context and x must have the same hidden dimension'
        b,l,d=x.shape
        q=self.q_proj(x).reshape(b,l,self.num_head,-1)
        k,v=self.kv_proj(context).chunk(2,dim=-1)
        k=k.reshape(context.shape[0],context.shape[1],self.num_head,-1)
        v=v.reshape(context.shape[0],context.shape[1],self.num_head,-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if USE_FLASH:
            x = flash_attn_func(q,k,v,causal=False)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            x = x.permute(0, 2, 1, 3)
        return self.out_proj(x.reshape(b, l, -1))

class st_encoder_layer(nn.Module):
    def __init__(self,hidden_dim,kv_dim,q_dim,expand_dim,num_head = 8,dropout_rate = 0.0):
        super().__init__()
        self.attn_s = SpatialAttention(hidden_dim,kv_dim,q_dim,num_head,dropout_rate)
        self.attn_t = TemporalAttention(hidden_dim,kv_dim,q_dim,num_head,dropout_rate)

        self.head_dim=hidden_dim // num_head
        self.num_head = num_head
            
        self.ff1=nn.Linear(hidden_dim,expand_dim)
        self.ff2=nn.Linear(expand_dim,hidden_dim)
        self.ln1=nn.LayerNorm(hidden_dim,elementwise_affine=False)
        self.ln2=nn.LayerNorm(hidden_dim,elementwise_affine=False)
        self.dropout=nn.Dropout(dropout_rate)

    def forward(self,x,context=None,attn_mask=None,padding_mask=None):
        if context is None:
            context = x
        b,t,l,d=x.shape
        res=x
        x=self.attn_s(x,context,attn_mask,padding_mask)
        x=x+res
        x=rearrange(x,'b t l d -> (b l) t d')
        res=x
        context=rearrange(context,'b t l d -> (b l) t d')
        x=self.attn_t(x,context)
        x=x+res
        x=rearrange(x,'(b l) t d -> b t l d',l=l)
        x=self.ln1(x)
        res=x
        x=self.ff1(x)
        x=F.gelu(x)
        x=self.dropout(x)
        x=self.ff2(x)
        x=self.dropout(x)
        x=self.ln2(x+res)

        return x
    
class st_encoder(nn.Module):
    def __init__(self,num_layer,hidden_dim,kv_dim,q_dim,expand_dim,num_head = 8,dropout_rate = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layer):
            self.layers.append(st_encoder_layer(hidden_dim,kv_dim,q_dim,expand_dim,num_head,dropout_rate))

        self.input_ln = nn.LayerNorm(hidden_dim,elementwise_affine=False)

    def forward(self,x,context=None,attn_mask=None,padding_mask=None):
        for i,layer in enumerate(self.layers):
            x=layer(x,context,attn_mask,padding_mask)

        return self.input_ln(x)

class st_NaViT(nn.Module):
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
        max_t,max_c,max_h,max_w=max_img_size
        assert (max_h % patch_size) == 0 and (max_w % patch_size) == 0, f'image width and height must be divisible by patch size {patch_size}'

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.token_dim=max_c*patch_size**2
        self.max_token_lim = max_token_lim
        self.max_time_lim = max_t
        self.patch_embedding=st_PatchEmbedding((max_c,max_h,max_w),patch_size,hidden_dim,pos_emb_type)
        self.time_embedding=fixed_position_encoding(torch.arange(max_t).unsqueeze(0),hidden_dim)
        self.encoder = st_encoder(num_layer,hidden_dim,kv_dim,q_dim,expand_dim,num_head,dropout_rate)

        self.pool_q1 = nn.Parameter(torch.randn(hidden_dim))
        self.pool1 = TemporalAttention(hidden_dim,hidden_dim,hidden_dim,num_head,dropout_rate)

        self.pool_q2 = nn.Parameter(torch.randn(hidden_dim))
        self.pool2 = Attention(hidden_dim,hidden_dim,hidden_dim,num_head,dropout_rate)

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
            version='v'
        )

        batched_img = []
        batched_pos = []
        batched_idx = []

        batched_len=[]
        max_len=0
        package_size=[]
        for package in image_packages:
            package_size.append(len(package))

            image_seq = torch.torch.zeros((self.max_time_lim,0,self.token_dim), device = self.device(), dtype = torch.float)
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
                seq = rearrange(image, 't c (x nx) (y ny) -> t (x y) (c nx ny)', nx = self.patch_size, ny = self.patch_size)

                if token_dropout_rate > 0:
                    selected_len = int(seq.shape[0] * (1 - token_dropout_rate))
                    select_indices = torch.randperm(seq.shape[0], device = self.device())[:selected_len]
                    seq = seq[select_indices]
                    xy_coor = xy_coor[select_indices]

                image_seq = torch.cat([image_seq,seq],dim=1)
                image_pos = torch.cat([image_pos,xy_coor],dim=0)
                image_idx = torch.cat((image_idx,torch.full((seq.shape[1],),idx,device=self.device(),dtype=torch.long)))
            batched_img.append(rearrange(image_seq,'t l d -> l (t d)'))
            batched_pos.append(image_pos)
            batched_idx.append(image_idx)

            batched_len.append(image_seq.shape[1])
            if image_seq.shape[1]>max_len:
                max_len=image_seq.shape[1]

        batched_img = nn.utils.rnn.pad_sequence(batched_img,batch_first=True)
        batched_pos = nn.utils.rnn.pad_sequence(batched_pos,batch_first=True)
        batched_idx = nn.utils.rnn.pad_sequence(batched_idx,batch_first=True)

        batched_len = torch.Tensor(batched_len, device = self.device()).long()

        attn_mask = batched_idx.unsqueeze(-1) != batched_idx.unsqueeze(1)
        padding_mask = batched_len.unsqueeze(-1) <= torch.arange(max_len, device = self.device(), dtype = torch.long).unsqueeze(0)
        padding_mask = padding_mask.unsqueeze(1)

        package_size = torch.Tensor(package_size, device = self.device()).long()        

        x = rearrange(batched_img,'b l (t d) -> b t l d',t=self.max_time_lim)
        x = self.patch_embedding(x,batched_pos) 
        x = rearrange(x,'b t l d -> (b l) t d',t=self.max_time_lim)
        x = x + self.time_embedding[:x.shape[1]]
        x = rearrange(x,'(b l) t d -> b t l d',l=batched_img.shape[1])      
        x = self.encoder(x,attn_mask=attn_mask,padding_mask=padding_mask)

        max_package_size = package_size.max().item()
        x = rearrange(x,'b t l d -> (b l) t d')
        pool_q1 = self.pool_q1.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],1,1)
        x = self.pool1(pool_q1,x)
        x = rearrange(x,'(b l) 1 d -> b l d',l=batched_img.shape[1])

        image_id = torch.arange(max_package_size, device = self.device(), dtype = torch.long)

        pool_mask = image_id.unsqueeze(0).unsqueeze(-1) != batched_idx.unsqueeze(1)


        pool_q2 = self.pool_q2.unsqueeze(0).unsqueeze(1).repeat(x.shape[0],max_package_size,1)
        x = self.pool2(pool_q2,x,attn_mask=pool_mask,padding_mask=padding_mask)
        image_id = image_id < package_size.unsqueeze(-1)
        x = x[image_id]
        if cls:
            x = self.class_logit_proj(x)
        return x