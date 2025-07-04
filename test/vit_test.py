import torch
import torch.nn as nn 

class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        #              图像大小     每个patch大小    输入维度      
        super().__init__()
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1]) #patch 的网格大小 224//16 = 14  (14, 14)
        self.num_patches = self.grid_size[0] * self.grid_size[1] # pathc 总数 14*14 = 196
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self,x):
        B,C,H,W = x.shape
        assert H==self.image_size[0] and W==self.image_size[1],\
            f"输入图像大小{H}*{W}与期望的图片大小{self.image_size[0]*self.image_size[1]}"
        x = self.proj(x)
        #B,3,244,244 -> B,768,14,14 -> B,768,196 -> B,196,768
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                dim,
                num_heads = 8,
                qkv_bias = False,
                qkv_scale = None,
                attn_drop = 0.,
                proj_drop = 0.,      ):
        super.__init__()
        self.num_heads = num_heads
        head_dim = dim / num_heads
        self.scale = qkv_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_biask) #通过全连接层生成QKV, 为了并行运算
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(pro_drop)
        #将每一个head的输出做concat连接 
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):
        B,N,C = x.shape #batch, num_patchs+1, embed_dim
        #B, N ,3*C -> B, N , 3, self.num_heads, C//self.num_heads
        #B, N , 3, self.num_heads, C//self.num_heads -> 3, B, self.num_heads, N, C//self.num_heads
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4) 
        #用切片的形式选取QKV, B, self.num_heads, N, C//self.num_heads
        q,k,v = qkv[0], qkv[1], qkv[2]
        # 计算q k 的点积计算注意力分数:{B, self.num_heads, N, C//self.num_heads}
        #        k.transpose(-2,-1):{B, self.num_heads,C//self.num_heads,N}
        attn = (q @ k.transpose(-2,-1)) * self.scale # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1) # 对每行进行处理
        # 注意力权重对V进行加权求和
        #
        x = (attn @ v.transpose(-2,-1)).reshape(B,N,C)
        #通过线性变换映射变回原本的维度
        x = self.proj(x)
        x = self.proj_drop(x) #防止过拟合
        return x
class MLP(nn.Module):
    def __init__(self,in_features,hidden_features=None,out_features=None,act_layer=nn.GELU,drop=0.):
        super.__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Block(nn.Module):
    