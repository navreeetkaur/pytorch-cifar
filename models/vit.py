import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8,  attn_drop=0., proj_drop=0., qkv_bias = False):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.to_out = nn.Sequential(
        	nn.Dropout(attn_drop)
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(emb_dim, Attention(emb_dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x)
            x = feedforward(x)
        return x

class ViT(nn.Module):
	def __init__(self, image_size, patch_size, num_classes=10, emb_dim, depth, mlp_dim, heads=64, channels = 3, dropout = 0., emb_dropout = 0.):
		super(VGG, self).__init__()
		assert image_size % patch_size == 0, 'cannot have fractional number of patches - image size should be divisible by patch size'
		self.image_size = image_size
		self.patch_size = patch_size
		# effective input sequence length for the Transformer
		num_patches = (image_size // patch_size) ** 2
		patch_dim = channels * (patch_size ** 2)

		# latten the patches and map to D dimensions with a trainable linear projection
        self.patch_embedding = nn.Linear(patch_dim, emb_dim)

        # prepend a learnable embedding to the sequence of embedded patches  
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
		self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
		self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(emb_dim, depth, heads, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x[:, 0])
        return self.mlp_head(x)