import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding3D(nn.Module):
    """Convert 3D image into patches and embed them"""
    def __init__(self, input_dim=40, patch_size=(8, 8, 8), embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Project patches to embedding dimension
        self.projection = nn.Sequential(
            Rearrange('b c (w pw) (h ph) (d pd) -> b (w h d) (pw ph pd c)',
                     pw=patch_size[0], ph=patch_size[1], pd=patch_size[2]),
            nn.Linear(patch_size[0] * patch_size[1] * patch_size[2] * input_dim, embed_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, channels, W, H, D)
        x = self.projection(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch, num_patches, embed_dim * 3)
        qkv = rearrange(qkv, 'b n (three h d) -> three b h n d', 
                       three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        out = torch.matmul(attention_probs, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.dropout(out)
        
        return out


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention and MLP"""
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    """3D Vision Transformer for binary classification"""
    def __init__(self, 
                 input_dim=40, 
                 input_size=(65, 77, 49),
                 patch_size=(13, 11, 7),  # Adjusted to divide input_size evenly
                 embed_dim=512,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 num_classes=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        # Calculate number of patches
        self.num_patches = (input_size[0] // patch_size[0]) * \
                          (input_size[1] // patch_size[1]) * \
                          (input_size[2] // patch_size[2])
        
        # Patch embedding
        self.patch_embed = PatchEmbedding3D(input_dim, patch_size, embed_dim)
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Layer norm and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create patch embeddings
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply layer norm
        x = self.norm(x)
        
        # Use class token for classification
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        
        return logits


# Example usage:
if __name__ == "__main__":
    # Create model
    model = ViT3D(
        input_dim=40,
        input_size=(65, 77, 49),
        patch_size=(13, 11, 7),  # Results in 5x7x7 = 245 patches
        embed_dim=512,
        depth=6,
        num_heads=8,
        num_classes=2
    )
    
    # Test with sample input
    x = torch.randn(2, 40, 65, 77, 49)  # (batch, channels, W, H, D)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")