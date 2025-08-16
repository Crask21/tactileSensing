import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ===== GelSight Encoder =====
class GelSightEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace FC layer with Identity so we can add our own
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.fc = nn.Linear(num_ftrs, latent_dim)

    def forward(self, x):
        feats = self.backbone(x)      # (B, num_ftrs)
        feats = self.fc(feats)        # (B, latent_dim)
        return F.normalize(feats, dim=1)


# ===== Shadow Hand Encoder =====
class ShadowHandEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(2, 3), stride=1, padding=0),  # (B, 16, 2, 15)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(2, 3), stride=1),  # (B, 32, 1, 13)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 1 * 13, latent_dim)
        )

    def forward(self, x):
        feats = self.encoder(x)       # (B, latent_dim)
        return F.normalize(feats, dim=1)


# ===== Combined Model =====
class CrossModalModel(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.gelsight_encoder = GelSightEncoder(latent_dim)
        self.shadowhand_encoder = ShadowHandEncoder(latent_dim)

    def forward(self, gelsight, shadowhand):
        z_gel = self.gelsight_encoder(gelsight)
        z_hand = self.shadowhand_encoder(shadowhand)
        return z_gel, z_hand


# ===== InfoNCE Loss =====
def info_nce_loss(z1, z2, temperature=0.07):
    """
    Compute InfoNCE loss between two sets of embeddings.
    z1, z2: shape (B, D) normalized embeddings
    """
    batch_size = z1.shape[0]

    # Cosine similarity matrix
    logits = torch.matmul(z1, z2.T) / temperature

    labels = torch.arange(batch_size, device=z1.device)
    # Cross-entropy over rows (z1 as query, z2 as key) and columns (symmetry)
    loss_i = F.cross_entropy(logits, labels)
    loss_j = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_j) / 2


# ===== Example Training Step =====
if __name__ == "__main__":
    latent_dim = 512
    model = CrossModalModel(latent_dim=latent_dim)

    gelsight_batch = torch.randn(8, 3, 224, 224)   # 8 samples
    shadowhand_batch = torch.randn(8, 3, 3, 17)    # 8 samples

    z_gel, z_hand = model(gelsight_batch, shadowhand_batch)
    loss = info_nce_loss(z_gel, z_hand)

    print(f"z_gel shape: {z_gel.shape}, z_hand shape: {z_hand.shape}")
    print(f"Loss: {loss.item():.4f}")
