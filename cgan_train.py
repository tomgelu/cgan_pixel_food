# cgan_train.py (patched for stability and clarity)
import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import argparse
import glob

# ========== CONFIG ==========
IMAGE_SIZE = 128
BATCH_SIZE = 1
EPOCHS = 1000
LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
DATASET_DIR = "dataset"
WEIGHTS_DIR = "weights"
CHECKPOINT_DIR = "checkpoints"

CSV_PATH = "combos/combo_metadata.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ========== TAGS ==========
ALL_TAGS = [
    "fish", "meat", "mushroom", "root_vegetables", "frog_legs",
    "grilled", "stewed", "raw", "baked",
    "curry", "brown_broth", "tomato_sauce", "green_emulsion",
    "chili_flakes", "flowers", "herbs"
]
TAG_DIM = len(ALL_TAGS)
TAG2IDX = {tag: i for i, tag in enumerate(ALL_TAGS)}

# ========== DATASET ==========
class FoodDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.df["filepath"] = self.df["filename"].apply(lambda f: os.path.join(self.image_dir, f))
        self.df = self.df[self.df["filepath"].apply(os.path.exists)].reset_index(drop=True)
        print(f"📦 Loaded {len(self.df)} samples from {self.image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.df)

    def encode_tags(self, row):
        vec = np.zeros(TAG_DIM, dtype=np.float32)
        for col in ["ingredients", "cooking_methods", "sauce", "garnishes"]:
            if pd.isna(row[col]):
                continue
            tags = row[col].split("|")
            for tag in tags:
                if tag != "none" and tag in TAG2IDX:
                    vec[TAG2IDX[tag]] = 1.0
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        tag_vec = self.encode_tags(row)
        return self.transform(image), torch.tensor(tag_vec, dtype=torch.float32)

# ========== MODELS ==========
class Generator(nn.Module):
    def __init__(self, latent_dim, tag_dim):
        super().__init__()
        self.input_dim = latent_dim + tag_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256 * 8 * 8),
            nn.ReLU(True)
        )
        self.net = nn.Sequential(
            nn.Unflatten(1, (256, 8, 8)),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.Conv2d(64, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(True),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.Conv2d(32, 3, 3, padding=1),
    nn.Tanh()
        )

    def forward(self, z, tags):
        x = torch.cat([z, tags], dim=1)
        x = self.fc(x)
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, tag_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        # Output of conv: [B, 512, 4, 4] => 512*4*4 = 8192
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8 + tag_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, tags):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, tags], dim=1)
        return self.fc(x)

# ========== TRAINING ==========
def save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, loss_G, loss_D, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss_G': loss_G,
        'loss_D': loss_D
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename, generator, discriminator, optimizer_G, optimizer_D):
    """Load training checkpoint."""
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return start_epoch
    return 0

def get_latest_checkpoint():
    """Find the latest checkpoint file in the checkpoints directory."""
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    def get_epoch_number(filename):
        return int(os.path.basename(filename).split('_')[2])
    
    latest_checkpoint = max(checkpoint_files, key=get_epoch_number)
    return latest_checkpoint

def train(resume_from=None):
    full_dataset = FoodDataset(CSV_PATH, DATASET_DIR)
    subset_indices = list(range(len(full_dataset)))
    dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(LATENT_DIM, TAG_DIM).to(DEVICE)
    discriminator = Discriminator(TAG_DIM).to(DEVICE)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    adversarial_loss = nn.BCELoss()
    
    start_epoch = 0
    # If no specific checkpoint is provided, try to find the latest one
    if resume_from is None:
        resume_from = get_latest_checkpoint()
        if resume_from:
            print(f"Found latest checkpoint: {resume_from}")
    
    if resume_from:
        start_epoch = load_checkpoint(resume_from, generator, discriminator, optimizer_G, optimizer_D)

    for epoch in range(start_epoch, EPOCHS):
        for i, (imgs, tags) in enumerate(dataloader):
            real_imgs = imgs.to(DEVICE)
            tags = tags.to(DEVICE)
            batch_size = real_imgs.size(0)

            valid = torch.full((batch_size, 1), 0.9, device=DEVICE)
            fake = torch.full((batch_size, 1), 0.0, device=DEVICE)

            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z, tags)
            g_loss = adversarial_loss(discriminator(gen_imgs, tags), valid)
            g_loss.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, tags), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), tags), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1:03}.pth")
            save_checkpoint(epoch, generator, discriminator, optimizer_G, optimizer_D, g_loss.item(), d_loss.item(), checkpoint_path)
            
        if (epoch + 1) % 100 == 0:
            vutils.save_image(gen_imgs.data[:1], f"{OUTPUT_DIR}/sample_epoch_{epoch+1:03}.png", normalize=True)
    
    torch.save(generator.state_dict(), os.path.join(WEIGHTS_DIR, "generator_final.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CGAN with checkpoint support')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume training from (optional)')
    args = parser.parse_args()
    
    train(resume_from=args.resume)
