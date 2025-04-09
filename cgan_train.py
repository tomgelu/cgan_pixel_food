# cgan_train.py (patched for stability and clarity with logging + improved loss + diffaug)
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
from torch.utils.tensorboard import SummaryWriter

# ========== CONFIG ==========
IMAGE_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 1000
LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
DATASET_DIR = "dataset"
WEIGHTS_DIR = "weights"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

CSV_PATH = "combos/combo_metadata.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ========== TAGS ==========
ALL_TAGS = [
    "fish", "meat", "mushroom", "root_vegetables", "frog_legs",
    "grilled", "stewed", "raw", "baked",
    "curry", "brown_broth", "tomato_sauce", "green_emulsion",
    "chili_flakes", "flowers", "herbs"
]
TAG_DIM = len(ALL_TAGS)
TAG2IDX = {tag: i for i, tag in enumerate(ALL_TAGS)}

# ========== DIFFAUG ========== (Simplified Color + Translation)
def diff_augment(x):
    shift = torch.randint(-2, 3, (x.size(0), 2, 1, 1), device=x.device)
    x = torch.roll(x, shifts=(shift[:, 0], shift[:, 1]), dims=(2, 3))
    return x

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
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8 + tag_dim, 1)
        )

    def forward(self, img, tags):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, tags], dim=1)
        return self.fc(x)

# ========== TRAINING ==========
def train():
    writer = SummaryWriter(LOG_DIR)
    dataset = FoodDataset(CSV_PATH, DATASET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(LATENT_DIM, TAG_DIM).to(DEVICE)
    discriminator = Discriminator(TAG_DIM).to(DEVICE)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def d_loss_fn(real, fake):
        return -(torch.mean(real) - torch.mean(fake))

    def g_loss_fn(fake):
        return -torch.mean(fake)

    for epoch in range(EPOCHS):
        for i, (imgs, tags) in enumerate(dataloader):
            real_imgs = diff_augment(imgs.to(DEVICE))
            tags = tags.to(DEVICE)
            batch_size = real_imgs.size(0)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
            gen_imgs = generator(z, tags)
            g_loss = g_loss_fn(discriminator(diff_augment(gen_imgs), tags))
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_pred = discriminator(real_imgs, tags)
            fake_pred = discriminator(diff_augment(gen_imgs.detach()), tags)
            d_loss = d_loss_fn(real_pred, fake_pred)
            d_loss.backward()
            optimizer_D.step()

        writer.add_scalar("Loss/Generator", g_loss.item(), epoch)
        writer.add_scalar("Loss/Discriminator", d_loss.item(), epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if (epoch + 1) % 100 == 0:
            vutils.save_image(gen_imgs.data[:1], f"{OUTPUT_DIR}/sample_epoch_{epoch+1:03}.png", normalize=True)
            torch.save(generator.state_dict(), os.path.join(WEIGHTS_DIR, f"generator_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    train()
