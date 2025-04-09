import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import time

# ========== CONFIG ==========
IMAGE_SIZE = 128
LATENT_DIM = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/checkpoint_epoch_2500.pth"
OUTPUT_DIR = "inference_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== TAGS ==========
ALL_TAGS = [
    "fish", "meat", "mushroom", "root_vegetables", "frog_legs",
    "grilled", "stewed", "raw", "baked",
    "curry", "brown_broth", "tomato_sauce", "green_emulsion",
    "chili_flakes", "flowers", "herbs"
]
TAG_DIM = len(ALL_TAGS)
TAG2IDX = {tag: i for i, tag in enumerate(ALL_TAGS)}

# ========== MODEL ==========
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

def create_tag_vector(tags):
    """Create a tag vector from a list of tags."""
    vec = np.zeros(TAG_DIM, dtype=np.float32)
    for tag in tags:
        if tag in TAG2IDX:
            vec[TAG2IDX[tag]] = 1.0
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

def generate_image(generator, tags, num_images=1, seed=None):
    """Generate images with the given tags."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Create tag vector
    tag_vec = create_tag_vector(tags)
    
    # Generate images
    images = []
    for _ in range(num_images):
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        with torch.no_grad():
            gen_img = generator(z, tag_vec)
        images.append(gen_img)
    
    return torch.cat(images, dim=0)

def main():
    # Load model
    generator = Generator(LATENT_DIM, TAG_DIM).to(DEVICE)
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    generator.eval()
    
    # Example usage
    tags = ["grilled", "meat", "mushroom", "herbs"]
    print(f"Generating image with tags: {tags}")
    
    start_time = time.time()
    # Generate image
    images = generate_image(generator, tags, num_images=1, seed=42)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # Save image
    filename = f"{'_'.join(tags)}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    vutils.save_image(images.data, output_path, normalize=True)
    print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    main() 