import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
import argparse
from cgan_train import Generator, TAG_DIM, LATENT_DIM, ALL_TAGS, TAG2IDX

def create_tag_vector(tags_str):
    """Convert a string of tags into a binary vector."""
    vec = torch.zeros(TAG_DIM, dtype=torch.float32)
    tags = tags_str.split(',')
    for tag in tags:
        tag = tag.strip()
        if tag in TAG2IDX:
            vec[TAG2IDX[tag]] = 1.0
    return vec

def generate_image(generator, tags, output_path, device='cpu'):
    """Generate an image from tags using the generator."""
    # Create random noise
    z = torch.randn(1, LATENT_DIM, device=device)
    
    # Convert tags to tensor and add batch dimension
    tag_tensor = tags.unsqueeze(0).to(device)
    
    # Generate image
    with torch.no_grad():
        generated = generator(z, tag_tensor)
    
    # Save the generated image
    vutils.save_image(generated.data, output_path, normalize=True)
    print(f"Generated image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate food images from tags')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained generator model or checkpoint')
    parser.add_argument('--tags', type=str, required=True, help='Comma-separated list of tags')
    parser.add_argument('--output', type=str, default='generated.png', help='Output image path')
    args = parser.parse_args()

    # Set device to CPU
    device = torch.device('cpu')
    
    # Load the generator
    generator = Generator(LATENT_DIM, TAG_DIM).to(device)
    
    # Load the checkpoint and extract generator state dict
    checkpoint = torch.load(args.model, map_location=device)
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint['epoch']})")
    else:
        generator.load_state_dict(checkpoint)
        print("Loaded model from final weights file")
    
    generator.eval()
    
    # Print available tags
    print("\nAvailable tags:")
    for tag in ALL_TAGS:
        print(f"- {tag}")
    
    # Create tag vector
    tags = create_tag_vector(args.tags)
    
    # Generate image
    generate_image(generator, tags, args.output, device)

if __name__ == "__main__":
    main() 