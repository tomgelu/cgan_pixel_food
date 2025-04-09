import os
from PIL import Image, ImageChops, ImageOps
from tqdm import tqdm

INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "preprocessed_dataset"
TARGET_SIZE = 128
USE_TRANSPARENT = True  # Set to False for white background

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def trim(img):
    # Remove white or transparent borders
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 0))
    else:
        bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        return img.crop(bbox)
    return img  # fallback

def process_image(path, output_path):
    img = Image.open(path).convert("RGBA")
    cropped = trim(img)
    

    resized = ImageOps.contain(cropped, (TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    print(resized.size)

    # Create output canvas
    if USE_TRANSPARENT:
        canvas = Image.new("RGBA", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255, 0))
    else:
        canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), (255, 255, 255))

    # Paste cropped image centered
    x = (TARGET_SIZE - resized.width) // 2
    y = (TARGET_SIZE - resized.height) // 2
    canvas.paste(resized, (x, y), resized)

    # Save
    if USE_TRANSPARENT:
        canvas.save(output_path)
    else:
        canvas.convert("RGB").save(output_path)

if __name__ == "__main__":
    for filename in tqdm(os.listdir(INPUT_FOLDER)):
        if not filename.lower().endswith(".png"):
            continue
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        process_image(input_path, output_path)


