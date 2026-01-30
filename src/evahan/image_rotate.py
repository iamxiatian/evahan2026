import os

import typer
from PIL import Image

# Supported image file extensions (can be added/modified manually)
supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')


def rotate_image(input_path, output_path):
    """Rotate the image 90 degrees clockwise and save"""
    with Image.open(input_path) as img:
        # Rotate clockwise 90 degrees
        rotated = img.rotate(-90, expand=True)
        # Save (Keep original formatting)
        rotated.save(output_path, quality=95)  # quality applies only to JPEG files
        print(f"Rotated and saved: {os.path.basename(output_path)}")

def main(input_folder:str, output_folder:str) -> None:
    """
    将文件夹下的图片旋转90度
    Args:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
    """
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed = 0
    skipped = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                rotate_image(input_path, output_path)
                processed += 1
            except Exception as e:
                print(f"Processing failed: {input_path} → {e}")
                skipped += 1

    print("Processing complete!")
    print(f"Successfully rotated and saved: {processed} images")
    print(f"Processing failed: {skipped} images")
    print(f"Output folder: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    typer.run(main)
