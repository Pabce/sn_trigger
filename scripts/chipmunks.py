#!/usr/bin/env python3

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_emoji_image(emoji, font_path, image_size=(128, 128)):
    """Creates an image of the emoji."""
    # Create an image with transparent background
    image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Load the font
    try:
        font = ImageFont.truetype(font_path, int(image_size[0] * 0.8))
    except IOError:
        print("Font not found.")
        sys.exit(1)

    # Get the size of the emoji
    text_width, text_height = draw.textsize(emoji, font=font)

    # Calculate position
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    # Draw the emoji onto the image
    draw.text(position, emoji, font=font, fill="black")

    return image

def mirror_image(image):
    """Mirrors the image horizontally."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)

def image_to_ascii(image, cols=80, scale=0.43):
    """Converts the image to ASCII art."""
    # Convert image to grayscale
    image = image.convert("L")

    # Store dimensions
    width, height = image.size

    # Compute width of tile
    W = width / cols

    # Compute tile height based on aspect ratio
    H = W / scale
    rows = int(height / H)

    # Check if image size is too small
    if cols > width or rows > height:
        print("Image too small for specified cols!")
        sys.exit(1)

    # List of ASCII characters
    ascii_chars = "@%#*+=-:. "

    # Generate the ASCII image
    ascii_image = []
    for j in range(rows):
        y1 = int(j * H)
        y2 = int((j + 1) * H)

        if j == rows -1:
            y2 = height

        ascii_line = ""
        for i in range(cols):
            x1 = int(i * W)
            x2 = int((i +1) * W)

            if i == cols -1:
                x2 = width

            # Crop the image to extract the tile
            img = image.crop((x1, y1, x2, y2))

            # Get the average luminance
            avg = int(np.array(img).mean())

            # Map the average to an ASCII char
            ascii_char = ascii_chars[int((avg * (len(ascii_chars) -1)) / 255)]
            ascii_line += ascii_char

        ascii_image.append(ascii_line)

    return "\n".join(ascii_image)

def main():
    emoji_input = input("Enter an emoji: ")

    # Path to a font that supports emojis
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",  # Common on Linux
        "/System/Library/Fonts/Apple Color Emoji.ttf",          # macOS
        "C:\\Windows\\Fonts\\seguiemj.ttf",                     # Windows
    ]

    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break

    if not font_path:
        print("Emoji font not found on your system.")
        sys.exit(1)

    image = get_emoji_image(emoji_input, font_path)
    mirrored_image = mirror_image(image)
    ascii_art = image_to_ascii(mirrored_image)
    print(ascii_art)

if __name__ == "__main__":
    main()
