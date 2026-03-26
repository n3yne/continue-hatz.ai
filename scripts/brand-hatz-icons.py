"""
Generate Hatz-branded icons by overlaying "Hatz" text on the Continue icons.

- Main icon (icon.png): Red "Hatz" text with dark background near the bottom.
- Sidebar icon (sidebar-icon.png): Monochrome alpha-only "Hatz" text in the
  center gap of the Continue logo, matching the icon's visual style.
"""
from PIL import Image, ImageDraw, ImageFont
import os

# Resolve project root (script is in <root>/scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MEDIA_DIR = os.path.join(PROJECT_ROOT, "extensions", "vscode", "media")


def _find_bold_font(font_size: int) -> ImageFont.FreeTypeFont | None:
    """Try to find a bold system font, return None if unavailable."""
    bold_fonts = [
        "arialbd.ttf",      # Windows
        "Arial Bold.ttf",   # Windows alt
        "Impact.ttf",       # Windows - good for overlays
        "calibrib.ttf",     # Windows Calibri Bold
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]
    for font_name in bold_fonts:
        try:
            return ImageFont.truetype(font_name, font_size)
        except (OSError, IOError):
            continue
    return None


def add_hatz_overlay_color(input_path: str, output_path: str, font_size_ratio: float = 0.25):
    """
    Overlay bold red "Hatz" text on the bottom portion of an RGBA icon.
    Used for the main extension icon.
    """
    img = Image.open(input_path).convert("RGBA")
    width, height = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = int(width * font_size_ratio)
    font = _find_bold_font(font_size)
    if font is None:
        font = ImageFont.load_default()
        print("Warning: Using default font (may be small)")

    text = "Hatz"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position: centered horizontally, near the bottom
    x = (width - text_width) // 2
    y = height - text_height - int(height * 0.08)

    # Semi-transparent dark background for readability
    padding = int(font_size * 0.15)
    bg_rect = [
        x - padding,
        y - padding,
        x + text_width + padding,
        y + text_height + padding,
    ]
    draw.rounded_rectangle(bg_rect, radius=padding, fill=(0, 0, 0, 160))

    # Red text
    draw.text((x, y), text, fill=(255, 40, 40, 255), font=font)

    result = Image.alpha_composite(img, overlay)
    result.save(output_path, "PNG")
    print(f"Saved: {output_path} ({width}x{height})")


def add_hatz_overlay_monochrome(input_path: str, output_path: str, font_size_ratio: float = 0.45):
    """
    Overlay a bold "H" in the center gap of a monochrome alpha-channel icon.
    VS Code sidebar icons are rendered using only the alpha channel, so we
    draw black pixels with full opacity to match the existing icon style.
    """
    img = Image.open(input_path).convert("RGBA")
    width, height = img.size

    draw = ImageDraw.Draw(img)

    font_size = int(width * font_size_ratio)
    font = _find_bold_font(font_size)
    if font is None:
        font = ImageFont.load_default()
        print("Warning: Using default font (may be small)")

    text = "H"

    # Render the glyph onto a temporary canvas to find the exact pixel bounds,
    # since textbbox is not accurate for centering.
    temp = Image.new("RGBA", img.size, (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.text((0, 0), text, fill=(0, 0, 0, 255), font=font)

    temp_pixels = temp.load()
    min_x, max_x, min_y, max_y = width, 0, height, 0
    for py in range(height):
        for px in range(width):
            if temp_pixels[px, py][3] > 0:
                min_x = min(min_x, px)
                max_x = max(max_x, px)
                min_y = min(min_y, py)
                max_y = max(max_y, py)

    glyph_w = max_x - min_x
    glyph_h = max_y - min_y

    # Calculate the shift needed to move the glyph center to the image center
    # Small left nudge (-30px) to compensate for visual weight of the "H" glyph
    shift_x = (width // 2) - (min_x + glyph_w // 2) - 60
    shift_y = (height // 2) - (min_y + glyph_h // 2)

    # Draw on the actual image at the corrected position
    draw.text((shift_x, shift_y), text, fill=(0, 0, 0, 255), font=font)

    img.save(output_path, "PNG")
    print(f"Saved: {output_path} ({width}x{height})")


def main():
    sidebar_path = os.path.join(MEDIA_DIR, "sidebar-icon.png")

    if os.path.exists(sidebar_path):
        add_hatz_overlay_monochrome(sidebar_path, sidebar_path)
    else:
        print(f"Not found: {sidebar_path}")

    print("\nDone! Sidebar icon has been updated with Hatz branding.")


if __name__ == "__main__":
    main()
