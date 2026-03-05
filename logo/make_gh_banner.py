from pathlib import Path
from PIL import Image

script_dir = Path(__file__).resolve().parent
logo = Image.open(script_dir / "lfkit_logo.png").convert("RGBA")

W, H = 1280, 640
canvas = Image.new("RGBA", (W, H), (255, 255, 255, 0))

# keep logo comfortably inside safe area
max_logo_height = 260
max_logo_width = 500

logo.thumbnail((max_logo_width, max_logo_height), Image.LANCZOS)

x = (W - logo.width) // 2
y = (H - logo.height) // 2

canvas.paste(logo, (x, y), logo)

out = script_dir / "lfkit_github_banner.png"
canvas.save(out)

print("Saved:", out)