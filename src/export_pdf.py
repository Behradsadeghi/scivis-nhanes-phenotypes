"""
Optional - export the presentation to a static PDF (one page per slide, animations in their final state).
Needs:  pip install playwright pillow && playwright install chromium
"""
from pathlib import Path
from playwright.sync_api import sync_playwright
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
deck = (ROOT / "presentation" / "index.html").resolve()
out = ROOT / "presentation" / "slides.pdf"

with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page(viewport={"width": 1600, "height": 900}, device_scale_factor=2)
    page.goto(deck.as_uri() + "?still")
    page.wait_for_timeout(1500)
    n = page.evaluate("document.querySelectorAll('.slide').length")
    shots = []
    for i in range(n):
        page.wait_for_timeout(700)
        path = ROOT / "presentation" / f"_s{i:02d}.png"
        page.screenshot(path=str(path))
        shots.append(path)
        page.keyboard.press("ArrowRight")
    b.close()

imgs = [Image.open(p).convert("RGB") for p in shots]
imgs[0].save(out, save_all=True, append_images=imgs[1:], resolution=200)
for p in shots:
    p.unlink()
print("wrote", out)
