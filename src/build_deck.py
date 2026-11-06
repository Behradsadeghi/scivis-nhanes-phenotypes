"""
Step 4 - Assemble the presentation: inline every SVG figure and the point data
into presentation/template.html -> presentation/index.html (one self-contained file).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Behradsadeghi/scivis-nhanes-phenotypes"


def clean_svg(name):
    s = (ROOT / "figures" / f"{name}.svg").read_text()
    s = s[s.index("<svg"):]
    s = re.sub(r"<metadata>.*?</metadata>", "", s, flags=re.S)
    pre = name.split("_")[0] + "_"
    s = re.sub(r'id="([^"]+)"', lambda m: f'id="{pre}{m.group(1)}"', s)
    s = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{pre}{m.group(1)})', s)
    s = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{pre}{m.group(1)}"', s)
    s = s.replace("<svg ", '<svg role="img" preserveAspectRatio="xMidYMid meet" ', 1)
    return s


html = (ROOT / "presentation" / "template.html").read_text()
html = re.sub(r"\{\{SVG:([\w]+)\}\}", lambda m: clean_svg(m.group(1)), html)
html = html.replace("{{DATA}}", (ROOT / "results" / "deck_points.json").read_text())
html = html.replace("{{REPO_SHORT}}", REPO.replace("https://", "")).replace("{{REPO}}", REPO)
out = ROOT / "presentation" / "index.html"
out.write_text(html)
print(out, round(out.stat().st_size / 1e6, 2), "MB")
