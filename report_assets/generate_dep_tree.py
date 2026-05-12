import spacy
from spacy import displacy
from pathlib import Path
import os

# Create directory if not exists
os.makedirs('report_assets', exist_ok=True)

nlp = spacy.load("en_core_web_sm")
doc = nlp("Party B shall pay the full rental amount before the 5th of each month.")

# Customize options to look like a legal dependency tree
options = {"compact": True, "bg": "#ffffff", "color": "#000000", "font": "Source Sans Pro"}
svg = displacy.render(doc, style="dep", options=options)

output_path = "report_assets/dependency_tree.svg"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(svg)

print(f"Generated dependency tree SVG at {output_path}")

# If we have cairosvg, convert to png (optional but better for LaTeX if not using svg package)
try:
    import cairosvg
    cairosvg.svg2png(url=output_path, write_to="report_assets/dependency_tree.png")
    print("Converted to dependency_tree.png")
except ImportError:
    print("cairosvg not installed, using SVG only.")
