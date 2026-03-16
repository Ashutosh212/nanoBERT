import re
import csv

from datasets import load_dataset
# Load WikiText-2 raw
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
print("Dataset loaded.")


def is_heading(line):
    return bool(re.match(r'^\s*=+\s*.+\s*=+\s*$', line))

def build_paragraphs(split_name):
    sections = []
    current = []
    for row in dataset[split_name]["text"]:
        line = row.strip()
        if is_heading(line):
            if current:
                sections.append(" ".join(current))
                current = []
        elif line:
            current.append(line)
    if current:
        sections.append(" ".join(current))
    return sections

all_paragraphs = []
for split in ["train", "validation", "test"]:
    all_paragraphs.extend(build_paragraphs(split))

output_path = "/home/jovyan/nanoBERT/dataset/wikitext2_paragraphs.csv"
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["paragraph"])
    for para in all_paragraphs:
        writer.writerow([para])

print(f"Saved {len(all_paragraphs):,} paragraphs to {output_path}")
print("\nSample paragraphs:")
for p in all_paragraphs[:3]:
    print("-", p[:120])
