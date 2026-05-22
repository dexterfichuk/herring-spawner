"""Terminal-based interactive labeling tool. Shows images and waits for keypress."""

import base64, json, shutil, sys, subprocess, tempfile, os
from pathlib import Path

thumb_dir = Path("data/review/thumbnails2")
png_files = sorted(thumb_dir.glob("*.png"))

# Load existing labels if available
labels = {}
labels_file = Path("/Users/dexterfichuk/Downloads/spawn_labels_2.json")
if labels_file.exists():
    try:
        for item in json.loads(labels_file.read_text()):
            labels[item["filename"]] = item["label"]
    except: pass

# Also check localStorage by offering to reload
print(f"\nTotal {len(png_files)} images to label")
print(f"Already labeled: {len(labels)}")
print()

# Process in batches
current_labels = {}
for path in png_files:
    fname = path.name
    if fname in labels:
        current_labels[fname] = labels[fname]
        continue
    
    print(f"\n{'='*60}")
    print(f"Image #{len(current_labels)+1}/{len(png_files)}")
    print(f"File: {fname}")
    
    # Open image using system default viewer
    subprocess.run(["open", str(path)], check=False)
    
    # Wait for user input
    while True:
        try:
            inp = input("Label? [s]pawn [n]ospawn [c]loudy [?]unknown [q]uit [b]ack: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            inp = "q"
        
        if inp == "q":
            # Save progress and exit
            break
        elif inp == "b" and current_labels:
            # Go back one
            removed = list(current_labels.keys())[-1]
            del current_labels[removed]
            print(f"Removed label for {removed}")
            continue
        elif inp in ("s", "n", "c", "?"):
            label_map = {"s": "spawn", "n": "nospawn", "c": "cloudy", "?": "unknown"}
            current_labels[fname] = label_map[inp]
            print(f"  -> {label_map[inp]}")
            break
        else:
            print("Invalid. Use: s, n, c, ?, q, or b")
    
    if inp == "q":
        break

# Save
output = []
for fname, label in current_labels.items():
    parts = fname.replace(".png","").split("_")
    event = parts[0]
    date_str = parts[1] if len(parts) > 1 else ""
    cloud = parts[2].replace("cld","") if len(parts) > 2 else ""
    output.append({"filename": fname, "label": label})

Path("spawn_labels_2.json").write_text(json.dumps(output, indent=2))
print(f"\nSaved {len(output)} labels to spawn_labels_2.json")

# Stats
from collections import Counter
stats = Counter(item["label"] for item in output)
print(f"Stats: {dict(stats)}")
