"""Generate embeddings from thumbnails using DINOv2 and run classification."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# DINOv2 - best self-supervised vision model for semantic similarity
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f'Using device: {device}')

transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# User-confirmed positives
POSITIVE_FILES = {
    'dfo-verified-anderson-point_2024-03-18_20240318.png',
    'dfo-verified-breakwater-island_2024-03-18_20240318.png',
    'dfo-verified-fan-island_2024-03-22_20240322.png',
    'dfo-verified-qualicum-beach_2024-03-15_20240315.png',
    'dfo-verified-ucluelet_2024-03-18_20240318.png',
    'news-salmon-beach-2025_2025-02-11_20250211.png',
}

review_dir = Path('data/review')
png_files = sorted(review_dir.glob('*.png'))
print(f'Found {len(png_files)} images')

# Generate embeddings
embeddings = {}
labels = {}
for path in png_files:
    fname = path.name
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor)
    emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
    embeddings[fname] = emb
    labels[fname] = 'positive' if fname in POSITIVE_FILES else 'unlabeled'
    print(f'  {fname}: {labels[fname]}')

# Compute mean positive embedding
pos_embs = [embeddings[f] for f in POSITIVE_FILES if f in embeddings]
mean_pos = np.mean(pos_embs, axis=0)
mean_pos = mean_pos / np.linalg.norm(mean_pos)

# Score all images by cosine similarity to mean positive
scored = []
for fname, emb in embeddings.items():
    sim = float(np.dot(mean_pos, emb))
    scored.append((fname, sim, labels[fname]))

scored.sort(key=lambda x: x[1], reverse=True)

print(f'\n=== Embedding similarity ranking (closest to known spawns) ===')
print(f'{"Rank":<5} {"Score":<8} {"Label":<10} {"Filename"}')
print('-' * 80)
for rank, (fname, sim, label) in enumerate(scored, 1):
    marker = ' ✓' if label == 'positive' else ''
    print(f'{rank:<5} {sim:<8.4f} {label:<10} {fname}{marker}')

# Statistics
pos_scores = [s for _, s, l in scored if l == 'positive']
unlab_scores = [s for _, s, l in scored if l == 'unlabeled']
print(f'\nPositive mean similarity: {np.mean(pos_scores):.4f} ± {np.std(pos_scores):.4f}')
print(f'Unlabeled mean similarity: {np.mean(unlab_scores):.4f} ± {np.std(unlab_scores):.4f}')
print(f'Separation: {abs(np.mean(pos_scores) - np.mean(unlab_scores)):.4f}')

# Generate HTML report
rows_html = ''
for rank, (fname, sim, label) in enumerate(scored, 1):
    is_pos = label == 'positive'
    img_path = review_dir / fname
    import base64
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    badge = '✅ POSITIVE' if is_pos else '⬜ unlabeled'
    color = '#d4edda' if is_pos else '#fff'
    rows_html += f"""
    <tr style="background:{color}">
        <td>{rank}</td>
        <td>{badge}</td>
        <td>{sim:.4f}</td>
        <td>{fname}</td>
        <td><img src="data:image/png;base64,{b64}" width="200"></td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html><head><title>Embedding Similarity Ranking</title>
<style>
body {{ font-family: sans-serif; margin: 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
th {{ background: #333; color: white; }}
img {{ max-width: 200px; }}
.summary {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
</style></head><body>
<h1>DINOv2 Embedding Similarity Ranking</h1>
<div class="summary">
    <p><strong>Method:</strong> DINOv2 (ViT-S/14) pretrained on ImageNet</p>
    <p><strong>Reference:</strong> Mean embedding of {len(pos_embs)} user-confirmed spawn images</p>
    <p><strong>Positive mean similarity:</strong> {np.mean(pos_scores):.4f} ± {np.std(pos_scores):.4f}</p>
    <p><strong>Unlabeled mean similarity:</strong> {np.mean(unlab_scores):.4f} ± {np.std(unlab_scores):.4f}</p>
    <p><strong>Separation:</strong> {abs(np.mean(pos_scores) - np.mean(unlab_scores)):.4f}</p>
</div>
<table>
<tr><th>Rank</th><th>Label</th><th>Similarity</th><th>Filename</th><th>Thumbnail</th></tr>
{rows_html}
</table></body></html>"""

Path('data/review/embedding_ranking.html').write_text(html)
print(f'\nReport: file://{Path("data/review/embedding_ranking.html").resolve()}')
