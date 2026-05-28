# CardGrader AI 🃏

An open-source machine learning project that predicts trading card grades (PSA, Beckett/BGS, SGC, CGC)
from a photo, then estimates the card's value after grading.

Runs **entirely in your browser** via GitHub Pages — no server, no subscription, no cost.

---

## How It Works

```
Card Photo → CNN Model → Sub-scores (centering, corners, edges, surface)
                              ↓
                    Grade Mapping (PSA / BGS / SGC / CGC)
                              ↓
                    Value Estimate (grading fee + market multiplier)
```

The model outputs four sub-scores (0–10), which are combined into a composite grade
and mapped to each grading company's scale.

---

## Project Structure

```
card-grader/
├── notebooks/
│   └── card_grader_training.ipynb   ← Full training pipeline (start here)
├── data/
│   ├── scraper/
│   │   └── ebay_scraper.py          ← Auto-collect labeled training images
│   ├── train/                       ← Training images + JSON labels (you create this)
│   ├── val/
│   └── test/
├── model/
│   ├── grade_mapping.py             ← PSA/BGS/SGC/CGC grade conversion logic
│   ├── export_onnx.py               ← Convert trained model → ONNX for browser
│   └── saved/                       ← Trained .keras model files go here
├── docs/                            ← GitHub Pages site (free hosting)
│   ├── index.html                   ← Browser-based grader UI
│   └── card_grader.onnx             ← Trained model (you add this after training)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/card-grader.git
cd card-grader
pip install -r requirements.txt
```

### 2. Collect Training Data

**Option A — Automated (recommended):**
```bash
# Get a free eBay Developer API key at https://developer.ebay.com
export EBAY_APP_ID='your_key_here'
python data/scraper/ebay_scraper.py --max 50
```

**Option B — Manual:**
Photograph your own cards (or cards you know the PSA grade of).
Create matching JSON label files:
```json
{
  "centering": 9.0,
  "corners":   9.5,
  "edges":     9.0,
  "surface":   8.5,
  "psa_grade": 9,
  "card_name": "Charizard Base Set Holo"
}
```
Place `card_001.jpg` + `card_001.json` pairs in `data/train/`.

### 3. Train the Model

Open `notebooks/card_grader_training.ipynb` in Google Colab or Jupyter.

> **Tip:** Use Google Colab with a free T4 GPU — training takes ~30–60s per epoch instead of minutes.

Run all cells. The best model saves automatically to `model/saved/best_mobilenet.keras`.

### 4. Export to ONNX

```bash
python model/export_onnx.py \
  --model model/saved/best_mobilenet.keras \
  --output docs/card_grader.onnx
```

### 5. Deploy to GitHub Pages

```bash
git add docs/
git commit -m "Add trained ONNX model"
git push
```

Then in your GitHub repo: **Settings → Pages → Source → Deploy from branch → main → /docs**

Your free grader is live at: `https://your-username.github.io/card-grader`

---

## Data Label Format

Each image needs a matching JSON file with the same base filename:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `centering` | float 1–10 | ✅ | How centered the image is within borders |
| `corners` | float 1–10 | ✅ | Corner sharpness (no fraying/wear) |
| `edges` | float 1–10 | ✅ | Edge cleanliness (no chips/roughness) |
| `surface` | float 1–10 | ✅ | Surface condition (no scratches/print lines) |
| `psa_grade` | int | optional | Actual PSA grade if known |
| `card_name` | string | optional | Card name for reference |

---

## Grading Scales Supported

| Company | Scale | Notes |
|---------|-------|-------|
| PSA | 1–10 (integers) | Most popular; PSA 10 = Gem Mint |
| Beckett (BGS) | 1–10 (half grades) | Sub-grades published; Black Label 10 = all 9.5+ |
| SGC | 1–10 (half grades) | Known for consistent centering standards |
| CGC | 1–10 (half grades) | Expanding from comics into trading cards |

---

## Model Architecture

Three models are trained and compared:

| Model | Backbone | Parameters | Best For |
|-------|----------|------------|----------|
| Custom CNN | From scratch | ~2M | Baseline, fast training |
| MobileNetV2 | ImageNet transfer | ~4M | Best size/accuracy tradeoff, recommended for deployment |
| EfficientNetB3 | ImageNet transfer | ~12M | Highest accuracy |

All models output 4 regression values [0,1] — one per grading dimension.
Loss function: MSE. Metric: Grade MAE (mean absolute error in grade points).

---

## Key Differences from Plant Disease Classifier

If you're coming from a classification project, here's what changed:

| Plant Disease Project | Card Grader |
|----------------------|-------------|
| 18 output classes (softmax) | 4 output values (sigmoid regression) |
| Categorical crossentropy loss | MSE loss |
| Folder-per-class structure | Image + JSON label pairs |
| `horizontal_flip=True` | `horizontal_flip=False` (card orientation matters!) |
| 128×128 images | 224×224 (card detail needs higher resolution) |
| Kaggle dataset | eBay scraper or manual collection |

---

## Limitations & Notes

- **Sub-score estimation:** The eBay scraper collects overall grades (PSA 10, BGS 9.5) but not the individual sub-scores (centering/corners/edges/surface). Sub-scores in scraped labels are *estimated* from the overall grade. For best accuracy, manually annotate sub-scores on a subset of your data.
- **Slab photos vs raw cards:** Graded card photos show cards inside slabs (plastic cases), which adds glare and reflection. Training on both slab photos and raw card photos improves generalization.
- **Grade accuracy:** The model estimates — actual grades depend on human graders at PSA/BGS/SGC/CGC and can vary. Use this as a guide, not a guarantee.
- **Price estimates:** Market multipliers are approximate and vary significantly by card, set, print run, and market conditions.

---

## Contributing

Pull requests welcome! Key areas for improvement:
- More training data (especially sub-score annotations)
- Better centering detection (geometric/algorithmic approach)
- Support for more grading companies
- Mobile camera integration for the web app

---

## License

MIT License — free to use, modify, and distribute.
