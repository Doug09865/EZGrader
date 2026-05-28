"""
ebay_scraper.py
---------------
Collects labeled training data for the card grader by scraping
eBay sold listings of graded cards.

Each sold listing of a PSA/BGS/SGC graded card in a slab gives us:
  - An image of the card (front of slab)
  - A known grade (from the listing title)

This creates a labeled dataset without needing to submit cards yourself.

Setup:
    1. Get a free eBay Developer API key at https://developer.ebay.com
    2. Set EBAY_APP_ID in your environment or replace below
    3. Run: python ebay_scraper.py

Output structure:
    data/
      train/  (80% of collected images)
        card_001.jpg
        card_001.json   ← {"centering": null, "corners": null, ..., "psa_grade": 10}
        ...
      val/    (10%)
      test/   (10%)

NOTE: Slab photos don't show individual sub-scores (centering/corners/edges/surface)
      because graders don't publish those. Two options:
      A) Use the overall PSA grade as a proxy — train model to predict composite only
      B) Manually annotate a subset with sub-scores for fine-grained training
      
      The scraper collects overall grades. Sub-score annotation can be done manually
      using the included labeling helper (label_tool.py).
"""

import os
import re
import sys
import json
import time
import random
import hashlib
import requests
import argparse
from pathlib import Path
from urllib.parse import quote_plus
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration — update these values
# ---------------------------------------------------------------------------
EBAY_APP_ID = os.environ.get('EBAY_APP_ID', 'YOUR_EBAY_APP_ID_HERE')

# Output directories
DATA_DIR   = Path(__file__).parent.parent
TRAIN_DIR  = DATA_DIR / 'train'
VAL_DIR    = DATA_DIR / 'val'
TEST_DIR   = DATA_DIR / 'test'

# Search configuration
DEFAULT_SEARCHES = [
    # Format: (search_query, grading_company)
    # Pokemon
    ('PSA 10 pokemon charizard graded',      'PSA'),
    ('PSA 9 pokemon charizard graded',       'PSA'),
    ('PSA 8 pokemon charizard graded',       'PSA'),
    ('PSA 10 pokemon pikachu graded',        'PSA'),
    ('PSA 9 pokemon pikachu graded',         'PSA'),
    ('BGS 9.5 pokemon graded beckett',       'BGS'),
    ('BGS 10 black label pokemon graded',    'BGS'),
    ('SGC 10 pokemon graded',                'SGC'),
    # Magic: The Gathering
    ('PSA 10 magic the gathering graded',    'PSA'),
    ('PSA 9 magic the gathering graded',     'PSA'),
    # Yu-Gi-Oh
    ('PSA 10 yugioh graded',                 'PSA'),
    ('PSA 9 yugioh graded',                  'PSA'),
    # Sports (high volume of graded cards)
    ('PSA 10 baseball card graded',          'PSA'),
    ('PSA 9 baseball card graded',           'PSA'),
]

# How many sold listings to fetch per search query
RESULTS_PER_QUERY = 50

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10 (implied)

# Delay between requests to be respectful
REQUEST_DELAY_SECONDS = 1.0

# ---------------------------------------------------------------------------
# eBay Finding API helper
# ---------------------------------------------------------------------------
EBAY_FINDING_API_URL = 'https://svcs.ebay.com/services/search/FindingService/v1'

def search_ebay_sold(
    query: str,
    app_id: str,
    max_results: int = 50,
    page: int = 1
) -> list[dict]:
    """
    Search eBay completed/sold listings using the Finding API.
    Returns a list of item dicts with title, image URL, and price.
    """
    params = {
        'OPERATION-NAME':          'findCompletedItems',
        'SERVICE-VERSION':         '1.0.0',
        'SECURITY-APPNAME':        app_id,
        'RESPONSE-DATA-FORMAT':    'JSON',
        'REST-PAYLOAD':            '',
        'keywords':                query,
        'itemFilter(0).name':      'SoldItemsOnly',
        'itemFilter(0).value':     'true',
        'itemFilter(1).name':      'ListingType',
        'itemFilter(1).value':     'AuctionWithBIN',
        'outputSelector(0)':       'PictureURLLarge',
        'paginationInput.pageNumber': str(page),
        'paginationInput.entriesPerPage': str(min(max_results, 100)),
        'sortOrder':               'EndTimeSoonest',
    }

    try:
        response = requests.get(EBAY_FINDING_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = (data
                   .get('findCompletedItemsResponse', [{}])[0]
                   .get('searchResult', [{}])[0]
                   .get('item', []))
        return results
    except Exception as e:
        print(f"  eBay API error for '{query}': {e}")
        return []


# ---------------------------------------------------------------------------
# Grade extraction from listing title
# ---------------------------------------------------------------------------
GRADE_PATTERNS = {
    'PSA': [
        r'\bPSA\s*10\b',
        r'\bPSA\s*9\.5\b',
        r'\bPSA\s*9\b(?!\.)',
        r'\bPSA\s*8\.5\b',
        r'\bPSA\s*8\b(?!\.)',
        r'\bPSA\s*7\.5\b',
        r'\bPSA\s*7\b(?!\.)',
        r'\bPSA\s*6\b',
        r'\bPSA\s*5\b',
        r'\bPSA\s*4\b',
        r'\bPSA\s*3\b',
        r'\bPSA\s*2\b',
        r'\bPSA\s*1\b',
    ],
    'BGS': [
        r'\bBGS\s*10\s*Black\s*Label\b',
        r'\bBGS\s*10\b',
        r'\bBGS\s*9\.5\b',
        r'\bBGS\s*9\b(?!\.)',
        r'\bBGS\s*8\.5\b',
        r'\bBGS\s*8\b(?!\.)',
        r'\bBGS\s*7\.5\b',
        r'\bBGS\s*7\b(?!\.)',
    ],
    'SGC': [
        r'\bSGC\s*10\b',
        r'\bSGC\s*9\.5\b',
        r'\bSGC\s*9\b(?!\.)',
        r'\bSGC\s*8\.5\b',
        r'\bSGC\s*8\b(?!\.)',
    ],
    'CGC': [
        r'\bCGC\s*10\b',
        r'\bCGC\s*9\.5\b',
        r'\bCGC\s*9\b(?!\.)',
    ],
}

def extract_grade_from_title(title: str) -> Optional[tuple[str, float]]:
    """
    Extract grading company and numeric grade from a listing title.
    
    Returns:
        (company, grade_float) or None if no grade found
    
    Examples:
        'PSA 10 Charizard Base Set Holo'  -> ('PSA', 10.0)
        'BGS 9.5 Gem Mint Pikachu'        -> ('BGS', 9.5)
    """
    title_upper = title.upper()
    
    for company, patterns in GRADE_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, title_upper, re.IGNORECASE)
            if match:
                matched_text = match.group(0)
                # Extract numeric grade
                num_match = re.search(r'(\d+(?:\.\d+)?)', matched_text)
                if num_match:
                    grade = float(num_match.group(1))
                    # Handle Black Label
                    if 'BLACK' in matched_text:
                        return (company, 10.0)  # Black Label = BGS 10
                    return (company, grade)
    return None


# ---------------------------------------------------------------------------
# Image downloader
# ---------------------------------------------------------------------------
def download_image(url: str, save_path: str) -> bool:
    """Download image from URL to save_path. Returns True on success."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; CardGraderBot/1.0)'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Verify it's actually an image
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            return False
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def get_image_url(item: dict) -> Optional[str]:
    """Extract the best available image URL from an eBay item."""
    # Try large picture first
    large = item.get('pictureURLLarge', [None])[0]
    if large:
        return large
    # Fall back to standard
    standard = item.get('galleryURL', [None])[0]
    return standard


# ---------------------------------------------------------------------------
# Label creator
# ---------------------------------------------------------------------------
def create_label_json(
    company: str,
    grade: float,
    card_name: str = '',
    image_path: str = '',
) -> dict:
    """
    Create a label JSON for a card with known overall grade but unknown sub-scores.
    
    Sub-scores are estimated from the overall grade using typical distributions.
    These are approximations — for production use, annotate sub-scores manually.
    """
    # Estimate sub-scores from overall grade
    # Cards graded 10 tend to have high sub-scores with some variation
    # These are heuristics — replace with real annotations when possible
    base = grade
    sub_scores = {
        'centering': round(min(10, max(1, base + random.uniform(-0.5, 0.3))), 1),
        'corners':   round(min(10, max(1, base + random.uniform(-0.3, 0.3))), 1),
        'edges':     round(min(10, max(1, base + random.uniform(-0.3, 0.3))), 1),
        'surface':   round(min(10, max(1, base + random.uniform(-0.5, 0.3))), 1),
    }

    return {
        **sub_scores,
        'overall_grade':    grade,
        'grading_company':  company,
        'card_name':        card_name,
        'source':           'ebay_sold_listing',
        'sub_scores_estimated': True,  # Flag: replace with manual annotations
        'image_path':       image_path,
    }


# ---------------------------------------------------------------------------
# Dataset split helper
# ---------------------------------------------------------------------------
def get_split_dir(index: int, total: int) -> Path:
    """Assign sample to train/val/test split based on index."""
    ratio = index / total
    if ratio < TRAIN_RATIO:
        return TRAIN_DIR
    elif ratio < TRAIN_RATIO + VAL_RATIO:
        return VAL_DIR
    else:
        return TEST_DIR


# ---------------------------------------------------------------------------
# Main scraper
# ---------------------------------------------------------------------------
def run_scraper(
    searches: list = None,
    max_per_query: int = RESULTS_PER_QUERY,
    dry_run: bool = False
):
    """
    Run the eBay scraper to collect labeled training data.
    
    Args:
        searches:      list of (query, company) tuples (uses DEFAULT_SEARCHES if None)
        max_per_query: max images to download per search query
        dry_run:       if True, show what would be downloaded without downloading
    """
    if EBAY_APP_ID == 'YOUR_EBAY_APP_ID_HERE':
        print("⚠️  eBay App ID not set!")
        print("   Get a free key at https://developer.ebay.com")
        print("   Then set: export EBAY_APP_ID='your_key_here'")
        print("")
        print("   Running in DEMO MODE — showing sample output only.")
        _run_demo_mode()
        return

    if searches is None:
        searches = DEFAULT_SEARCHES

    # Create output directories
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    all_samples = []

    print(f"Starting eBay scraper — {len(searches)} search queries")
    print(f"Max {max_per_query} results per query\n")

    for query, company in searches:
        print(f"Searching: '{query}'")
        items = search_ebay_sold(query, EBAY_APP_ID, max_results=max_per_query)
        print(f"  Found {len(items)} results")

        downloaded = 0
        for item in items:
            title     = item.get('title', [''])[0]
            image_url = get_image_url(item)
            price     = (item.get('sellingStatus', [{}])[0]
                             .get('convertedCurrentPrice', [{}])[0]
                             .get('__value__', '0'))

            # Extract grade from title
            grade_info = extract_grade_from_title(title)
            if not grade_info:
                continue  # Skip if we can't determine the grade

            grading_co, grade_value = grade_info

            if not image_url:
                continue

            # Create unique filename from URL hash
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
            img_name = f"card_{grading_co.lower()}_{int(grade_value*10):03d}_{url_hash}"

            all_samples.append({
                'img_name':   img_name,
                'image_url':  image_url,
                'company':    grading_co,
                'grade':      grade_value,
                'card_name':  title[:80],
                'price_usd':  float(price),
            })

        time.sleep(REQUEST_DELAY_SECONDS)

    # Shuffle before splitting so grade distributions are even across splits
    random.shuffle(all_samples)
    total = len(all_samples)
    print(f"\nTotal samples collected: {total}")

    if dry_run:
        print("\n[DRY RUN] Would download:")
        for s in all_samples[:5]:
            print(f"  {s['img_name']}.jpg  {s['company']} {s['grade']}  {s['card_name'][:50]}")
        print("  ...")
        return

    # Download images and save labels
    success = 0
    skipped = 0
    for i, sample in enumerate(all_samples):
        split_dir = get_split_dir(i, total)
        img_path  = split_dir / f"{sample['img_name']}.jpg"
        lbl_path  = split_dir / f"{sample['img_name']}.json"

        # Skip if already downloaded
        if img_path.exists() and lbl_path.exists():
            skipped += 1
            continue

        print(f"  [{i+1}/{total}] {sample['company']} {sample['grade']} — {sample['card_name'][:50]}")

        ok = download_image(sample['image_url'], str(img_path))
        if not ok:
            continue

        label = create_label_json(
            company    = sample['company'],
            grade      = sample['grade'],
            card_name  = sample['card_name'],
            image_path = str(img_path),
        )
        with open(lbl_path, 'w') as f:
            json.dump(label, f, indent=2)

        success += 1
        time.sleep(REQUEST_DELAY_SECONDS * 0.5)  # Small delay between downloads

    print(f"\n✅ Done!")
    print(f"   Downloaded : {success}")
    print(f"   Skipped    : {skipped} (already existed)")
    print(f"   Failed     : {total - success - skipped}")
    print(f"\n   Train: {TRAIN_DIR}")
    print(f"   Val:   {VAL_DIR}")
    print(f"   Test:  {TEST_DIR}")
    print(f"\nNOTE: Sub-scores in labels are estimated from overall grade.")
    print(f"For better model accuracy, manually annotate sub-scores")
    print(f"using label_tool.py.")


def _run_demo_mode():
    """Show what the scraper would produce without needing an API key."""
    print("=== DEMO MODE OUTPUT ===\n")
    example_titles = [
        "PSA 10 GEM MINT 2016 Pokemon Charizard EX #12 Holo",
        "BGS 9.5 GEM MINT 1999 Pokemon Base Pikachu #58",
        "PSA 9 MINT Magic: The Gathering Black Lotus Alpha",
        "SGC 10 Pristine 2023 Pokemon Scarlet Violet Charizard",
        "PSA 8 NM-MT 1st Edition Shadowless Charizard Holo",
    ]
    for title in example_titles:
        grade_info = extract_grade_from_title(title)
        if grade_info:
            company, grade = grade_info
            label = create_label_json(company, grade, title)
            print(f"Title   : {title}")
            print(f"Parsed  : {company} {grade}")
            print(f"Label   : {json.dumps(label, indent=10)}")
            print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape eBay for labeled card grading data')
    parser.add_argument('--max', type=int, default=RESULTS_PER_QUERY,
                        help='Max results per search query')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without downloading')
    args = parser.parse_args()

    run_scraper(max_per_query=args.max, dry_run=args.dry_run)
