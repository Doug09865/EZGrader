"""
grade_mapping.py
----------------
Converts raw model sub-scores (centering, corners, edges, surface)
into official grades for PSA, Beckett (BGS), SGC, and CGC grading scales.

Also provides price estimation utilities based on grade + card identity.

Usage:
    from grade_mapping import predict_grade, estimate_price, GradeScales
    
    sub_scores = {'centering': 9.2, 'corners': 9.5, 'edges': 9.0, 'surface': 8.5}
    grades = predict_grade(sub_scores)
    # {'PSA': 9, 'Beckett': '8.5', 'SGC': 9, 'CGC': 9}
"""

# ---------------------------------------------------------------------------
# Grade Weights
# Mirrors the emphasis each grading company places on each dimension.
# Surface and corners carry more weight than centering across all companies.
# ---------------------------------------------------------------------------
GRADE_WEIGHTS = {
    'centering': 0.20,
    'corners':   0.25,
    'edges':     0.25,
    'surface':   0.30,
}

GRADE_DIMENSIONS = list(GRADE_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Composite Score Calculator
# ---------------------------------------------------------------------------
def compute_composite(sub_scores: dict) -> float:
    """
    Compute a weighted composite score (0–10) from the four sub-scores.

    Args:
        sub_scores: dict with keys centering, corners, edges, surface (each 0-10)

    Returns:
        float: composite score 0–10
    """
    return sum(sub_scores[k] * GRADE_WEIGHTS[k] for k in GRADE_DIMENSIONS)


# ---------------------------------------------------------------------------
# PSA Grade Mapping
# PSA uses integer grades 1–10 (no half grades on standard service)
# PSA 10 = Gem Mint, PSA 9 = Mint, PSA 8 = NM-MT, etc.
# ---------------------------------------------------------------------------
class PSAScale:
    NAME = 'PSA'

    # (min_composite, grade_label)
    THRESHOLDS = [
        (9.75, '10 (Gem Mint)'),
        (9.40, '9 (Mint)'),
        (8.75, '8 (NM-MT)'),
        (8.00, '7 (NM)'),
        (7.25, '6 (EX-MT)'),
        (6.50, '5 (EX)'),
        (5.50, '4 (VG-EX)'),
        (4.50, '3 (VG)'),
        (3.00, '2 (GOOD)'),
        (0.00, '1 (Poor)'),
    ]

    @classmethod
    def grade(cls, sub_scores: dict) -> str:
        composite = compute_composite(sub_scores)
        for min_score, label in cls.THRESHOLDS:
            if composite >= min_score:
                return label
        return '1 (Poor)'

    @classmethod
    def numeric_grade(cls, sub_scores: dict) -> float:
        """Return numeric grade only (for calculations)."""
        label = cls.grade(sub_scores)
        return float(label.split(' ')[0])


# ---------------------------------------------------------------------------
# Beckett (BGS) Grade Mapping
# BGS uses half grades (8.0, 8.5, 9.0, 9.5) and a special "Black Label 10"
# for cards where ALL four sub-scores are 10.
# BGS grades each dimension separately AND gives an overall grade.
# ---------------------------------------------------------------------------
class BeckettScale:
    NAME = 'Beckett (BGS)'

    @classmethod
    def grade(cls, sub_scores: dict) -> str:
        composite = compute_composite(sub_scores)
        scores    = list(sub_scores.values())

        # Special case: Black Label 10 requires all sub-scores >= 9.5
        if all(s >= 9.5 for s in scores):
            return '10 Black Label'

        # BGS Pristine 10 — composite >= 9.9 and no sub-score below 9.5
        if composite >= 9.90 and min(scores) >= 9.5:
            return '10 Pristine'

        # Half-grade scale
        thresholds = [
            (9.60, '9.5 Gem Mint'),
            (9.20, '9 Mint'),
            (8.70, '8.5 NM-MT+'),
            (8.20, '8 NM-MT'),
            (7.70, '7.5 NM+'),
            (7.20, '7 NM'),
            (6.70, '6.5 EX-MT+'),
            (6.20, '6 EX-MT'),
            (5.70, '5.5 EX+'),
            (5.20, '5 EX'),
            (0.00, '4 VG-EX'),
        ]
        for min_score, label in thresholds:
            if composite >= min_score:
                return label
        return '4 VG-EX'

    @classmethod
    def sub_grades(cls, sub_scores: dict) -> dict:
        """Return individual BGS sub-grades (rounded to nearest 0.5)."""
        result = {}
        for dim, score in sub_scores.items():
            rounded = round(score * 2) / 2  # Round to nearest 0.5
            rounded = max(1.0, min(10.0, rounded))
            result[dim] = rounded
        return result


# ---------------------------------------------------------------------------
# SGC Grade Mapping
# SGC uses a numeric scale with half grades (9.5, 10)
# Known for being slightly more lenient than PSA on surface grades
# ---------------------------------------------------------------------------
class SGCScale:
    NAME = 'SGC'

    THRESHOLDS = [
        (9.80, '10 Pristine'),
        (9.50, '9.5 Mint+'),
        (9.20, '9 Mint'),
        (8.80, '8.5 NM-MT+'),
        (8.40, '8 NM-MT'),
        (7.80, '7.5 NM+'),
        (7.20, '7 NM'),
        (6.60, '6.5 EX-MT+'),
        (6.00, '6 EX-MT'),
        (5.00, '5 EX'),
        (4.00, '4 VG-EX'),
        (3.00, '3 VG'),
        (2.00, '2 GOOD'),
        (0.00, '1 Poor'),
    ]

    @classmethod
    def grade(cls, sub_scores: dict) -> str:
        composite = compute_composite(sub_scores)
        for min_score, label in cls.THRESHOLDS:
            if composite >= min_score:
                return label
        return '1 Poor'


# ---------------------------------------------------------------------------
# CGC Grade Mapping
# CGC (Certified Guaranty Company — primarily known for comics, expanding to cards)
# Uses 0.5-step scale similar to SGC
# ---------------------------------------------------------------------------
class CGCScale:
    NAME = 'CGC'

    THRESHOLDS = [
        (9.80, '10 Pristine'),
        (9.50, '9.5 Gem Mint'),
        (9.10, '9 Mint'),
        (8.60, '8.5 NM/MT+'),
        (8.10, '8 NM/MT'),
        (7.60, '7.5 NM+'),
        (7.10, '7 NM'),
        (6.10, '6 EX/NM'),
        (5.10, '5 EX'),
        (4.10, '4 VG/EX'),
        (3.10, '3 VG'),
        (2.10, '2 GOOD'),
        (0.00, '1 Poor'),
    ]

    @classmethod
    def grade(cls, sub_scores: dict) -> str:
        composite = compute_composite(sub_scores)
        for min_score, label in cls.THRESHOLDS:
            if composite >= min_score:
                return label
        return '1 Poor'


# ---------------------------------------------------------------------------
# Master predict_grade function
# ---------------------------------------------------------------------------
def predict_grade(sub_scores: dict) -> dict:
    """
    Predict grades across all supported grading scales.

    Args:
        sub_scores: dict with keys centering, corners, edges, surface (each 0-10)

    Returns:
        dict: {scale_name: grade_label, ...} plus composite score
    
    Example:
        >>> sub_scores = {'centering': 9.2, 'corners': 9.5, 'edges': 9.0, 'surface': 8.5}
        >>> predict_grade(sub_scores)
        {
            'composite': 9.1,
            'PSA': '9 (Mint)',
            'Beckett (BGS)': '9 Mint',
            'SGC': '9 Mint',
            'CGC': '9 Mint'
        }
    """
    composite = compute_composite(sub_scores)
    return {
        'composite':     round(composite, 2),
        'PSA':           PSAScale.grade(sub_scores),
        'Beckett (BGS)': BeckettScale.grade(sub_scores),
        'SGC':           SGCScale.grade(sub_scores),
        'CGC':           CGCScale.grade(sub_scores),
        'bgs_sub_grades': BeckettScale.sub_grades(sub_scores),
    }


# ---------------------------------------------------------------------------
# Grade Uncertainty / Confidence Range
# Because model predictions have error, we also compute a ±0.5 range
# ---------------------------------------------------------------------------
def predict_grade_range(sub_scores: dict, uncertainty: float = 0.5) -> dict:
    """
    Predict grade with a confidence range by applying ±uncertainty to composite.
    
    Args:
        sub_scores:   dict with centering, corners, edges, surface (0-10)
        uncertainty:  grade point uncertainty (default ±0.5)
    
    Returns:
        dict with low/mid/high grade estimates for each scale
    """
    # Low estimate: subtract uncertainty from each sub-score
    low_scores = {k: max(0, v - uncertainty) for k, v in sub_scores.items()}
    # High estimate: add uncertainty
    high_scores = {k: min(10, v + uncertainty) for k, v in sub_scores.items()}

    return {
        'mid':  predict_grade(sub_scores),
        'low':  predict_grade(low_scores),
        'high': predict_grade(high_scores),
    }


# ---------------------------------------------------------------------------
# Price Estimation
# Uses grading cost + market premium multipliers
# These are rough estimates — real prices vary heavily by card and market
# ---------------------------------------------------------------------------

# Grading service costs (USD) — update periodically
GRADING_COSTS = {
    'PSA':           25,   # PSA Standard service (as of 2024)
    'Beckett (BGS)': 22,   # BGS Standard
    'SGC':           20,   # SGC Standard
    'CGC':           25,   # CGC Standard
}

# Grade multipliers — how much more is a graded card worth vs raw NM?
# Based on historical market data (varies greatly by card)
PSA_MULTIPLIERS = {
    '10': 5.0,   # PSA 10 typically 4-8x raw NM price
    '9':  2.0,   # PSA 9  typically 1.5-3x
    '8':  1.3,
    '7':  1.0,
    '6':  0.8,
    '5':  0.6,
}

def estimate_value(
    sub_scores: dict,
    raw_nm_price: float,
    scale: str = 'PSA'
) -> dict:
    """
    Estimate the value of a card after grading.

    Args:
        sub_scores:    model-predicted sub-scores
        raw_nm_price:  current market price of the card in NM (ungraded) condition
        scale:         grading scale to estimate for ('PSA', 'Beckett (BGS)', etc.)

    Returns:
        dict with grading cost, estimated value, and expected ROI
    """
    grades       = predict_grade(sub_scores)
    grade_label  = grades[scale]
    grade_num    = grade_label.split(' ')[0]  # Extract numeric part

    # Get multiplier (default to 1.0 if grade not in table)
    multiplier  = PSA_MULTIPLIERS.get(grade_num, 1.0)
    grading_fee = GRADING_COSTS.get(scale, 25)

    graded_value    = raw_nm_price * multiplier
    net_gain        = graded_value - raw_nm_price - grading_fee
    roi_percent     = (net_gain / (raw_nm_price + grading_fee)) * 100 if raw_nm_price > 0 else 0

    return {
        'scale':           scale,
        'predicted_grade': grade_label,
        'raw_nm_price':    round(raw_nm_price, 2),
        'grading_fee':     grading_fee,
        'estimated_value': round(graded_value, 2),
        'net_gain':        round(net_gain, 2),
        'roi_percent':     round(roi_percent, 1),
        'worth_grading':   net_gain > 0,
    }


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------
class GradeScales:
    PSA     = PSAScale
    Beckett = BeckettScale
    SGC     = SGCScale
    CGC     = CGCScale


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== Grade Mapping Test ===\n")
    
    test_cases = [
        {'centering': 9.5, 'corners': 9.5, 'edges': 9.5, 'surface': 9.5},  # Near perfect
        {'centering': 9.0, 'corners': 9.5, 'edges': 9.0, 'surface': 8.5},  # PSA 9 range
        {'centering': 8.0, 'corners': 8.5, 'edges': 8.0, 'surface': 7.5},  # PSA 7-8 range
        {'centering': 6.0, 'corners': 6.5, 'edges': 6.0, 'surface': 5.5},  # PSA 5-6 range
    ]

    for i, scores in enumerate(test_cases, 1):
        print(f"Test {i}: {scores}")
        result = predict_grade(scores)
        print(f"  Composite : {result['composite']}")
        print(f"  PSA       : {result['PSA']}")
        print(f"  BGS       : {result['Beckett (BGS)']}")
        print(f"  SGC       : {result['SGC']}")
        print(f"  CGC       : {result['CGC']}")
        
        val = estimate_value(scores, raw_nm_price=50.0, scale='PSA')
        print(f"  Value est : ${val['estimated_value']} (ROI: {val['roi_percent']}%)")
        print()
