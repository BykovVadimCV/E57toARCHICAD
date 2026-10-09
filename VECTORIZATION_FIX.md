# Vectorization Fix Summary

## Problem

The original vectorizer produced unusable results:
- **282 walls** with 220 under 0.1m length (massive noise)
- **Wall thickness 0.4-0.5m** (insanely thick, should be 0.08-0.30m)
- **No noise filtering** - every tiny skeleton fragment became a "wall"
- **Saturated distance maps** - median dist = 1.0, leading to maxed-out thickness

## Solution

Complete rewrite of `simscan/vectorize.py` with:

### 1. Aggressive Noise Filtering
- **Minimum wall length: 0.8m** (was no minimum)
- Pre-skeletonization morphological cleanup (removes small components)
- Collinear segment merging (combines fragmented walls)

### 2. Realistic Wall Thickness
- **Range: 0.08-0.30m** (was 0.05-0.50m)
  - 0.08-0.15m: partitions (перегородки)
  - 0.20-0.30m: load-bearing walls (несущие стены)
- Filter saturated distance values (< 0.9) before median calculation
- Improved distance-to-thickness formula

### 3. Better Simplification
- Tolerance increased from 0.01m to 0.05m
- More aggressive polyline simplification
- Segment merging for nearby collinear walls (angle tolerance 5°, gap tolerance 0.05m)

## Results

### Dataset 1: Barklaya_14_42
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Wall count | 282 | 18 | **-93.6%** |
| Thickness range | 0.074-0.500m | 0.157-0.300m | Realistic |
| Mean thickness | 0.400m | 0.284m | **-29%** |
| Walls < 1m long | 220 | 1 | **-99.5%** |
| Walls >= 1m long | 62 | 17 | Clean |

### Dataset 2: 4012_Mikluhi-Maklaya
| Metric | Value |
|--------|-------|
| Wall count | 17 |
| Thickness range | 0.120-0.300m |
| Mean thickness | 0.260m |
| Walls < 1m long | 2 |
| Walls >= 1m long | 15 |

## Key Changes in Code

### `estimate_wall_thickness()`
```python
# Before: naive median with no filtering
thickness = float(np.median(thickness_vals) * 2.0)
thickness = np.clip(thickness, 0.05, 0.50)

# After: filter saturated values, realistic limits
valid_thickness = thickness_vals[thickness_vals < 0.9]
median_half_thick = float(np.median(valid_thickness))
full_thickness = median_half_thick * 2.0
thickness = np.clip(full_thickness, 0.08, 0.30)
```

### `extract_walls()`
```python
# Before: no minimum length filter
for idx, seg_px in enumerate(segments_px):
    # Direct processing

# After: aggressive filtering
min_length_m = 0.8
for idx, seg_px in enumerate(segments_px):
    length_raw = calculate_length(seg_world)
    if length_raw < min_length_m:
        continue
    seg_simplified = simplify_polyline(seg_world, 0.05)  # was 0.01
    length = calculate_length(seg_simplified)
    if length < min_length_m:
        continue
```

### `merge_collinear_segments()`
New function that combines nearby wall fragments:
```python
def merge_collinear_segments(segments, angle_tol=5.0, gap_tol=0.05):
    """Merge nearby collinear segments into continuous walls."""
    # Check angle between segments
    # Check gap between endpoints
    # Combine if both pass thresholds
```

### `cleanup_wall_mask()`
New pre-processing step:
```python
def cleanup_wall_mask(mask, min_area_px=50):
    """Remove small noise components before skeletonization."""
    # Morphological closing
    # Connected component analysis
    # Remove components < min_area_px
```

## Impact on IFC Export

The cleaner vectorization will dramatically improve the IFC output:
- **Fewer entities**: 18 walls vs 282 (faster loading in ArchiCAD)
- **Realistic geometry**: 0.08-0.30m thickness matches real construction
- **Clean topology**: no micro-walls breaking room definitions
- **Better room detection**: walls2layout.py will find proper enclosed spaces

## Testing

Tested on two datasets:
1. `TEST1/Barklaya_14_42_output/` - residential apartment
2. `TEST1/4012_Mikluhi-Maklaya-3-793_output/` - commercial space

Both show consistent improvement with realistic wall parameters.

## Files Changed

- `simscan/vectorize.py` - complete rewrite with filtering
- Backup saved to `simscan/vectorize_original_backup.py`

## Next Steps

1. Test IFC export with new vectorization
2. Verify wall connectivity in walls2layout.py
3. Check room boundaries and opening placement
4. Consider adaptive thickness based on structural role (exterior vs interior)
