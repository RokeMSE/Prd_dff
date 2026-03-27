# Alignment Validation: In-Depth Technical Breakdown

## Purpose
> **NOTE:** THE ALIGNMENT PROCESS IS A SIMULATION OF HOW A HUMAN 
> WOULD APPROACH THIS PROBLEM EXCEPT WITH ALGORITHMIC TOOLS INSTEAD OF VISUAL INTUITION AND PHOTOSHOP :))

Aligns **OG (reference)** images to **process** images from different imaging stations on a semiconductor/packaging inspection line. The images share the same orientation but differ in:

- **Field of view (FOV)** some have wide black borders with white corner dots
- **Scale**: independent X/Y scale factors (non-uniform)
- **Contrast**: may be inverted between stations
- **Resolution**: different pixel dimensions

The alignment produces an **axis-aligned affine transform** with only **4 free parameters**:

> $x' = sx·x + tx$ 

> $y' = sy·y + ty$


This is deliberately simpler than a full homography (8 params) -> no rotation or shear is needed, so the reduced model is more robust.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. Load images (OG + process)                              │
│  2. Compute scale prior from image dimensions               │
│  3. Detect active region -> mask black borders              │
│  4. Generate image variants (enhanced, inverted, both)      │
│  5. Run 8 feature-matching strategies (SIFT/AKAZE ×         │
│     enhanced/CLAHE × normal/inverted)                       │
│  6. For each strategy: detect -> match -> RANSAC fit        │
│  7. Pick best result by inlier count (early exit if strong) │
│  8. Validate against detected package rectangle landmarks   │
│  9. Output diagnostic images                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Detail

### Step 1: Preprocessing: Active Region Detection

**Function:** `detect_active_region()` -> `make_active_mask()`

Some images have wide black borders (with white corner dots used for machine calibration). Features detected in these regions are noise: they don't correspond to real die/package content.

**How it works:**
1. Threshold the grayscale image at pixel value 15 -> binary mask of "bright" pixels
2. Morphological close (15×15 rect kernel) to fill small gaps inside the border
3. Find external contours -> take the largest one
4. If that contour is < 40% of the full image area, it represents the real content region -> crop to its bounding box
5. Otherwise, assume no significant border -> use full frame

**Output:** A binary mask (255 = active, 0 = border) used to constrain feature detection.

### Step 2: Preprocessing: Contrast Enhancement

**Function:** `enhance_for_alignment()`

The two images may come from completely different imaging conditions. Enhancement normalizes them for feature matching.

**Two-stage approach:**
1. **Percentile stretch**: map [p2, p98] intensity range to [0, 255]. This removes extreme outliers while maximizing dynamic range.
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) with a high clip limit (6.0): boosts local contrast to make structural landmarks (die edges, QR code, text) stand out.

### Step 3: Preprocessing: Contrast Inversion Detection

**Function:** `detect_contrast_inversion()`

Some station pairs produce inverted contrast (bright <-> dark). The aligner tests both polarities automatically.

**Detection method:**
1. Downsample both images to 200×200
2. Z-normalize both (zero-mean, unit-variance)
3. Compute Pearson correlation
4. Also check if mean brightness diverges dramatically (one > 150, other < 100)
5. If correlation < -0.15 OR brightness diverges -> inverted

**The aligner doesn't rely on this detection alone.** It always generates an inverted copy (`cv2.bitwise_not`) and runs strategies with both polarities. The best result wins regardless.

### Step 4: Image Variants

Before feature matching, four grayscale variants of the OG image are prepared:

| Variant | Description |
|---------|-------------|
| `g1` | Raw grayscale |
| `g1_enh` | Enhanced (percentile stretch + CLAHE) |
| `g1_inv` | Inverted (bitwise NOT) |
| `g1_inv_enh` | Inverted then enhanced |

For the process image: `g2` (raw) and `g2_enh` (enhanced).

### Step 5: Feature Detection & Matching (12 Strategies)

**Class:** `AxisAligner._detect_and_match()`

Three feature detectors are used, each with different strengths:

| Detector | Type | Descriptor Norm | Strengths |
|----------|------|-----------------|-----------|
| **SIFT** | Float (128-D) | L2 | Best overall accuracy, scale-invariant |
| **AKAZE** | Binary (variable) | Hamming | Good with contrast changes, faster than SIFT |

Each detector is run in **4 modes**: enhanced/CLAHE × normal/inverted = **8 total strategies**.

**Matching pipeline:**
1. **Downscale** both images to max 1500px on longest side (speed optimization)
2. Optionally apply CLAHE (for "CLAHE" strategies)
3. `detectAndCompute()` with the active-region mask
4. **BFMatcher** with k-NN (k=2), then **Lowe's ratio test** (threshold 0.75) to filter ambiguous matches
5. Rescale keypoint coordinates back to original resolution

**Early exit:** If any strategy produces ≥80 inliers with p95 error < 5px, stop immediately: no need to try remaining strategies.

### Step 6: RANSAC Axis-Aligned Affine Fitting

**Function:** `fit_axis_affine_ransac()`

This is the core geometric estimation. Unlike OpenCV's `findHomography` (which fits 8 params), this fits only 4 params for an *axis-aligned* affine.

**Algorithm:**
1. **Sample** 2 random point pairs (minimum to solve 4 unknowns: sx, sy, tx, ty)
2. **Solve exactly** from 2 pairs:
   - $sx = \frac{dst_j.x - dst_i.x}{src_j.x - src_i.x}$
   - $tx = dst_i.x - sx·src_i.x$
   - Same for Y axis independently
3. **Reject** if:
   - Scale is non-positive or extreme (< 0.05 or > 20)
   - Scale deviates > 35% from the **scale prior** (ratio of image dimensions)
   - The two source points are too close in X or Y (< 1px separation -> numerically unstable)
4. **Count inliers**: points where reprojection error < threshold (default 5px)
5. **Repeat** for 3000 iterations, keep best hypothesis
6. **Refine** on inliers using least-squares (independent X and Y regression)
7. **Recompute** errors with refined parameters, report p95 of inlier errors

**Why custom RANSAC instead of OpenCV's?**
- OpenCV doesn't have an axis-aligned affine model
- The 4-parameter model is much more constrained, reducing false positives
- The scale prior from image dimensions provides a strong sanity check
- Only 2 points needed per hypothesis (vs. 4 for homography) -> faster convergence

### Step 7: Quality Assessment

The `AxisAffine.ok` property requires:
- **≥10 inliers** (enough statistical support)
- **p95 reprojection error < 50px** (reasonable geometric accuracy)

The `adaptive_pad` property computes safety padding: `30 + 5.0 × p95_error` pixels: used downstream when mapping defect ROIs to account for alignment uncertainty.

### Step 8: Landmark Validation

**Function:** `validate_with_landmarks()`

Independent cross-check using structural landmarks:
1. Detect the **package rectangle** in both images using contour detection (or Hough lines as fallback)
2. Map OG rectangle corners through the affine transform
3. Compare mapped corners to detected process corners
4. If max corner error < 30px -> "OK", otherwise -> "WARN"

**Package rectangle detection** (`find_package_rect()`):
1. Enhance + Canny edge detection
2. Find external contours
3. Filter by area (5-95% of image) and polygon simplicity (4-6 vertices)
4. Fallback: Hough line detection -> find horizontal/vertical line clusters -> form rectangle

### Step 9: Diagnostic Outputs

Four visualization images are saved:

| Diagnostic | What it shows |
|------------|---------------|
| **Checkerboard** | Alternating 80px blocks from process and warped-OG: misalignment shows as discontinuities at block boundaries |
| **Alpha overlay** | 50% blend of process and warped-OG: perfect alignment = sharp, misalignment = ghosting |
| **Edge overlay** | Canny edges from both images colored cyan (OG) and magenta (process), white where overlapping: good alignment = mostly white |
| **Side-by-side** | OG, Process, and Warped-OG thumbnails with metric bar (sx, sy, tx, ty, inliers, p95, method) |

---

## Key Data Structures

### `AxisAffine` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `sx`, `sy` | float | Independent X/Y scale factors |
| `tx`, `ty` | float | X/Y translation (pixels, in process space) |
| `inliers` | int | Number of RANSAC inlier matches |
| `total_matches` | int | Total matches before RANSAC |
| `reproj_p95` | float | 95th percentile reprojection error (px) |
| `method` | str | Which strategy produced this result |

**Key methods:**
- `forward_pt(x, y)`: map OG point to process coordinates
- `inverse()`: compute the reverse transform (process -> OG)
- `to_2x3()` / `to_3x3()`: matrix forms for `cv2.warpAffine` / `cv2.perspectiveTransform`
- `H` / `H_inv` properties: compatibility with the `AlignResult` API in `defect_traceback.py`