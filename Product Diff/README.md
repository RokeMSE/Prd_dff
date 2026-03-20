# Manufacturing Defect Traceback Pipeline

This project traces manufacturing defects back to their process origin across a series of historical process images. Given a final confirmed defect (e.g., from a final Automated Optical Inspection / DVI station), the pipeline automatically aligns historical process images to the original defect reference frame and analyzes the exact spatial zone to pinpoint the specific manufacturing step where the defect was introduced.

The pipeline offers two traceback strategies under a unified image processing umbrella:
1. **Algorithmic Traceback** (`defect_traceback.py`): Uses classical computer vision similarity metrics (SSIM, NCC, Spatial Anomalies).
2. **VLM-Assisted Traceback** (`defect_traceback_vlm.py`): Uses Vision Language Models (Gemini, GPT-4o, Ollama) to contextually "look" at the process step crops.

---

## Architecture & Flow

1. **Parse Input Data**: Reads target defect bounding boxes, locations, and descriptions from a CSV (`DVI_box_data.csv`).
2. **Reference Frame Selection**: Picks the best-matching original (OG) reference frame to act as the primary structural anchor.
3. **Cross-Process Alignment**: Radically different process images (dark-field, bright-field, inverted contrasts) are globally aligned to the OG reference.
4. **Origin Detection / Evaluation**: The defect zone is mapped back to process images, evaluated for presence/absence, and the true origin is pinpointed.
5. **Visualization Reporting**: Generates checkerboards, bounding-box trace panels, and origin summary reports.

---

## In-Depth Image Processing Pipeline

The core complexity of this project stems from analyzing images captured at drastically different manufacturing stages under varying lighting, contrast, and imaging techniques.

### 1. Robust Pre-processing (`alignment_validation.py` & `defect_traceback.py`)
Manufacturing images often vary in contrast, polarity, and FOV (Field of View).
- **Active Region Masking**: Uses morphological thresholding (`cv2.MORPH_CLOSE`) to mask out black field-of-view boundaries and UI overlays (like corner dots) to ensure feature matching works on structural content rather than framing artifacts.
- **Auto-Contrast Inversion**: The pipeline dynamically compares the Mean Intensity and Cross-Correlation across a downscaled center crop layout. If correlation is `< -0.2` and one image is largely inverted, it automatically applies a bitwise inversion (`cv2.bitwise_not()`) to standardize polarities.
- **Aggressive Image Enhancement**: Dark-field process images go through a severe enhancement pipeline to force hidden structural features to become visible:
  1. *Histogram Stretch*: Maps the 2nd–98th percentiles (`np.percentile`) linearly to a `0-255` scale.
  2. *Gamma Brightening*: Non-linear `gamma=0.6` LUT to lift extreme shadows.
  3. *CLAHE*: Contrast Limited Adaptive Histogram Equalization (`clipLimit=4.0`, `8x8` tiles) locally amplifies structural gradients.
  4. *Unsharp Masking*: Subtracts a Gaussian Blur (`sigmaX=2`) from the multiplied frame to rigorously sharpen texture.

### 2. Multi-Strategy Alignment (`AxisAligner`)
Images from different stages do not share the exact same scale or registration but are guaranteed to be rotationally identical.
- **Axis-Aligned Affine Model**: Instead of using a standard full 8-DOF homography that can wildly distort due to noise, the pipeline models alignment purely as translation and independent X/Y scaling: 
  `x' = sx*x + tx`,  `y' = sy*y + ty` (4-DOF).
- **RANSAC Matcher Pipeline**: Iterates over multiple fallback feature detection techniques on downscaled copies of the masks:
  - Tries `SIFT`, `AKAZE`, and `ORB` (Harris score).
  - Tries these models over combinations of raw, CLAHE-enhanced, and contrast-inverted images.
  - Generates structural priors by matching extracted package rectangles (via Canny Edges + `cv2.approxPolyDP`).
- **Adaptive Padding Uncertainty**: Generates an adaptive padding multiplier (`adaptive_pad`) based directly on the RANSAC 95th-percentile (P95) reprojection error to capture alignment uncertainty.

### 3. Algorithmic Origin Evaluation (`defect_traceback.py`)
For classical evaluation, the alignment isn't perfect, so a local refinement occurs before similarity metrics run.
- **Local Residual Relief**: Extracts an expanded coordinate patch based on the global Affine mapping and uses Normalized Cross-Correlation (`cv2.matchTemplate`) on CLAHE-enhanced patches restricted to a localized radius. This fixes micro-pixel alignment drift.
- **Multi-Metric Fingerprinting**: Maps the refined process zone to the original defect and votes using:
  1. *SSIM (Structural Similarity)*: Gaussian blurred covariance check (`blur=11x11`, `sigma=3`).
  2. *Morphological Edge Overlap*: `cv2.Canny` edges from both images are dilated to allow 1px slop.
  3. *Line Profiling (For Thin Defects)*: Extracts the median row/col horizontal or vertical intensity distribution profile and calculates Pearson correlation. Great for hair-scratches.
  4. *Spatial Anomalies (Context Agnostic)*: Computes Z-Scores purely inside the process area by comparing the bounding box's inner Median Absolute Deviation (MAD), Texture Variance (STD), and Laplacian Gradient Density against the immediately surrounding outer ring context.

### 4. Vision Language Model (VLM) Traceback (`defect_traceback_vlm.py`)
Replaces the complex Multi-Metric Fingerprinting with a zero-shot LLM request.
- Groups the Reference Defect patch and historically ordered Process Step crops.
- Embeds standardized visual banners and context lines via `cv2.putText()`.
- Batches crops (`VLM_BATCH_LIMIT = 50`) into a single image completion prompt to `Gemini 1.5`, `GPT-4o`, or local `Ollama` models. 
- Parses JSON outputs using spatial and historical reasoning injected directly by the foundation models.

## Diagnostic Outputs
The project outputs exhaustive diagnostic visualization directly heavily utilizing OpenCV.
- **`ALIGN_checker.jpg`**: Grid-alternating `0/1` checkerboard of the matched warped images.
- **`ALIGN_edges.jpg`**: Multi-channel overlay. Maps Original Canny Edges to Cyan, Process Edges to Magenta, providing instant sub-pixel evaluation.
- **`PANEL_*`**: Side-by-side sequence of defect patches across time for visual validation alongside algorithmic confidence scores or VLM reasoning.
