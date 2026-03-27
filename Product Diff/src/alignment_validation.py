"""
Alignment Validation — axis-aligned affine aligner for OG <-> process images
===========================================================================
Key assumptions (verified via manual alignment):
  - Same orientation in all images (no flip / rotation needed)
  - Contrast may be inverted between OG and process images
  - Different FOV: some images have wide black borders with white corner dots
  - Different contrast / filter across imaging stations
  - X and Y axes can have INDEPENDENT scale factors
  - Clear structural landmarks: package rectangle, central die square,
    QR code (bottom-right), unit text (bottom-left)

Algorithm:
  1. Detect & mask black-border FOV regions so features come from real content
  2. Enhance contrast aggressively to reveal structural landmarks
  3. Handle contrast inversion (auto-detect + test both polarities)
  4. Feature matching (SIFT / AKAZE / ORB) -> fit axis-aligned affine via RANSAC 
  (NOTE: RANSAC is used to mitigate the influence of outliers, since all the images are SO noisy 
  and have many non-overlapping features. The axis-aligned affine model is chosen for its robustness, 
  as it has fewer degrees of freedom compared to a full homography, and the assumptions about the 
  images suggest that rotation/shear are not needed.)
     Model: x' = sx·x + tx,  y' = sy·y + ty   (4 free parameters)
  5. Use image-size ratio as sanity prior on (sx, sy)
  6. Generate visual diagnostics (checkerboard, overlay, side-by-side)

Usage — standalone:
  python alignment_validation.py <og_image> <process_image_or_dir> [-o outdir]

Usage — as module:
  from alignment_validation import AxisAligner, AxisAffine
  aligner = AxisAligner()
  affine  = aligner.align(og_bgr, proc_bgr)
"""
import cv2
import numpy as np
import os
import re
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

# Data Classes
@dataclass
class AxisAffine:
    """Axis-aligned affine: x' = sx·x + tx,  y' = sy·y + ty"""
    sx: float
    sy: float
    tx: float
    ty: float
    inliers: int = 0
    total_matches: int = 0
    reproj_p95: float = 999.0
    method: str = ""

    @property
    def ok(self) -> bool:
        return self.inliers >= 10 and self.reproj_p95 < 50

    @property
    def adaptive_pad(self) -> int:
        """Padding that accounts for alignment uncertainty (matches AlignResult API)."""
        return int(30 + 5.0 * self.reproj_p95)

    # ---- point / rect mapping (OG -> process) ----
    def forward_pt(self, x: float, y: float) -> Tuple[float, float]:
        return (self.sx * x + self.tx, self.sy * y + self.ty)

    def forward_pts(self, pts: np.ndarray) -> np.ndarray:
        """pts: Nx2"""
        out = np.empty_like(pts, dtype=np.float64)
        out[:, 0] = self.sx * pts[:, 0] + self.tx
        out[:, 1] = self.sy * pts[:, 1] + self.ty
        return out

    def inverse(self) -> Optional["AxisAffine"]:
        if abs(self.sx) < 1e-10 or abs(self.sy) < 1e-10:
            return None
        return AxisAffine(
            1.0 / self.sx, 1.0 / self.sy,
            -self.tx / self.sx, -self.ty / self.sy,
            self.inliers, self.total_matches,
            self.reproj_p95, self.method,
        )

    # ---- matrix forms ----
    def to_2x3(self) -> np.ndarray: 
        """For cv2.warpAffine."""
        return np.array([
            [self.sx, 0, self.tx],
            [0, self.sy, self.ty],
        ], dtype=np.float64)

    def to_3x3(self) -> np.ndarray:
        """For cv2.perspectiveTransform / drop-in with existing code."""
        return np.array([
            [self.sx, 0, self.tx],
            [0, self.sy, self.ty],
            [0, 0, 1],
        ], dtype=np.float64)

    # ---- compatibility aliases (match AlignResult in defect_traceback.py) ----
    @property
    def H(self) -> np.ndarray:
        return self.to_3x3()

    @property
    def H_inv(self) -> Optional[np.ndarray]:
        inv = self.inverse()
        return inv.to_3x3() if inv else None


# Preprocessing
def detect_active_region(gray: np.ndarray, border_thresh: int = 15,
                         min_fraction: float = 0.4) -> Tuple[int, int, int, int]:
    """Bounding box of active content, ignoring black borders / corner dots.

    Returns (x1, y1, x2, y2) of the active region.
    If the image has no significant border the full frame is returned.
    """
    _, mask = cv2.threshold(gray, border_thresh, 255, cv2.THRESH_BINARY)

    # Close small gaps (text / noise inside border)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, gray.shape[1], gray.shape[0])

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    img_area = gray.shape[0] * gray.shape[1]
    if w * h < min_fraction * img_area:
        # Active area is much smaller → real border present
        return (x, y, x + w, y + h)

    return (0, 0, gray.shape[1], gray.shape[0])


def make_active_mask(gray: np.ndarray, border_thresh: int = 15) -> np.ndarray:
    """Binary mask (255 = active content, 0 = black border)."""
    x1, y1, x2, y2 = detect_active_region(gray, border_thresh)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def detect_contrast_inversion(g1: np.ndarray, g2: np.ndarray) -> bool:
    """True if the two grayscale images appear contrast-inverted."""
    small = (200, 200)
    a = cv2.resize(g1, small).ravel().astype(np.float64)
    b = cv2.resize(g2, small).ravel().astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    corr = float(np.dot(a, b) / len(a))

    m1, m2 = float(g1.mean()), float(g2.mean())
    bright_inv = (m1 > 150 and m2 < 100) or (m1 < 100 and m2 > 150)
    return corr < -0.15 or bright_inv


def enhance_for_alignment(gray: np.ndarray, clip_limit: float = 6.0) -> np.ndarray:
    """Aggressive enhancement to reveal structural landmarks (edges, QR, text)."""
    # Histogram stretch [p2, p98] → [0, 255]
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 - p2 > 10:
        out = np.clip((gray.astype(np.float64) - p2) / (p98 - p2) * 255,
                       0, 255).astype(np.uint8)
    else:
        out = gray.copy()

    # CLAHE with high clip limit for local contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out = clahe.apply(out)
    return out


# RANSAC axis-aligned affine fitter
def _fit_from_two(src: np.ndarray, dst: np.ndarray,
                  i: int, j: int) -> Optional[Tuple[float, float, float, float]]:
    """Solve (sx, sy, tx, ty) exactly from two point pairs."""
    dx_s = src[j, 0] - src[i, 0]
    dy_s = src[j, 1] - src[i, 1]
    if abs(dx_s) < 1.0 or abs(dy_s) < 1.0:
        return None
    sx = (dst[j, 0] - dst[i, 0]) / dx_s
    tx = dst[i, 0] - sx * src[i, 0]
    sy = (dst[j, 1] - dst[i, 1]) / dy_s
    ty = dst[i, 1] - sy * src[i, 1]
    return (sx, sy, tx, ty)

def fit_axis_affine_ransac(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    threshold: float = 5.0,
    max_iter: int = 1000,
    scale_prior: Optional[Tuple[float, float]] = None,
    scale_tolerance: float = 0.35,
) -> Optional[AxisAffine]:
    """RANSAC fit of x' = sx·x + tx,  y' = sy·y + ty.

    Parameters
    ----------
    src_pts, dst_pts : Nx2 matched point coordinates
    threshold        : inlier reprojection distance (px)
    max_iter         : RANSAC iterations
    scale_prior      : (sx_expected, sy_expected) from image-size ratio — used to
                       reject wildly wrong hypotheses early
    scale_tolerance  : allowed relative deviation from scale_prior

    Returns AxisAffine or None.
    """
    N = len(src_pts)
    if N < 4:
        return None

    src = src_pts.reshape(-1, 2).astype(np.float64)
    dst = dst_pts.reshape(-1, 2).astype(np.float64)

    best_n = 0
    best_params = None

    for _ in range(max_iter):
        i, j = np.random.choice(N, 2, replace=False)
        params = _fit_from_two(src, dst, i, j)
        if params is None:
            continue
        sx, sy, tx, ty = params

        # Reject non-positive or extreme scale
        if sx <= 0.05 or sx > 20 or sy <= 0.05 or sy > 20:
            continue

        # If we have a scale prior, reject far-off hypotheses
        if scale_prior is not None:
            sx_exp, sy_exp = scale_prior
            if (abs(sx - sx_exp) / sx_exp > scale_tolerance or
                    abs(sy - sy_exp) / sy_exp > scale_tolerance):
                continue

        # Count inliers
        pred_x = sx * src[:, 0] + tx
        pred_y = sy * src[:, 1] + ty
        errs = np.sqrt((pred_x - dst[:, 0]) ** 2 + (pred_y - dst[:, 1]) ** 2)
        n_in = int((errs < threshold).sum())

        if n_in > best_n:
            best_n = n_in
            best_params = (sx, sy, tx, ty, errs < threshold)

    if best_params is None or best_n < 4:
        return None

    _, _, _, _, inlier_mask = best_params
    src_in = src[inlier_mask]
    dst_in = dst[inlier_mask]

    # Least-squares refinement on inliers (independent X / Y)
    Ax = np.column_stack([src_in[:, 0], np.ones(len(src_in))])
    sx, tx = np.linalg.lstsq(Ax, dst_in[:, 0], rcond=None)[0]

    Ay = np.column_stack([src_in[:, 1], np.ones(len(src_in))])
    sy, ty = np.linalg.lstsq(Ay, dst_in[:, 1], rcond=None)[0]

    # Recompute errors with refined params
    pred_x = sx * src[:, 0] + tx
    pred_y = sy * src[:, 1] + ty
    errs = np.sqrt((pred_x - dst[:, 0]) ** 2 + (pred_y - dst[:, 1]) ** 2)
    final_mask = errs < threshold
    inlier_errs = errs[final_mask]
    p95 = float(np.percentile(inlier_errs, 95)) if len(inlier_errs) > 0 else 999.0

    return AxisAffine(
        sx=float(sx), sy=float(sy), tx=float(tx), ty=float(ty),
        inliers=int(final_mask.sum()),
        total_matches=N,
        reproj_p95=p95,
    )


# Feature Matching + Axis-Aligned Aligner
class AxisAligner:
    """Drop-in replacement for Aligner in defect_traceback.py.

    Uses axis-aligned affine (4 params) instead of full homography (8 params).
    More robust because fewer degrees of freedom — rotation / shear are not needed.
    """
    def __init__(self, n_feat: int = 2000, ratio: float = 0.75,
                 ransac_thresh: float = 5.0):
        self.n_feat = n_feat
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh

    # ---- internal helpers ----
    @staticmethod
    def _downscale(gray, max_dim=1500):
        """Downscale for faster feature detection; return (scaled, factor)."""
        h, w = gray.shape[:2]
        f = min(1.0, max_dim / max(h, w))
        if f < 1.0:
            return cv2.resize(gray, None, fx=f, fy=f), f
        return gray, 1.0

    def _detect_and_match(self, g1, g2, detector, norm_type,
                          use_clahe=False, mask1=None, mask2=None):
        # Work on downscaled copies for speed
        g1s, f1 = self._downscale(g1)
        g2s, f2 = self._downscale(g2)
        m1 = cv2.resize(mask1, (g1s.shape[1], g1s.shape[0])) if mask1 is not None and f1 < 1.0 else mask1
        m2 = cv2.resize(mask2, (g2s.shape[1], g2s.shape[0])) if mask2 is not None and f2 < 1.0 else mask2

        if use_clahe:
            cl = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            g1s = cl.apply(g1s)
            g2s = cl.apply(g2s)

        kp1, d1 = detector.detectAndCompute(g1s, m1)
        kp2, d2 = detector.detectAndCompute(g2s, m2)
        if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, [], 1.0, 1.0

        bf = cv2.BFMatcher(norm_type)
        raw = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in raw if m.distance < self.ratio * n.distance]
        return kp1, kp2, good, f1, f2

    def _try_fit(self, kp1, kp2, good, name, scale_prior, f1, f2):
        """Fit axis-affine from matches. f1/f2 are downscale factors."""
        if kp1 is None or len(good) < 10:
            return None
        # Keypoint coords are in downscaled space — undo that
        src = np.float32([kp1[m.queryIdx].pt for m in good]) / f1
        dst = np.float32([kp2[m.trainIdx].pt for m in good]) / f2
        result = fit_axis_affine_ransac(
            src, dst,
            threshold=self.ransac_thresh,
            scale_prior=scale_prior,
        )
        if result is not None:
            result.method = name
        return result

    # ---- public API ----
    def align(self, og: np.ndarray, proc: np.ndarray,
              verbose: bool = False) -> AxisAffine:
        """Align OG -> process. Returns AxisAffine.

        Automatically handles:
          - Contrast inversion (tests both polarities)
          - Black-border FOV masking
          - Aggressive contrast enhancement
          - Scale prior from image dimensions
        """
        g1 = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY) if og.ndim == 3 else og.copy()
        g2 = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim == 3 else proc.copy()

        # Scale prior from image dimensions
        sx_prior = proc.shape[1] / og.shape[1]
        sy_prior = proc.shape[0] / og.shape[0]
        scale_prior = (sx_prior, sy_prior)
        if verbose:
            print(f"  Scale prior from dimensions: sx={sx_prior:.4f}  sy={sy_prior:.4f}")

        # Active-region masks (ignore black borders)
        mask1 = make_active_mask(g1)
        mask2 = make_active_mask(g2)

        # Image variants
        g1_enh = enhance_for_alignment(g1)
        g2_enh = enhance_for_alignment(g2)
        g1_inv = cv2.bitwise_not(g1)

        # Detectors
        sift = cv2.SIFT_create(nfeatures=self.n_feat)
        akaze = cv2.AKAZE_create()

        strategies = [
            # (name,              img1,        img2,   detector, norm,             clahe)
            ("SIFT+CLAHE",        g1,          g2,     sift,  cv2.NORM_L2,      True),
            ("SIFT+CLAHE+inv",    g1_inv,      g2,     sift,  cv2.NORM_L2,      True),
            ("AKAZE+enh",         g1_enh,      g2_enh, akaze, cv2.NORM_HAMMING, False),
        ]

        best: Optional[AxisAffine] = None

        for name, a, b, det, norm, clahe in strategies:
            kp1, kp2, good, f1, f2 = self._detect_and_match(
                a, b, det, norm, clahe, mask1, mask2)
            result = self._try_fit(kp1, kp2, good, name, scale_prior, f1, f2)
            if result is None:
                continue

            if verbose:
                print(f"    {name}: {result.inliers}/{result.total_matches} inl  "
                      f"p95={result.reproj_p95:.2f}px  "
                      f"sx={result.sx:.4f} sy={result.sy:.4f}")

            if best is None or result.inliers > best.inliers:
                best = result

            # Early exit: if we already have a strong result, stop trying
            if best is not None and best.inliers >= 80 and best.reproj_p95 < 5.0:
                if verbose:
                    print(f"    Early exit — strong result with {best.inliers} inliers")
                break

        if best is None:
            return AxisAffine(sx_prior, sy_prior, 0, 0, 0, 0, 999.0, "FAILED")
        return best

    # ---- coordinate mapping helpers (match Aligner API) ----

    def map_rect(self, rect, affine: AxisAffine, pad: int = 0):
        """Map (x1,y1,x2,y2) OG → process with padding."""
        if not affine.ok:
            return None
        x1, y1, x2, y2 = rect
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
        a, b = affine.forward_pt(x1, y1)
        c, d = affine.forward_pt(x2, y2)
        return (int(min(a, c)), int(min(b, d)),
                int(max(a, c)), int(max(b, d)))

    def map_point(self, pt, affine: AxisAffine):
        if not affine.ok:
            return None
        x, y = affine.forward_pt(pt[0], pt[1])
        return (int(x), int(y))

    def warp_to_og(self, proc, og_shape, affine: AxisAffine):
        inv = affine.inverse()
        if inv is None or not affine.ok:
            return None
        h, w = og_shape[:2]
        return cv2.warpAffine(proc, inv.to_2x3(), (w, h))

    def warp_to_process(self, og, proc_shape, affine: AxisAffine):
        if not affine.ok:
            return None
        h, w = proc_shape[:2]
        return cv2.warpAffine(og, affine.to_2x3(), (w, h))


# Visual Diagnostics
def _checkerboard_blend(img1: np.ndarray, img2: np.ndarray,
                        block: int = 80) -> np.ndarray:
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    h, w = img1.shape[:2]
    ys = np.arange(h) // block
    xs = np.arange(w) // block
    grid = (ys[:, None] + xs[None, :]) % 2  # 0/1 checkerboard
    mask = grid.astype(np.float32)
    if img1.ndim == 3:
        mask = mask[:, :, None]
    return (img1.astype(np.float32) * mask +
            img2.astype(np.float32) * (1 - mask)).astype(np.uint8)


def _alpha_blend(img1: np.ndarray, img2: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    if img1.shape[:2] != img2.shape[:2]:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


def _thumb(img: np.ndarray, target_h: int) -> np.ndarray:
    s = target_h / img.shape[0]
    return cv2.resize(img, None, fx=s, fy=s)


def draw_diagnostics(og: np.ndarray, proc: np.ndarray,
                     affine: AxisAffine, outdir: str,
                     label: str = "") -> Dict[str, str]:
    """Write diagnostic images and return their paths."""
    prefix = f"{label}_" if label else ""
    h_p, w_p = proc.shape[:2]
    og_warped = cv2.warpAffine(og, affine.to_2x3(), (w_p, h_p))

    paths = {}

    # 1 — Checkerboard
    checker = _checkerboard_blend(proc, og_warped)
    p = os.path.join(outdir, f"{prefix}ALIGN_checker.jpg")
    cv2.imwrite(p, checker, [cv2.IMWRITE_JPEG_QUALITY, 95])
    paths["checker"] = p

    # 2 — Alpha overlay
    overlay = _alpha_blend(proc, og_warped)
    p = os.path.join(outdir, f"{prefix}ALIGN_overlay.jpg")
    cv2.imwrite(p, overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
    paths["overlay"] = p

    # 3 — Edge overlay (edges from both images, OG=cyan, proc=magenta)
    og_gray_w = cv2.cvtColor(og_warped, cv2.COLOR_BGR2GRAY) if og_warped.ndim == 3 else og_warped
    pr_gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim == 3 else proc
    og_edges = cv2.Canny(enhance_for_alignment(og_gray_w, 4.0), 40, 120)
    pr_edges = cv2.Canny(enhance_for_alignment(pr_gray, 4.0), 40, 120)
    edge_vis = np.zeros((h_p, w_p, 3), dtype=np.uint8)
    edge_vis[:, :, 0] = pr_edges   # B → process edges in blue channel
    edge_vis[:, :, 1] = og_edges   # G → OG edges in green channel
    edge_vis[:, :, 2] = pr_edges   # R → process edges also → magenta
    # OG only green → cyan would need B too; let's just do:
    # cyan = OG edges, magenta = process edges
    edge_vis = np.zeros((h_p, w_p, 3), dtype=np.uint8)
    edge_vis[og_edges > 0] = (255, 255, 0)   # cyan for OG
    edge_vis[pr_edges > 0] = (255, 0, 255)   # magenta for process
    both = (og_edges > 0) & (pr_edges > 0)
    edge_vis[both] = (255, 255, 255)          # white where both overlap
    p = os.path.join(outdir, f"{prefix}ALIGN_edges.jpg")
    cv2.imwrite(p, edge_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
    paths["edges"] = p

    # 4 — Side-by-side with metrics
    TH = 600
    og_t = _thumb(og, TH)
    pr_t = _thumb(proc, TH)
    warp_t = _thumb(og_warped, TH)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for im, lbl in [(og_t, "OG (reference)"),
                     (pr_t, "Process"),
                     (warp_t, "OG warped to process")]:
        cv2.rectangle(im, (0, 0), (im.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(im, lbl, (5, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    gap = np.zeros((TH, 8, 3), dtype=np.uint8)
    row = np.hstack([og_t, gap, pr_t, gap, warp_t])

    mbar = np.zeros((40, row.shape[1], 3), dtype=np.uint8)
    info = (f"sx={affine.sx:.4f}  sy={affine.sy:.4f}  "
            f"tx={affine.tx:.1f}  ty={affine.ty:.1f}  "
            f"inliers={affine.inliers}/{affine.total_matches}  "
            f"p95={affine.reproj_p95:.2f}px  "
            f"method={affine.method}")
    cv2.putText(mbar, info, (5, 28), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    combined = np.vstack([mbar, row])
    p = os.path.join(outdir, f"{prefix}ALIGN_sidebyside.jpg")
    cv2.imwrite(p, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    paths["sidebyside"] = p

    return paths


# Structural landmark detection — package rectangle

def find_package_rect(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the outer package rectangle as (x1, y1, x2, y2).

    Works on enhanced grayscale. Falls back to Hough lines if contour
    detection fails.  Returns None when nothing convincing is found.
    """
    enh = enhance_for_alignment(gray, clip_limit=4.0)
    edges = cv2.Canny(enh, 50, 150)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray.shape[0] * gray.shape[1]
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.05 * img_area or area > 0.95 * img_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if 4 <= len(approx) <= 6 and area > best_area:
            best_area = area
            x, y, w, h = cv2.boundingRect(approx)
            best = (x, y, x + w, y + h)

    if best is not None:
        return best

    # Fallback — Hough lines
    h, w = gray.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=min(h, w) // 4, maxLineGap=20)
    if lines is None or len(lines) < 4:
        return None

    h_ys, v_xs = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ang = abs(np.arctan2(y2 - y1, x2 - x1))
        if ang < 0.15 or ang > np.pi - 0.15:
            h_ys.append((y1 + y2) / 2)
        elif abs(ang - np.pi / 2) < 0.15:
            v_xs.append((x1 + x2) / 2)

    if len(h_ys) < 2 or len(v_xs) < 2:
        return None
    h_ys.sort(); v_xs.sort()
    rect = (int(v_xs[0]), int(h_ys[0]), int(v_xs[-1]), int(h_ys[-1]))
    if (rect[2] - rect[0]) < w * 0.3 or (rect[3] - rect[1]) < h * 0.3:
        return None
    return rect


def validate_with_landmarks(og_gray: np.ndarray, proc_gray: np.ndarray,
                            affine: AxisAffine) -> Dict:
    """Cross-check affine alignment using detected package rectangles.

    Returns dict with comparison metrics (empty if landmarks not found).
    """
    og_rect = find_package_rect(og_gray)
    pr_rect = find_package_rect(proc_gray)
    if og_rect is None or pr_rect is None:
        return {"landmark_check": "SKIPPED — could not detect package rect"}

    # Map OG rect corners into process space
    corners = np.array([
        [og_rect[0], og_rect[1]],
        [og_rect[2], og_rect[1]],
        [og_rect[2], og_rect[3]],
        [og_rect[0], og_rect[3]],
    ], dtype=np.float64)
    mapped = affine.forward_pts(corners)

    # Compare to detected process rect corners
    pr_corners = np.array([
        [pr_rect[0], pr_rect[1]],
        [pr_rect[2], pr_rect[1]],
        [pr_rect[2], pr_rect[3]],
        [pr_rect[0], pr_rect[3]],
    ], dtype=np.float64)

    diffs = np.sqrt(((mapped - pr_corners) ** 2).sum(axis=1))
    return {
        "landmark_check": "OK" if diffs.max() < 30 else "WARN",
        "corner_errors_px": [round(d, 1) for d in diffs.tolist()],
        "max_corner_error": round(float(diffs.max()), 1),
        "mean_corner_error": round(float(diffs.mean()), 1),
        "og_rect": og_rect,
        "proc_rect": pr_rect,
    }


# Top-level API
def validate_alignment(og_path: str, proc_path: str,
                       outdir: Optional[str] = None,
                       verbose: bool = True) -> AxisAffine:
    """Align one OG <-> process pair with full diagnostics."""
    og = cv2.imread(og_path)
    proc = cv2.imread(proc_path)
    if og is None:
        raise FileNotFoundError(f"Cannot read: {og_path}")
    if proc is None:
        raise FileNotFoundError(f"Cannot read: {proc_path}")

    if verbose:
        print(f"OG:      {og_path}  ({og.shape[1]}x{og.shape[0]})")
        print(f"Process: {proc_path}  ({proc.shape[1]}x{proc.shape[0]})")
        print(f"Size-ratio prior:  sx={proc.shape[1]/og.shape[1]:.4f}  "
              f"sy={proc.shape[0]/og.shape[0]:.4f}")

    aligner = AxisAligner()
    affine = aligner.align(og, proc, verbose=verbose)

    if verbose:
        tag = "OK" if affine.ok else "FAIL"
        print(f"\nResult: {tag}")
        print(f"  Method:  {affine.method}")
        print(f"  Scale:   sx={affine.sx:.4f}  sy={affine.sy:.4f}")
        print(f"  Shift:   tx={affine.tx:.1f}  ty={affine.ty:.1f}")
        print(f"  Inliers: {affine.inliers}/{affine.total_matches}")
        print(f"  P95 err: {affine.reproj_p95:.2f}px")

    # Landmark validation
    g1 = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    lm = validate_with_landmarks(g1, g2, affine)
    if verbose and lm:
        print(f"  Landmark: {lm.get('landmark_check', 'N/A')}")
        if "max_corner_error" in lm:
            print(f"    Max corner error: {lm['max_corner_error']}px  "
                  f"Mean: {lm['mean_corner_error']}px")

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        proc_label = os.path.splitext(os.path.basename(proc_path))[0]
        paths = draw_diagnostics(og, proc, affine, outdir, label=proc_label)
        if verbose:
            print(f"\nDiagnostics saved to: {outdir}")
            for k, v in paths.items():
                print(f"  {k}: {v}")

    return affine


def validate_all(og_path: str, proc_dir: str,
                 outdir: Optional[str] = None,
                 verbose: bool = True) -> Dict[str, AxisAffine]:
    """Validate alignment for every process image in a directory."""
    if outdir is None:
        outdir = os.path.join(proc_dir, "align_validation")
    os.makedirs(outdir, exist_ok=True)

    og = cv2.imread(og_path)
    if og is None:
        raise FileNotFoundError(f"Cannot read: {og_path}")

    proc_pat = re.compile(r'\d+_(In|Out)\.(jpg|jpeg|png)', re.IGNORECASE)
    proc_files = sorted(f for f in os.listdir(proc_dir) if proc_pat.match(f))

    if verbose:
        print(f"OG: {og_path}  ({og.shape[1]}x{og.shape[0]})")
        print(f"Process images ({len(proc_files)}): {proc_files}")
        print()

    aligner = AxisAligner()
    results: Dict[str, AxisAffine] = {}

    for fname in proc_files:
        proc_path = os.path.join(proc_dir, fname)
        proc = cv2.imread(proc_path)
        if proc is None:
            continue

        if verbose:
            print(f"--- {fname} ({proc.shape[1]}x{proc.shape[0]}) ---")

        affine = aligner.align(og, proc, verbose=verbose)
        results[fname] = affine

        if verbose:
            tag = "OK" if affine.ok else "FAIL"
            print(f"  => {tag}  sx={affine.sx:.4f} sy={affine.sy:.4f} "
                  f"tx={affine.tx:.1f} ty={affine.ty:.1f}  "
                  f"inliers={affine.inliers}  p95={affine.reproj_p95:.2f}px  "
                  f"method={affine.method}")

        label = fname.replace(".jpg", "").replace(".png", "")
        draw_diagnostics(og, proc, affine, outdir, label=label)
        if verbose:
            print()

    # Summary table
    if verbose and results:
        print("=" * 80)
        print("ALIGNMENT SUMMARY")
        print("=" * 80)
        print(f"{'Image':<22s} {'OK?':<6s} {'sx':>8s} {'sy':>8s} "
              f"{'tx':>8s} {'ty':>8s} {'Inl':>6s} {'p95':>8s}  Method")
        print("-" * 80)
        for fname, af in results.items():
            tag = "OK" if af.ok else "FAIL"
            print(f"{fname:<22s} {tag:<6s} {af.sx:>8.4f} {af.sy:>8.4f} "
                  f"{af.tx:>8.1f} {af.ty:>8.1f} {af.inliers:>6d} "
                  f"{af.reproj_p95:>8.2f}  {af.method}")
        print("=" * 80)

    return results

# CLI (for when using separately just for aligning images)
def main():
    parser = argparse.ArgumentParser(
        description="Axis-aligned alignment validation for OG <-> process images")
    parser.add_argument("og", help="Path to OG reference image")
    parser.add_argument("proc", nargs="?",
                        help="Process image path, or directory for batch mode")
    parser.add_argument("--outdir", "-o", help="Output directory for diagnostics")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet

    if args.proc and os.path.isdir(args.proc):
        validate_all(args.og, args.proc, args.outdir, verbose)
    elif args.proc:
        outdir = args.outdir or os.path.join(
            os.path.dirname(args.proc), "align_validation")
        validate_alignment(args.og, args.proc, outdir, verbose)
    else:
        # No proc given — batch all process images in OG's directory
        og_dir = os.path.dirname(args.og) or "."
        validate_all(args.og, og_dir, args.outdir, verbose)


if __name__ == "__main__":
    main()
