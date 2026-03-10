"""
FLOW
Traces back defects across manufacturing process images:
1. Parses defect bounding boxes from DVI CSV
2. Uses Frame2 as primary alignment anchor (Frame8 is pixel-aligned, inverted contrast)
3. Aligns process images via ORB feature matching + homography
4. Maps defect coordinates with adaptive padding to compensate alignment error
5. Analyzes each process image to determine if defect is present → identifies origin
6. Generates a visual traceback report with origin callout
"""

import cv2
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime 
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
ORIG_WIDTH = 3000
ORIG_HEIGHT = 5500

# Adaptive padding: base + multiplier on reprojection error
PAD_BASE = 30         # Minimum padding in OG pixel coords
PAD_ERROR_MULT = 5.0  # Multiply p95 reprojection error by this

# Colors (BGR)
C_RED     = (0, 0, 255)
C_GREEN   = (0, 200, 0)
C_CYAN    = (255, 200, 0)
C_ORANGE  = (0, 165, 255)
C_YELLOW  = (0, 255, 255)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0, 0, 0)

# Origin detection thresholds
ZONE_NONZERO_THRESH = 0.5        # Min fraction of non-black pixels in warped zone
EDGE_SIMILARITY_THRESH = 0.15    # Normalized edge overlap threshold


# ============================================================
# Data Classes
# ============================================================
@dataclass
class DefectBox:
    lot: str
    visual_id: str
    dr_result: str
    dr_sub_item: str
    box_ctr_x: float
    box_ctr_y: float
    box_side_x: float
    box_side_y: float
    image_path: str
    m_pos_x: float
    m_pos_y: float
    og_frame: str = ""

    def __post_init__(self):
        self.og_frame = os.path.basename(self.image_path)

    def to_pixel_rect(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """(x1, y1, x2, y2) in uploaded-image pixel coords."""
        sx = img_w / ORIG_WIDTH
        sy = img_h / ORIG_HEIGHT
        cx = self.box_ctr_x * sx
        cy = self.box_ctr_y * sy
        hw = max(self.box_side_x * sx / 2, 1)
        hh = max(self.box_side_y * sy / 2, 1)
        return (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))

    def center_pixel(self, img_w: int, img_h: int) -> Tuple[int, int]:
        sx = img_w / ORIG_WIDTH
        sy = img_h / ORIG_HEIGHT
        return (int(self.box_ctr_x * sx), int(self.box_ctr_y * sy))


@dataclass
class AlignResult:
    H: Optional[np.ndarray]          # OG → process
    H_inv: Optional[np.ndarray]      # process → OG
    inliers: int
    total_good: int
    reproj_p95: float                 # p95 reprojection error in pixels
    method: str
    ok: bool

    @property
    def adaptive_pad(self) -> int:
        """Padding that accounts for alignment uncertainty."""
        return int(PAD_BASE + PAD_ERROR_MULT * self.reproj_p95)


@dataclass
class OriginVerdict:
    """Whether a defect is visible in a given process image."""
    filename: str
    status: str          # PRESENT, ABSENT, INCONCLUSIVE, OUT_OF_VIEW, ALIGN_FAIL
    confidence: float    # 0-1
    detail: str          # Human-readable explanation
    metrics: dict = field(default_factory=dict)


# ============================================================
# CSV Parser
# ============================================================
def parse_csv(path: str) -> List[DefectBox]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    boxes = []
    for _, r in df.iterrows():
        boxes.append(DefectBox(
            lot=str(r['LOT']).strip(),
            visual_id=str(r['VISUAL_ID']).strip(),
            dr_result=str(r['DR_RESULT']).strip(),
            dr_sub_item=str(r['DR_SUB_ITEM']).strip(),
            box_ctr_x=float(r['BOX_CTR_X']),
            box_ctr_y=float(r['BOX_CTR_Y']),
            box_side_x=float(r['BOX_SIDE_X']),
            box_side_y=float(r['BOX_SIDE_Y']),
            image_path=str(r['IMAGE_FULL_PATH']).strip(),
            m_pos_x=float(r['M_POSITION_X']),
            m_pos_y=float(r['M_POSITION_Y']),
        ))
    return boxes


# ============================================================
# Image Aligner
# ============================================================
class Aligner:
    def __init__(self, n_feat=10000, ratio=0.75, ransac=5.0):
        self.n_feat = n_feat
        self.ratio = ratio
        self.ransac = ransac

    def _match(self, g1, g2, use_clahe):
        if use_clahe:
            cl = cv2.createCLAHE(3.0, (8, 8))
            g1, g2 = cl.apply(g1), cl.apply(g2)
        orb = cv2.ORB_create(nfeatures=self.n_feat, scoreType=cv2.ORB_HARRIS_SCORE)
        kp1, d1 = orb.detectAndCompute(g1, None)
        kp2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw = bf.knnMatch(d1, d2, k=2)
        good = [m for m, n in raw if m.distance < self.ratio * n.distance]
        return kp1, kp2, good

    def align(self, og: np.ndarray, proc: np.ndarray) -> AlignResult:
        g1 = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY) if og.ndim == 3 else og
        g2 = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY) if proc.ndim == 3 else proc

        best = AlignResult(None, None, 0, 0, 999., "", False)
        strategies = [
            ("ORB+CLAHE",  g1, g2, True),
            ("ORB_direct", g1, g2, False),
            ("ORB+inv_og", cv2.bitwise_not(g1), g2, True),
            ("ORB+inv_pr", g1, cv2.bitwise_not(g2), True),
        ]
        for name, a, b, clahe in strategies:
            kp1, kp2, good = self._match(a, b, clahe)
            if kp1 is None or len(good) < 10:
                continue
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac)
            if H is None:
                continue
            inl = int(mask.ravel().sum())
            # reprojection error
            inl_mask = mask.ravel().astype(bool)
            proj = cv2.perspectiveTransform(src[inl_mask], H)
            errs = np.sqrt(((proj - dst[inl_mask]) ** 2).sum(axis=2)).ravel()
            p95 = float(np.percentile(errs, 95)) if len(errs) > 0 else 999.
            if inl > best.inliers:
                try:
                    Hi = np.linalg.inv(H)
                    best = AlignResult(H, Hi, inl, len(good), p95, name, True)
                except np.linalg.LinAlgError:
                    pass
        return best

    def map_rect(self, rect, align: AlignResult, pad: int = 0):
        """Map (x1,y1,x2,y2) from OG → process, with padding."""
        if not align.ok:
            return None
        x1, y1, x2, y2 = rect
        # expand by pad before transforming
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad
        corners = np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).reshape(-1,1,2)
        t = cv2.perspectiveTransform(corners, align.H)
        pts = t.reshape(-1, 2)
        return (int(pts[:,0].min()), int(pts[:,1].min()),
                int(pts[:,0].max()), int(pts[:,1].max()))

    def warp_to_og(self, proc, og_shape, align: AlignResult):
        if not align.ok:
            return None
        h, w = og_shape[:2]
        return cv2.warpPerspective(proc, align.H_inv, (w, h))


# ============================================================
# Defect Origin Detector
# ============================================================
class OriginDetector:
    """Compares the defect zone between OG and each warped process image.
    Handles contrast inversion (bright-field OG vs dark-field process)."""

    def _auto_invert(self, og_zone, pr_zone, valid):
        """If OG and process have opposite contrast, invert process to match."""
        og_valid = og_zone[valid].astype(float)
        pr_valid = pr_zone[valid].astype(float)
        if og_valid.size < 10:
            return pr_zone, False
        corr = np.corrcoef(og_valid, pr_valid)[0, 1]
        if corr < -0.3:  # Negative correlation → inverted contrast
            return cv2.bitwise_not(pr_zone), True
        return pr_zone, False

    def analyze(self, og_gray: np.ndarray, warped_gray: np.ndarray,
                rect: Tuple[int,int,int,int], pad: int,
                filename: str) -> OriginVerdict:
        h, w = og_gray.shape[:2]
        x1, y1, x2, y2 = rect
        # expanded zone for analysis
        ex1 = max(0, x1 - pad)
        ey1 = max(0, y1 - pad)
        ex2 = min(w, x2 + pad)
        ey2 = min(h, y2 + pad)

        og_zone = og_gray[ey1:ey2, ex1:ex2]
        pr_zone = warped_gray[ey1:ey2, ex1:ex2]

        if og_zone.size == 0 or pr_zone.size == 0:
            return OriginVerdict(filename, "OUT_OF_VIEW", 0, "Zone empty", {})

        # Check coverage (non-black pixels in warped zone)
        nonzero = np.count_nonzero(pr_zone) / pr_zone.size
        if nonzero < ZONE_NONZERO_THRESH:
            return OriginVerdict(filename, "OUT_OF_VIEW", 0,
                                 f"Only {nonzero*100:.0f}% coverage in warped zone",
                                 {"coverage": float(nonzero)})

        valid = pr_zone > 0

        # Auto-detect and handle contrast inversion
        pr_zone_adj, was_inverted = self._auto_invert(og_zone, pr_zone, valid)

        # ---- Metric 1: NCC on expanded zone (contrast-adjusted) ----
        if og_zone.shape == pr_zone_adj.shape and og_zone.size > 0:
            ncc = cv2.matchTemplate(
                pr_zone_adj.astype(np.float32),
                og_zone.astype(np.float32),
                cv2.TM_CCOEFF_NORMED
            )
            ncc_val = float(ncc.max()) if ncc.size > 0 else 0
        else:
            ncc_val = 0

        # ---- Metric 2: Gradient-based structural similarity ----
        # Use Sobel gradients — contrast-invariant structural comparison
        og_grad = cv2.Sobel(og_zone, cv2.CV_64F, 1, 1, ksize=3)
        pr_grad = cv2.Sobel(pr_zone_adj, cv2.CV_64F, 1, 1, ksize=3)
        og_gm = np.abs(og_grad)
        pr_gm = np.abs(pr_grad)
        # Normalize gradients
        og_gn = og_gm / (og_gm.max() + 1e-6)
        pr_gn = pr_gm / (pr_gm.max() + 1e-6)
        grad_diff = np.abs(og_gn - pr_gn)[valid].mean() if valid.any() else 1.0
        grad_sim = 1.0 - float(grad_diff)

        # ---- Metric 3: Edge overlap ----
        og_edges = cv2.Canny(og_zone, 30, 80)
        pr_edges = cv2.Canny(pr_zone_adj, 30, 80)
        og_e_valid = og_edges[valid]
        pr_e_valid = pr_edges[valid]
        og_edge_px = (og_e_valid > 0).sum()
        if og_edge_px > 0:
            overlap = ((og_e_valid > 0) & (pr_e_valid > 0)).sum()
            edge_overlap = float(overlap / og_edge_px)
        else:
            edge_overlap = 0.0

        # ---- Metric 4: Tight-zone anomaly (defect-specific) ----
        # Compare pixel distribution in the tight defect box
        tight_og = og_zone[max(0,y1-ey1):min(ey2-ey1,y2-ey1),
                           max(0,x1-ex1):min(ex2-ex1,x2-ex1)]
        tight_pr = pr_zone_adj[max(0,y1-ey1):min(ey2-ey1,y2-ey1),
                               max(0,x1-ex1):min(ex2-ex1,x2-ex1)]
        if tight_og.size > 4 and tight_pr.size > 4:
            # Check if tight zone has distinct features relative to surround
            og_tight_std = float(tight_og.std())
            pr_tight_std = float(tight_pr.std())
            # Normalize both and compute correlation
            og_t = (tight_og.astype(float) - tight_og.mean()) / (og_tight_std + 1e-6)
            pr_t = (tight_pr.astype(float) - tight_pr.mean()) / (pr_tight_std + 1e-6)
            if og_t.size == pr_t.size:
                tight_corr = float(np.corrcoef(og_t.ravel(), pr_t.ravel())[0, 1])
            else:
                tight_corr = 0.0
        else:
            tight_corr = 0.0
            og_tight_std = 0.0
            pr_tight_std = 0.0

        metrics = {
            "coverage": round(float(nonzero), 3),
            "ncc": round(ncc_val, 3),
            "grad_sim": round(grad_sim, 3),
            "edge_overlap": round(edge_overlap, 3),
            "tight_corr": round(tight_corr, 3),
            "inverted": was_inverted,
        }

        # ---- Decision logic ----
        # Score combines: NCC (global match), gradient similarity (structural),
        # edge overlap (feature match), and tight correlation (defect-specific)
        score = 0.25 * max(ncc_val, 0) + 0.25 * grad_sim + 0.25 * edge_overlap + 0.25 * max(tight_corr, 0)

        if score > 0.45:
            confidence = min(1.0, score)
            status = "PRESENT"
            detail = (f"Defect zone structurally matches OG (score={score:.2f}, "
                      f"NCC={ncc_val:.2f}, grad={grad_sim:.2f}, edge={edge_overlap:.2f}, "
                      f"tight_corr={tight_corr:.2f})")
        elif score < 0.20:
            confidence = min(1.0, 1.0 - score)
            status = "ABSENT"
            detail = (f"Defect zone differs from OG (score={score:.2f}, "
                      f"NCC={ncc_val:.2f}, grad={grad_sim:.2f}, edge={edge_overlap:.2f})")
        else:
            confidence = 0.3
            status = "INCONCLUSIVE"
            detail = (f"Ambiguous match (score={score:.2f}, "
                      f"NCC={ncc_val:.2f}, grad={grad_sim:.2f}, edge={edge_overlap:.2f}, "
                      f"tight_corr={tight_corr:.2f})")

        return OriginVerdict(filename, status, confidence, detail, metrics)


# ============================================================
# Drawing Helpers
# ============================================================
def draw_box(img, rect, label, color=C_RED, pad=0, thickness=2):
    out = img.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = rect

    # Padded quarantine zone
    qx1 = max(0, x1 - pad)
    qy1 = max(0, y1 - pad)
    qx2 = min(w-1, x2 + pad)
    qy2 = min(h-1, y2 + pad)
    cv2.rectangle(out, (qx1, qy1), (qx2, qy2), C_ORANGE, 1)

    # Tight box
    bx1 = max(0, x1); by1 = max(0, y1)
    bx2 = min(w-1, x2); by2 = min(h-1, y2)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), color, thickness)

    # Crosshair
    cx, cy = (bx1+bx2)//2, (by1+by2)//2
    arm = max(pad, 15)
    cv2.line(out, (cx-arm, cy), (cx+arm, cy), color, 1)
    cv2.line(out, (cx, cy-arm), (cx, cy+arm), color, 1)

    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
    lx = max(0, bx1)
    ly = max(th+4, by1-5)
    cv2.rectangle(out, (lx, ly-th-4), (lx+tw+4, ly), color, -1)
    cv2.putText(out, label, (lx+2, ly-2), font, fs, C_WHITE, 1, cv2.LINE_AA)
    return out


def banner(img, text, color=C_CYAN, height=30):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0,0), (w, height), C_BLACK, -1)
    cv2.putText(out, text, (5, height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def hstack_padded(imgs, target_h):
    """Resize images to same height and hstack."""
    resized = []
    for im in imgs:
        if im is None:
            continue
        s = target_h / im.shape[0]
        resized.append(cv2.resize(im, None, fx=s, fy=s))
    if not resized:
        return np.zeros((target_h, 200, 3), dtype=np.uint8)
    return np.hstack(resized)


def vstack_padded(imgs, target_w):
    """Pad images to same width and vstack."""
    padded = []
    for im in imgs:
        if im is None:
            continue
        if im.shape[1] < target_w:
            p = np.zeros((im.shape[0], target_w - im.shape[1], 3), dtype=np.uint8)
            im = np.hstack([im, p])
        elif im.shape[1] > target_w:
            im = im[:, :target_w]
        padded.append(im)
    if not padded:
        return np.zeros((100, target_w, 3), dtype=np.uint8)
    return np.vstack(padded)


# ============================================================
# Process Image Sorting
# ============================================================
def proc_sort_key(fname):
    m = re.match(r'(\d+)_(In|Out)', fname, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        is_out = m.group(2).lower() == 'out'
        return (num, 1 if is_out else 0)
    return (0, 0)


# ============================================================
# Main 
# ============================================================
def main():
    uploads = './img'
    outdir  = './img/output'
    os.makedirs(outdir, exist_ok=True)

    # ---------- 1. Parse CSV ----------
    print("=" * 60)
    print("STEP 1: Parsing defect CSV")
    print("=" * 60)
    defects = parse_csv(os.path.join(uploads, 'DVI_box_data.csv'))
    for d in defects:
        print(f"  {d.dr_sub_item}: ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f}) size=({d.box_side_x:.0f}x{d.box_side_y:.0f})")

    # ---------- 2. Load images ----------
    print("\n" + "=" * 60)
    print("STEP 2: Loading images")
    print("=" * 60)

    og_imgs = {}   # filename → ndarray
    proc_imgs = {}
    og_pat  = re.compile(r'X\w+_\d+_\d+_.*FRAME\d+.*\.(jpg|jpeg|png)', re.IGNORECASE)
    proc_pat = re.compile(r'\d+_(In|Out)\.(jpg|jpeg|png)', re.IGNORECASE)

    for f in sorted(os.listdir(uploads)):
        path = os.path.join(uploads, f)
        if og_pat.match(f):
            img = cv2.imread(path)
            if img is not None:
                og_imgs[f] = img
                print(f"  OG     : {f}  {img.shape[1]}x{img.shape[0]}")
        elif proc_pat.match(f):
            img = cv2.imread(path)
            if img is not None:
                proc_imgs[f] = img
                print(f"  Process: {f}  {img.shape[1]}x{img.shape[0]}")

    # Find the primary reference frame (Frame2 — matches process images best)
    ref_key = None
    for k in og_imgs:
        if 'FRAME2' in k.upper() or 'Frame2' in k:
            ref_key = k
            break
    if ref_key is None:
        ref_key = list(og_imgs.keys())[0]
    ref_img = og_imgs[ref_key]
    print(f"\n  Primary reference frame: {ref_key}")

    proc_sorted = sorted(proc_imgs.keys(), key=proc_sort_key)
    print(f"  Process order: {proc_sorted}")

    # ---------- 3. Align every process image to reference ----------
    print("\n" + "=" * 60)
    print("STEP 3: Aligning process images")
    print("=" * 60)

    aligner = Aligner()
    alignments: Dict[str, AlignResult] = {}

    for fname in proc_sorted:
        ar = aligner.align(ref_img, proc_imgs[fname])
        alignments[fname] = ar
        tag = "OK" if ar.ok else "FAIL"
        print(f"  {fname}: {tag}  method={ar.method}  inliers={ar.inliers}  p95_err={ar.reproj_p95:.2f}px  adaptive_pad={ar.adaptive_pad}px")

    # ---------- 4. For each defect, trace back and detect origin ----------
    print("\n" + "=" * 60)
    print("STEP 4: Traceback + origin detection")
    print("=" * 60)

    detector = OriginDetector()
    all_results = []  # list of (defect, list_of_verdicts, panel_img)

    for d in defects:
        print(f"\n  --- Defect: {d.dr_sub_item}  ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f}) ---")

        # Find matching OG image for this defect
        og_key = None
        for k in og_imgs:
            if d.dr_sub_item.replace('-','').replace('_','').upper() in k.replace('-','').replace('_','').upper():
                og_key = k
                break
        if og_key is None:
            og_key = ref_key
            print(f"    No exact OG match → using {ref_key}")
        og_img = og_imgs[og_key]
        h_og, w_og = og_img.shape[:2]

        # Defect rect in OG pixel coords (tight)
        og_rect = d.to_pixel_rect(w_og, h_og)
        og_gray = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY)

        # If OG frame is not the reference, need to map coords
        # Since Frame2 and Frame8 are pixel-aligned, coords transfer directly
        ref_rect = og_rect  # same coords in reference space

        verdicts = []
        annotated_imgs = []  # (fname, annotated_img)

        for fname in proc_sorted:
            ar = alignments[fname]
            proc = proc_imgs[fname]
            ph, pw = proc.shape[:2]

            if not ar.ok or ar.inliers < 15:
                v = OriginVerdict(fname, "ALIGN_FAIL", 0,
                                  f"Alignment failed ({ar.inliers} inliers)", {})
                verdicts.append(v)
                ann = banner(proc, f"{fname} | ALIGN FAIL | INCONCLUSIVE", C_ORANGE)
                annotated_imgs.append((fname, ann))
                continue

            pad = ar.adaptive_pad

            # Map defect box from OG/ref → process with padding
            proc_rect = aligner.map_rect(ref_rect, ar, pad=pad)
            if proc_rect is None:
                v = OriginVerdict(fname, "INCONCLUSIVE", 0, "Box mapping failed", {})
                verdicts.append(v)
                continue

            px1, py1, px2, py2 = proc_rect
            in_bounds = (px1 >= -pad and py1 >= -pad and px2 < pw + pad and py2 < ph + pad)

            if not in_bounds:
                v = OriginVerdict(fname, "OUT_OF_VIEW", 0,
                                  f"Mapped box ({px1},{py1})-({px2},{py2}) outside image ({pw}x{ph})", {})
                verdicts.append(v)
                ann = banner(proc, f"{fname} | OUT OF VIEW | N/A", C_ORANGE)
                annotated_imgs.append((fname, ann))
                continue

            # --- Origin analysis: warp process → OG space and compare ---
            warped = aligner.warp_to_og(proc, og_img.shape, ar)
            if warped is not None:
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                v = detector.analyze(og_gray, warped_gray, og_rect, pad, fname)
            else:
                v = OriginVerdict(fname, "INCONCLUSIVE", 0, "Warp failed", {})
            verdicts.append(v)

            # --- Annotate process image ---
            # Draw padded box (quarantine zone is the outer rect, tight box inside)
            tight_rect = aligner.map_rect(ref_rect, ar, pad=0)
            ann = draw_box(proc, proc_rect, d.dr_sub_item, color=C_RED, pad=0, thickness=2)
            if tight_rect is not None:
                tx1, ty1, tx2, ty2 = tight_rect
                tx1 = max(0, tx1); ty1 = max(0, ty1)
                tx2 = min(pw-1, tx2); ty2 = min(ph-1, ty2)
                cv2.rectangle(ann, (tx1, ty1), (tx2, ty2), C_YELLOW, 1)

            # Status color
            scol = {
                "PRESENT": C_RED, "ABSENT": C_GREEN,
                "INCONCLUSIVE": C_ORANGE, "OUT_OF_VIEW": C_ORANGE
            }.get(v.status, C_CYAN)
            ann = banner(ann,
                f"{fname} | {ar.method} {ar.inliers}inl p95={ar.reproj_p95:.1f}px pad={pad}px | {v.status} ({v.confidence:.0%})",
                scol)
            annotated_imgs.append((fname, ann))

            # Save individual
            cv2.imwrite(os.path.join(outdir, f"TB_{d.dr_sub_item}_{fname}"), ann)
            print(f"    {fname}: {v.status} ({v.confidence:.0%}) — {v.detail}")

        # --- Determine origin ---
        origin = "UNKNOWN"
        # Traceback order: newest → oldest. Origin = first process image where defect appears.
        # Process images are sorted oldest → newest, so we scan forward.
        for v in verdicts:
            if v.status == "PRESENT":
                origin = v.filename
                break
        if origin == "UNKNOWN":
            for v in verdicts:
                if v.status == "INCONCLUSIVE":
                    origin = f"INCONCLUSIVE (possibly {v.filename})"
                    break
        if origin == "UNKNOWN" and all(v.status == "ABSENT" for v in verdicts):
            origin = "DVI (defect first appears at final inspection)"

        print(f"    >>> ORIGIN: {origin}")

        # --- Build panel ---
        THUMB_H = 600
        panel_imgs = []

        # OG image with defect
        og_ann = draw_box(og_img.copy(), og_rect, d.dr_sub_item, pad=30)
        og_ann = banner(og_ann, f"OG: {og_key}", C_RED)
        panel_imgs.append(og_ann)

        # Arrow separator
        arrow = np.zeros((THUMB_H, 50, 3), dtype=np.uint8)
        cv2.arrowedLine(arrow, (45, THUMB_H//2), (5, THUMB_H//2), C_WHITE, 2, tipLength=0.25)
        panel_imgs.append(arrow)

        # Process images in reverse (traceback direction)
        for fname, ann in reversed(annotated_imgs):
            panel_imgs.append(ann)
            arr = np.zeros((THUMB_H, 50, 3), dtype=np.uint8)
            cv2.arrowedLine(arr, (45, THUMB_H//2), (5, THUMB_H//2), C_WHITE, 2, tipLength=0.25)
            panel_imgs.append(arr)
        panel_imgs = panel_imgs[:-1]  # remove trailing arrow

        panel = hstack_padded(panel_imgs, THUMB_H)

        # Title bar
        title = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(title,
            f"Traceback: {d.dr_sub_item} | ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f}) size=({d.box_side_x:.0f}x{d.box_side_y:.0f}) | ORIGIN: {origin}",
            (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_WHITE, 1, cv2.LINE_AA)
        panel = np.vstack([title, panel])

        ppath = os.path.join(outdir, f"PANEL_{d.dr_sub_item}.jpg")
        cv2.imwrite(ppath, panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

        all_results.append((d, verdicts, origin, ppath))

    # ---------- 5. Annotated OG images ----------
    print("\n" + "=" * 60)
    print("STEP 5: Annotated OG images")
    print("=" * 60)
    defects_by_og = defaultdict(list)
    for d in defects:
        for k in og_imgs:
            if d.dr_sub_item.replace('-','').replace('_','').upper() in k.replace('-','').replace('_','').upper():
                defects_by_og[k].append(d)
                break

    og_annotated = {}
    for k, img in og_imgs.items():
        ann = img.copy()
        h, w = ann.shape[:2]
        for dd in defects_by_og.get(k, []):
            rect = dd.to_pixel_rect(w, h)
            ann = draw_box(ann, rect, dd.dr_sub_item, pad=30)
        n = len(defects_by_og.get(k, []))
        ann = banner(ann, f"OG: {k} | {n} defect(s)", C_RED if n else C_GREEN)
        og_annotated[k] = ann
        cv2.imwrite(os.path.join(outdir, f"OG_{k}"), ann)
        print(f"  {k}: {n} defects")

    # ---------- 6. Quarantine zone close-ups ----------
    print("\n" + "=" * 60)
    print("STEP 6: Quarantine zone close-ups")
    print("=" * 60)
    for d in defects:
        for k, img in og_imgs.items():
            if d.dr_sub_item.replace('-','').replace('_','').upper() in k.replace('-','').replace('_','').upper():
                h, w = img.shape[:2]
                rect = d.to_pixel_rect(w, h)
                x1, y1, x2, y2 = rect
                zp = 60
                zx1 = max(0, x1-zp); zy1 = max(0, y1-zp)
                zx2 = min(w, x2+zp); zy2 = min(h, y2+zp)
                crop = img[zy1:zy2, zx1:zx2].copy()
                cv2.rectangle(crop, (x1-zx1, y1-zy1), (x2-zx1, y2-zy1), C_RED, 2)
                crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
                zpath = os.path.join(outdir, f"ZONE_{d.dr_sub_item}_ctr{d.box_ctr_x:.0f}_{d.box_ctr_y:.0f}.jpg")
                cv2.imwrite(zpath, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"  {zpath}")
                break

    # ---------- 7. Summary Report ----------
    print("\n" + "=" * 60)
    print("STEP 7: Building summary report")
    print("=" * 60)

    W = 2800
    sections = []

    # Title
    title = np.zeros((55, W, 3), dtype=np.uint8)
    cv2.putText(title, "DEFECT TRACEBACK REPORT", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_WHITE, 2, cv2.LINE_AA)
    sections.append(title)

    # Info
    info = np.zeros((35, W, 3), dtype=np.uint8)
    lot = defects[0].lot if defects else "N/A"
    vid = defects[0].visual_id if defects else "N/A"
    cv2.putText(info, f"LOT: {lot}  |  VID: {vid}  |  Defects: {len(defects)}  |  Process images: {len(proc_imgs)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)
    sections.append(info)

    sep = np.full((3, W, 3), 80, dtype=np.uint8)
    sections.append(sep)

    # OG images row
    og_row = hstack_padded(list(og_annotated.values()), 500)
    if og_row.shape[1] < W:
        og_row = np.hstack([og_row, np.zeros((og_row.shape[0], W-og_row.shape[1], 3), dtype=np.uint8)])
    elif og_row.shape[1] > W:
        og_row = og_row[:, :W]
    sections.append(og_row)
    sections.append(sep.copy())

    # Origin summary table
    tbl_h = 35 + 30 * len(all_results)
    tbl = np.zeros((tbl_h, W, 3), dtype=np.uint8)
    cv2.putText(tbl, "ORIGIN SUMMARY", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2, cv2.LINE_AA)
    for i, (d, verdicts, origin, _) in enumerate(all_results):
        y = 55 + i * 30
        ocol = C_RED if "UNKNOWN" not in origin and "DVI" not in origin else (C_GREEN if "DVI" in origin else C_ORANGE)
        text = f"{d.dr_sub_item}  ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f})  -->  ORIGIN: {origin}"
        cv2.putText(tbl, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ocol, 1, cv2.LINE_AA)
        # Verdict chain
        chain = "  |  ".join(f"{v.filename}:{v.status}" for v in verdicts)
        cv2.putText(tbl, chain, (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1, cv2.LINE_AA)
    sections.append(tbl)
    sections.append(sep.copy())

    # Traceback panels
    for d, verdicts, origin, ppath in all_results:
        panel = cv2.imread(ppath)
        if panel is not None:
            if panel.shape[1] > W:
                s = W / panel.shape[1]
                panel = cv2.resize(panel, None, fx=s, fy=s)
            if panel.shape[1] < W:
                panel = np.hstack([panel, np.zeros((panel.shape[0], W-panel.shape[1], 3), dtype=np.uint8)])
            sections.append(panel)
            sections.append(sep.copy())

    summary = np.vstack(sections)
    spath = os.path.join(outdir, "SUMMARY_REPORT.jpg")
    cv2.imwrite(spath, summary, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Summary: {spath}")

    # ---------- 8. Origin text report ----------
    rpath = os.path.join(outdir, "ORIGIN_REPORT.txt")
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DEFECT ORIGIN TRACEBACK REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"LOT:       {lot}\n")
        f.write(f"VISUAL_ID: {vid}\n")
        f.write(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference: {ref_key}\n")
        f.write(f"Process images ({len(proc_sorted)}): {', '.join(proc_sorted)}\n")
        f.write("\n")

        for d, verdicts, origin, _ in all_results:
            f.write("-" * 70 + "\n")
            f.write(f"DEFECT: {d.dr_sub_item}\n")
            f.write(f"  Location:  ctr=({d.box_ctr_x:.0f}, {d.box_ctr_y:.0f})  "
                    f"size=({d.box_side_x:.0f} x {d.box_side_y:.0f})\n")
            f.write(f"  OG Image:  {d.og_frame}\n")
            f.write(f"  ORIGIN:    {origin}\n")
            f.write(f"\n  Process-by-process verdicts (oldest → newest):\n")
            for v in verdicts:
                f.write(f"    {v.filename:20s}  {v.status:15s}  conf={v.confidence:.0%}  {v.detail}\n")
                if v.metrics:
                    mstr = ", ".join(f"{k}={v2}" for k, v2 in v.metrics.items())
                    f.write(f"{'':26s}metrics: {mstr}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("LEGEND:\n")
        f.write("  PRESENT      — Defect zone matches OG pattern → defect likely exists at this step\n")
        f.write("  ABSENT       — Defect zone differs from OG → defect likely not yet introduced\n")
        f.write("  INCONCLUSIVE — Cannot determine with confidence\n")
        f.write("  OUT_OF_VIEW  — Defect zone is outside the process image's field of view\n")
        f.write("  ALIGN_FAIL   — Could not align process image to reference\n")
        f.write("\n  ORIGIN = earliest process image where defect is determined PRESENT.\n")
        f.write("  If all ABSENT → defect introduced at DVI/final inspection.\n")
        f.write("=" * 70 + "\n")

    print(f"  Origin report: {rpath}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return outdir


if __name__ == "__main__":
    main()
