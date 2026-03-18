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
from typing import List, Tuple, Dict
from collections import defaultdict
from alignment_validation import AxisAligner, AxisAffine

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
# Defect Origin Detector
# ============================================================
class OriginDetector:
    """Compares the defect zone between OG and process images.
    Works in PROCESS image space: extracts the OG defect patch, maps it to
    process coords, then does a local template-match refinement to correct
    residual alignment error before comparing."""

    def _prep_gray(self, img):
        """Convert to grayscale if needed."""
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _auto_invert(self, og_zone, pr_zone):
        """If OG and process have opposite contrast, invert process to match."""
        if og_zone.size < 10 or pr_zone.size < 10:
            return pr_zone, False
        corr = np.corrcoef(og_zone.ravel().astype(float),
                           pr_zone.ravel().astype(float))[0, 1]
        if corr < -0.3:
            return cv2.bitwise_not(pr_zone), True
        return pr_zone, False

    def _match_brightness(self, og_zone, pr_zone):
        """Normalize process zone brightness to match OG zone."""
        og_f = og_zone.astype(np.float32)
        pr_f = pr_zone.astype(np.float32)
        og_mean, og_std = float(og_f.mean()), float(og_f.std()) + 1e-6
        pr_mean, pr_std = float(pr_f.mean()), float(pr_f.std()) + 1e-6
        result = (pr_f - pr_mean) * (og_std / pr_std) + og_mean
        return np.clip(result, 0, 255).astype(np.uint8)

    def _warp_og_patch(self, og_gray, og_rect, pad, H, proc_shape):
        """Warp a local OG region into process space using the homography.
        This handles scale/rotation differences properly."""
        oh, ow = og_gray.shape[:2]
        ph, pw = proc_shape[:2]
        x1, y1, x2, y2 = og_rect

        # Generous context around defect in OG space
        ctx = pad + 40
        ox1 = max(0, x1 - ctx); oy1 = max(0, y1 - ctx)
        ox2 = min(ow, x2 + ctx); oy2 = min(oh, y2 + ctx)

        # Map OG region corners to process space to determine output size
        corners = np.float32([[ox1, oy1], [ox2, oy1],
                              [ox2, oy2], [ox1, oy2]]).reshape(-1, 1, 2)
        proc_corners = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        pmin = proc_corners.min(axis=0).astype(int)
        pmax = proc_corners.max(axis=0).astype(int)

        # Clamp to process image bounds
        pmin = np.maximum(pmin, 0)
        pmax = np.minimum(pmax, [pw, ph])
        out_w = int(pmax[0] - pmin[0])
        out_h = int(pmax[1] - pmin[1])
        if out_w < 5 or out_h < 5:
            return None, None, None

        # Adjust homography to output into a local crop (offset by pmin)
        T = np.array([[1, 0, -pmin[0]], [0, 1, -pmin[1]], [0, 0, 1]], dtype=np.float64)
        H_local = T @ H
        warped_og = cv2.warpPerspective(og_gray, H_local, (out_w, out_h))

        # Also map the tight defect box into the local crop coords
        tight = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        tight_proc = cv2.perspectiveTransform(tight, H_local.astype(np.float64)).reshape(-1, 2)
        tx1 = max(0, int(tight_proc[:, 0].min()))
        ty1 = max(0, int(tight_proc[:, 1].min()))
        tx2 = min(out_w, int(tight_proc[:, 0].max()))
        ty2 = min(out_h, int(tight_proc[:, 1].max()))

        return warped_og, (pmin[0], pmin[1], pmin[0] + out_w, pmin[1] + out_h), (tx1, ty1, tx2, ty2)

    def _local_refine_ncc(self, template, search_img, init_x, init_y, search_radius=15):
        """Refine position via NCC template matching. Returns (dx, dy, score)."""
        th, tw = template.shape[:2]
        sh, sw = search_img.shape[:2]
        if th >= sh or tw >= sw or th < 5 or tw < 5:
            return 0, 0, 0.0

        best_score = -1
        best_dx, best_dy = 0, 0
        for inv in [False, True]:
            tmpl = cv2.bitwise_not(template) if inv else template
            ncc = cv2.matchTemplate(search_img.astype(np.float32),
                                    tmpl.astype(np.float32),
                                    cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(ncc)
            if max_val > best_score:
                best_score = max_val
                # Expected match at center of search region minus half template
                expected_x = search_radius
                expected_y = search_radius
                best_dx = max_loc[0] - expected_x
                best_dy = max_loc[1] - expected_y

        return best_dx, best_dy, float(best_score)

    def analyze_in_process_space(self, og_gray, proc_gray,
                                  og_rect, proc_center, pad,
                                  align_result, filename, outdir=None,
                                  defect_id="") -> OriginVerdict:
        """Compare OG defect zone against process image in process space.
        Warps the OG patch using the homography to match process resolution/scale,
        then refines locally with template matching."""
        ph, pw = proc_gray.shape[:2]
        cx, cy = proc_center

        if cx < 0 or cy < 0 or cx >= pw or cy >= ph:
            return OriginVerdict(filename, "OUT_OF_VIEW", 0,
                                 f"Mapped center ({cx},{cy}) outside process ({pw}x{ph})", {})

        # Warp OG defect region into process space (handles scale + rotation)
        warp_result = self._warp_og_patch(og_gray, og_rect, pad, align_result.H, proc_gray.shape)
        if warp_result[0] is None:
            return OriginVerdict(filename, "OUT_OF_VIEW", 0, "Warped OG patch too small", {})

        warped_og, (rx1, ry1, rx2, ry2), (tx1, ty1, tx2, ty2) = warp_result
        wh, ww = warped_og.shape[:2]

        # Detect thin/line defects (high aspect ratio)
        defect_w = tx2 - tx1
        defect_h = ty2 - ty1
        min_dd = min(max(defect_w, 1), max(defect_h, 1))
        max_dd = max(defect_w, defect_h)
        is_thin = max_dd > 3 * min_dd and min_dd < 30

        # Extract corresponding process region (with extra margin for refinement)
        sr = 15  # search radius for local refinement
        px1 = max(0, rx1 - sr); py1 = max(0, ry1 - sr)
        px2 = min(pw, rx2 + sr); py2 = min(ph, ry2 + sr)
        proc_region = proc_gray[py1:py2, px1:px2]

        if proc_region.shape[0] < wh or proc_region.shape[1] < ww:
            return OriginVerdict(filename, "OUT_OF_VIEW", 0, "Process region too small for comparison", {})

        # Apply CLAHE for better comparison
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        warped_og_c = clahe.apply(warped_og)
        proc_region_c = clahe.apply(proc_region)

        # Local refinement via template match
        dx, dy, tmatch_score = self._local_refine_ncc(
            warped_og_c, proc_region_c, 0, 0, search_radius=sr)

        # Extract refined process patch (same size as warped OG)
        off_x = (rx1 - px1) + dx
        off_y = (ry1 - py1) + dy
        off_x = max(0, min(off_x, proc_region.shape[1] - ww))
        off_y = max(0, min(off_y, proc_region.shape[0] - wh))
        pr_patch = proc_region[off_y:off_y + wh, off_x:off_x + ww]

        if pr_patch.shape != warped_og.shape:
            # Edge case: resize to match
            pr_patch = cv2.resize(pr_patch, (ww, wh))

        # Contrast normalization
        pr_patch, was_inverted = self._auto_invert(warped_og, pr_patch)
        pr_patch = self._match_brightness(warped_og, pr_patch)

        # CLAHE for final comparison
        og_comp = clahe.apply(warped_og)
        pr_comp = clahe.apply(pr_patch)

        # ---- Metric 1: NCC on aligned patches ----
        ncc = cv2.matchTemplate(pr_comp.astype(np.float32),
                                og_comp.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        ncc_val = float(ncc.max()) if ncc.size > 0 else 0

        # ---- Metric 2: SSIM ----
        C1, C2 = 6.5025, 58.5225
        og_f = og_comp.astype(np.float64)
        pr_f = pr_comp.astype(np.float64)
        mu1 = cv2.GaussianBlur(og_f, (11, 11), 3)
        mu2 = cv2.GaussianBlur(pr_f, (11, 11), 3)
        sig1_sq = cv2.GaussianBlur(og_f ** 2, (11, 11), 3) - mu1 ** 2
        sig2_sq = cv2.GaussianBlur(pr_f ** 2, (11, 11), 3) - mu2 ** 2
        sig12 = cv2.GaussianBlur(og_f * pr_f, (11, 11), 3) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1_sq + sig2_sq + C2))
        ssim_val = float(ssim_map.mean())

        # ---- Metric 3: Edge overlap ----
        og_edges = cv2.Canny(og_comp, 30, 80)
        pr_edges = cv2.Canny(pr_comp, 30, 80)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pr_edges_d = cv2.dilate(pr_edges, kernel, iterations=1)
        og_edge_px = (og_edges > 0).sum()
        if og_edge_px > 0:
            overlap = ((og_edges > 0) & (pr_edges_d > 0)).sum()
            edge_overlap = float(overlap / og_edge_px)
        else:
            edge_overlap = 0.0

        # ---- Metric 4: Tight defect zone correlation ----
        if is_thin:
            expand = max(2, min_dd)
        else:
            expand = max(8, pad // 3)
        dtx1 = max(0, tx1 - expand); dty1 = max(0, ty1 - expand)
        dtx2 = min(ww, tx2 + expand); dty2 = min(wh, ty2 + expand)
        tight_og = og_comp[dty1:dty2, dtx1:dtx2]
        tight_pr = pr_comp[dty1:dty2, dtx1:dtx2]
        if tight_og.size > 10 and tight_pr.size > 10 and tight_og.shape == tight_pr.shape:
            og_t = tight_og.ravel().astype(float)
            pr_t = tight_pr.ravel().astype(float)
            og_t = (og_t - og_t.mean()) / (og_t.std() + 1e-6)
            pr_t = (pr_t - pr_t.mean()) / (pr_t.std() + 1e-6)
            tight_corr = float(np.corrcoef(og_t, pr_t)[0, 1])
        else:
            tight_corr = 0.0

        # ---- Metric 5: Line profile correlation (thin defects) ----
        profile_corr = 0.0
        if is_thin and defect_w > 0 and defect_h > 0:
            if defect_w > defect_h:
                # Horizontal thin defect — compare vertical intensity profiles
                pmargin = max(min_dd * 4, 15)
                pry1 = max(0, ty1 - pmargin)
                pry2 = min(wh, ty2 + pmargin)
                prx1, prx2 = max(0, tx1), min(ww, tx2)
                if pry2 - pry1 > 5 and prx2 - prx1 > 5:
                    og_prof = og_comp[pry1:pry2, prx1:prx2].mean(axis=1).astype(float)
                    pr_prof = pr_comp[pry1:pry2, prx1:prx2].mean(axis=1).astype(float)
                    if og_prof.std() > 1 and pr_prof.std() > 1:
                        og_pn = (og_prof - og_prof.mean()) / (og_prof.std() + 1e-6)
                        pr_pn = (pr_prof - pr_prof.mean()) / (pr_prof.std() + 1e-6)
                        profile_corr = float(np.corrcoef(og_pn, pr_pn)[0, 1])
            else:
                # Vertical thin defect — compare horizontal intensity profiles
                pmargin = max(min_dd * 4, 15)
                prx1 = max(0, tx1 - pmargin)
                prx2 = min(ww, tx2 + pmargin)
                pry1, pry2 = max(0, ty1), min(wh, ty2)
                if prx2 - prx1 > 5 and pry2 - pry1 > 5:
                    og_prof = og_comp[pry1:pry2, prx1:prx2].mean(axis=0).astype(float)
                    pr_prof = pr_comp[pry1:pry2, prx1:prx2].mean(axis=0).astype(float)
                    if og_prof.std() > 1 and pr_prof.std() > 1:
                        og_pn = (og_prof - og_prof.mean()) / (og_prof.std() + 1e-6)
                        pr_pn = (pr_prof - pr_prof.mean()) / (pr_prof.std() + 1e-6)
                        profile_corr = float(np.corrcoef(og_pn, pr_pn)[0, 1])

        # ---- Metric 6: Spatial anomaly (process-internal) ----
        # Check if the process image itself shows a localized disturbance at
        # the expected defect position.  Catches defects that change shape
        # across process steps but remain spatially anchored in the zone.
        # Uses the RAW process patch (pr_patch) to avoid CLAHE artifacts.
        spatial_anomaly = 0.0
        outer_mask = np.ones((wh, ww), dtype=bool)
        outer_mask[dty1:dty2, dtx1:dtx2] = False
        inner_px = pr_patch[dty1:dty2, dtx1:dtx2].astype(float)
        outer_px = pr_patch[outer_mask].astype(float)
        if inner_px.size > 10 and outer_px.size > 50:
            # Intensity: median-based z-score (robust to structure)
            inner_med = float(np.median(inner_px))
            outer_med = float(np.median(outer_px))
            outer_mad = float(np.median(np.abs(outer_px - outer_med))) + 1e-6
            iz = abs(inner_med - outer_med) / (outer_mad * 1.4826)  # MAD → σ

            # Texture: difference in local variability (std)
            inner_std = float(inner_px.std())
            outer_std = float(outer_px.std()) + 1e-6
            tz = abs(inner_std - outer_std) / outer_std

            # Edge density: localized edge concentration
            pr_edge_raw = cv2.Canny(pr_patch, 30, 80)
            inner_ed = float((pr_edge_raw[dty1:dty2, dtx1:dtx2] > 0).mean())
            outer_ed = float((pr_edge_raw[outer_mask] > 0).mean()) + 1e-6
            ez = abs(inner_ed - outer_ed) / outer_ed

            # Gradient energy: Laplacian concentration in defect zone
            lap = cv2.Laplacian(pr_patch, cv2.CV_64F)
            inner_lap = float(np.abs(lap[dty1:dty2, dtx1:dtx2]).mean())
            outer_lap = float(np.abs(lap[outer_mask]).mean()) + 1e-6
            gz = abs(inner_lap - outer_lap) / outer_lap

            spatial_anomaly = min(1.0,
                0.30 * min(iz, 3.0) / 3.0 +
                0.25 * min(tz, 3.0) / 3.0 +
                0.25 * min(ez, 3.0) / 3.0 +
                0.20 * min(gz, 3.0) / 3.0)

        metrics = {
            "tmatch": round(tmatch_score, 3),
            "ncc": round(ncc_val, 3),
            "ssim": round(ssim_val, 3),
            "edge_overlap": round(edge_overlap, 3),
            "tight_corr": round(tight_corr, 3),
            "profile_corr": round(profile_corr, 3),
            "spatial_anomaly": round(spatial_anomaly, 3),
            "is_thin": is_thin,
            "inverted": was_inverted,
            "refine_shift": f"({dx},{dy})",
        }

        # Save debug: warped OG | process patch | diff
        if outdir:
            diff = cv2.absdiff(og_comp, pr_comp)
            row = np.hstack([og_comp, pr_comp, diff])
            scale = max(1, 300 // max(row.shape[0], 1))
            if scale > 1:
                row = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            tag = f"_{defect_id}" if defect_id else ""
            dbg_path = os.path.join(outdir, f"DBG_{filename.replace('.jpg', '')}{tag}_OG_vs_PROC.jpg")
            cv2.imwrite(dbg_path, row)

        # ---- Decision logic ----
        # Two complementary signals:
        #   1. OG similarity — does the zone look like the OG defect? (shape-preserving)
        #   2. Spatial anomaly — is there ANY disturbance here? (shape-agnostic)
        #
        # A defect is PRESENT if either signal is strong.
        # A zone is ABSENT only if BOTH signals are weak.

        # -- OG similarity component --
        t_term = tmatch_score if tmatch_score > 0 else tmatch_score * 0.5
        s_term = ssim_val if ssim_val > 0 else ssim_val * 0.5
        tight_term = 0.25 * tight_corr if tight_corr >= 0 else 0.10 * tight_corr

        if is_thin:
            og_score = (0.15 * t_term +
                        0.10 * s_term +
                        0.05 * edge_overlap +
                        tight_term +
                        0.25 * max(profile_corr, 0))
        else:
            og_score = (0.20 * t_term +
                        0.15 * s_term +
                        0.10 * edge_overlap +
                        tight_term)

        # Context gate: if structural layer is fundamentally different,
        # discount OG similarity (but don't touch spatial anomaly).
        if tmatch_score < 0.15:
            og_score *= 0.4

        if was_inverted:
            og_score *= 0.85

        # -- Combined score: OG match OR spatial anomaly --
        # Use the stronger of the two signals, with a weighted blend
        # so that both contribute when they agree.
        score = max(og_score, 0.70 * spatial_anomaly,
                    0.50 * og_score + 0.50 * spatial_anomaly)

        metrics["og_score"] = round(og_score, 3)
        metrics["score"] = round(score, 3)

        prof_str = f", profile={profile_corr:.2f}" if is_thin else ""
        sa_str = f", sa={spatial_anomaly:.2f}"
        if score > 0.40:
            confidence = min(1.0, score)
            status = "PRESENT"
            detail = (f"Defect detected (score={score:.2f}, "
                      f"og={og_score:.2f}{sa_str}, "
                      f"tmatch={tmatch_score:.2f}, tight={tight_corr:.2f}{prof_str})")
        elif score < 0.18:
            confidence = min(1.0, 1.0 - score)
            status = "ABSENT"
            detail = (f"Zone clean (score={score:.2f}, "
                      f"og={og_score:.2f}{sa_str}, "
                      f"tmatch={tmatch_score:.2f}{prof_str})")
        else:
            confidence = 0.3
            status = "INCONCLUSIVE"
            detail = (f"Ambiguous (score={score:.2f}, og={og_score:.2f}{sa_str}, "
                      f"tmatch={tmatch_score:.2f}, tight={tight_corr:.2f}{prof_str})")

        return OriginVerdict(filename, status, confidence, detail, metrics)

    def compare_in_out(self, in_gray, out_gray, aligner,
                       in_align, out_align, og_rect, ref_rect, pad,
                       ref_w, ref_h) -> float:
        """Compare the defect zone between In and Out of the same step.
        Returns a differential score: high = zone changed (defect introduced),
        low = zone unchanged (defect was not introduced at this step).

        Both images are already aligned to the OG reference via their own
        homographies, so we map the defect zone into each and compare directly.
        """
        def _extract_zone(proc_gray, align_result):
            """Extract the defect zone from a process image using its alignment."""
            proc_center = aligner.map_point(
                ((og_rect[0]+og_rect[2])//2, (og_rect[1]+og_rect[3])//2),
                align_result)
            if proc_center is None:
                return None
            ph, pw = proc_gray.shape[:2]
            cx, cy = proc_center
            if cx < 0 or cy < 0 or cx >= pw or cy >= ph:
                return None

            proc_rect = aligner.map_rect(ref_rect, align_result, pad=pad)
            if proc_rect is None:
                return None
            px1, py1, px2, py2 = proc_rect
            px1 = max(0, px1); py1 = max(0, py1)
            px2 = min(pw, px2); py2 = min(ph, py2)
            if px2 - px1 < 5 or py2 - py1 < 5:
                return None
            return proc_gray[py1:py2, px1:px2]

        in_zone = _extract_zone(in_gray, in_align)
        out_zone = _extract_zone(out_gray, out_align)

        if in_zone is None or out_zone is None:
            return 0.0

        # Resize to common dimensions for comparison
        target_h = min(in_zone.shape[0], out_zone.shape[0])
        target_w = min(in_zone.shape[1], out_zone.shape[1])
        if target_h < 5 or target_w < 5:
            return 0.0
        in_z = cv2.resize(in_zone, (target_w, target_h)).astype(float)
        out_z = cv2.resize(out_zone, (target_w, target_h)).astype(float)

        # Normalize brightness to make comparison robust
        in_z = (in_z - in_z.mean()) / (in_z.std() + 1e-6)
        out_z = (out_z - out_z.mean()) / (out_z.std() + 1e-6)

        # Correlation: high = similar (no change), low/negative = changed
        corr = float(np.corrcoef(in_z.ravel(), out_z.ravel())[0, 1])

        # SSIM between zones
        C1, C2 = 6.5025, 58.5225
        mu1 = cv2.GaussianBlur(in_z, (7, 7), 1.5)
        mu2 = cv2.GaussianBlur(out_z, (7, 7), 1.5)
        sig1_sq = cv2.GaussianBlur(in_z**2, (7, 7), 1.5) - mu1**2
        sig2_sq = cv2.GaussianBlur(out_z**2, (7, 7), 1.5) - mu2**2
        sig12 = cv2.GaussianBlur(in_z * out_z, (7, 7), 1.5) - mu1 * mu2
        ssim_map = ((2*mu1*mu2 + C1) * (2*sig12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sig1_sq + sig2_sq + C2))
        ssim_val = float(ssim_map.mean())

        # Differential: 0 = identical (no defect introduced), 1 = very different
        diff_score = max(0.0, 1.0 - (0.5 * max(corr, 0) + 0.5 * max(ssim_val, 0)))
        return round(diff_score, 3)


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
# Image Preprocessing
# ============================================================
def normalize_contrast(img: np.ndarray, clip_limit=3.0, grid=(8, 8)) -> np.ndarray:
    """Apply CLAHE to enhance contrast on a BGR image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def enhance_process_image(img: np.ndarray, clip_limit=4.0, gamma=0.6) -> np.ndarray:
    """Aggressively enhance a dark-field process image to reveal subtle features.
    Pipeline: histogram stretch → gamma brighten → CLAHE → unsharp mask."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 1. Histogram stretch: map [p2, p98] → [0, 255]
    p2, p98 = np.percentile(l, (2, 98))
    if p98 - p2 > 10:
        l = np.clip((l.astype(float) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

    # 2. Gamma correction to brighten dark regions
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    l = cv2.LUT(l, lut)

    # 3. CLAHE for local contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Unsharp mask to sharpen edges
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    return enhanced


def auto_match_og_to_process(og: np.ndarray, proc_sample: np.ndarray) -> np.ndarray:
    """If OG has inverted contrast relative to process images, invert it.
    Then apply CLAHE to normalize both to a comparable range."""
    og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    pr_gray = cv2.cvtColor(proc_sample, cv2.COLOR_BGR2GRAY)

    # Compare mean brightness — if OG is much brighter, it's likely inverted
    og_mean = float(og_gray.mean())
    pr_mean = float(pr_gray.mean())

    # Also check correlation on a center crop
    h, w = min(og_gray.shape[0], pr_gray.shape[0]), min(og_gray.shape[1], pr_gray.shape[1])
    ch, cw = h // 4, w // 4
    og_crop = cv2.resize(og_gray, (cw, ch))
    pr_crop = cv2.resize(pr_gray, (cw, ch))
    corr = np.corrcoef(og_crop.ravel().astype(float), pr_crop.ravel().astype(float))[0, 1]

    inverted = False
    if corr < -0.2 or (og_mean > 150 and pr_mean < 100) or (og_mean < 100 and pr_mean > 150):
        og = cv2.bitwise_not(og)
        inverted = True

    og = enhance_process_image(og)
    return og, inverted


# ============================================================
# Process Image Sorting
# ============================================================
def proc_sort_key(fname):
    m = re.match(r'(\d+)_(In|Out)', fname, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        is_out = m.group(2).lower() == 'out'
        return (num, 0 if is_out else 1)
    return (0, 0)

        
# ============================================================
# Main 
# ============================================================
def main():
    uploads = './U65E35A201073'
    outdir  = './U65E35A201073/output'
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
    ref_img_raw = og_imgs[ref_key]
    print(f"\n  Primary reference frame: {ref_key}")

    proc_sorted = sorted(proc_imgs.keys(), key=proc_sort_key)
    print(f"  Process order: {proc_sorted}")

    # ---------- 2b. Preprocess: match OG contrast to process images ----------
    print("\n  Preprocessing: normalizing OG contrast to match process images...")
    proc_sample = proc_imgs[proc_sorted[0]]
    ref_img, og_was_inverted = auto_match_og_to_process(ref_img_raw, proc_sample)
    print(f"  OG inverted: {og_was_inverted}")

    # Enhanced images (aggressive) → for alignment (feature matching)
    proc_imgs_enhanced = {}
    for fname in proc_sorted:
        proc_imgs_enhanced[fname] = enhance_process_image(proc_imgs[fname])

    # Mild-normalized images → for origin comparison (preserve pixel fidelity)
    # Same inversion + CLAHE only, matching what OG gets
    ref_img_mild = ref_img  # already inverted + enhanced via auto_match
    proc_imgs_mild = {}
    for fname in proc_sorted:
        proc_imgs_mild[fname] = normalize_contrast(proc_imgs[fname], clip_limit=3.0)
    # Also make a mild OG for comparison (invert + CLAHE only, no gamma/sharpen)
    ref_img_mild = cv2.bitwise_not(ref_img_raw) if og_was_inverted else ref_img_raw.copy()
    ref_img_mild = normalize_contrast(ref_img_mild, clip_limit=3.0)

    print(f"  Enhanced {len(proc_imgs_enhanced)} process images for alignment")
    print(f"  Mild-normalized {len(proc_imgs_mild)} process images + OG for comparison")

    # ---------- 3. Align every process image to reference ----------
    print("\n" + "=" * 60)
    print("STEP 3: Aligning process images")
    print("=" * 60)

    aligner = AxisAligner()
    alignments: Dict[str, AxisAffine] = {}

    for fname in proc_sorted:
        ar = aligner.align(ref_img_raw, proc_imgs[fname])
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

        # Use mild-normalized OG for origin comparison (preserves pixel fidelity)
        og_key = ref_key
        og_img = ref_img_mild
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
            proc = proc_imgs[fname]            # raw for annotation drawing
            proc_n = proc_imgs_mild[fname]   # mild-normalized for origin analysis
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

            # --- Origin analysis: compare in PROCESS space with local refinement ---
            proc_center = aligner.map_point(d.center_pixel(w_og, h_og), ar)
            if proc_center is not None:
                proc_gray_n = cv2.cvtColor(proc_n, cv2.COLOR_BGR2GRAY)
                v = detector.analyze_in_process_space(
                    og_gray, proc_gray_n, og_rect, proc_center, pad,
                    ar, fname, outdir=outdir,
                    defect_id=f"{d.dr_sub_item}_ctr{int(d.box_ctr_x)}_{int(d.box_ctr_y)}")
            else:
                v = OriginVerdict(fname, "INCONCLUSIVE", 0, "Center mapping failed", {})
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

        # Build In/Out step pairs from verdicts and filenames
        step_pairs: Dict[int, Dict[str, OriginVerdict]] = {}
        for v in verdicts:
            if v.status in ("ALIGN_FAIL", "OUT_OF_VIEW"):
                continue
            m = re.match(r'(\d+)_(In|Out)', v.filename, re.IGNORECASE)
            if m:
                step_num = int(m.group(1))
                io_type = m.group(2).lower()
                step_pairs.setdefault(step_num, {})[io_type] = v

        # Strategy 1: In/Out DIFFERENTIAL — directly compare the defect
        # zone between In and Out of each step.  Shape-agnostic: detects
        # ANY change at the expected location, regardless of what the
        # defect looks like.
        io_diffs: Dict[int, float] = {}
        for step_num in sorted(step_pairs.keys(), reverse=True):
            pair = step_pairs[step_num]
            if 'in' not in pair or 'out' not in pair:
                continue
            in_fname = pair['in'].filename
            out_fname = pair['out'].filename
            in_ar = alignments[in_fname]
            out_ar = alignments[out_fname]
            if not in_ar.ok or not out_ar.ok:
                continue
            in_gray = cv2.cvtColor(proc_imgs_mild[in_fname], cv2.COLOR_BGR2GRAY)
            out_gray = cv2.cvtColor(proc_imgs_mild[out_fname], cv2.COLOR_BGR2GRAY)
            step_pad = max(in_ar.adaptive_pad, out_ar.adaptive_pad)
            diff_score = detector.compare_in_out(
                in_gray, out_gray, aligner,
                in_ar, out_ar, og_rect, ref_rect, step_pad,
                w_og, h_og)
            io_diffs[step_num] = diff_score
            print(f"    In/Out diff step {step_num}: {diff_score:.3f}")

        # Find step with highest differential (most change at defect zone)
        if io_diffs:
            best_step = max(io_diffs, key=lambda k: io_diffs[k])
            best_diff = io_diffs[best_step]
            if best_diff > 0.15:  # significant change detected
                origin = f"{best_step}_Out.jpg"
            elif best_diff > 0.08:
                # Moderate change — combine with OG scores for decision
                pair = step_pairs[best_step]
                out_score = pair['out'].metrics.get('score', 0)
                in_score = pair['in'].metrics.get('score', 0)
                if out_score > in_score:
                    origin = f"{best_step}_Out.jpg"

        # Strategy 2: OG-score based — clear PRESENT/ABSENT transitions
        if origin == "UNKNOWN":
            for step_num in sorted(step_pairs.keys(), reverse=True):
                pair = step_pairs[step_num]
                if 'in' not in pair or 'out' not in pair:
                    continue
                in_v, out_v = pair['in'], pair['out']
                if out_v.status == "PRESENT" and in_v.status != "PRESENT":
                    origin = out_v.filename
                    break

        # Strategy 3: Score-based fallback for all-INCONCLUSIVE
        if origin == "UNKNOWN":
            valid = [v for v in verdicts if v.status not in ("ALIGN_FAIL", "OUT_OF_VIEW")]
            if len(valid) >= 2:
                scores = [v.metrics.get('score', 0) for v in valid]
                max_score = max(scores)
                max_idx = scores.index(max_score)
                other_scores = [s for i, s in enumerate(scores) if i != max_idx]
                avg_others = sum(other_scores) / len(other_scores) if other_scores else 0
                if max_score > 0.25 and max_score > avg_others * 1.3:
                    origin = valid[max_idx].filename

        # Strategy 4: Fallbacks
        if origin == "UNKNOWN":
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

        # Process images in traceback order (newest → oldest, Out → In)
        for fname, ann in annotated_imgs:
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

        ppath = os.path.join(outdir, f"PANEL_{d.dr_sub_item}_ctr{int(d.box_ctr_x)}_{int(d.box_ctr_y)}.jpg")
        cv2.imwrite(ppath, panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

        all_results.append((d, verdicts, origin, ppath))

    # ---------- 5. Annotated OG images ----------
    print("\n" + "=" * 60)
    print("STEP 5: Annotated OG images")
    print("=" * 60)
    defects_by_og = defaultdict(list)
    for d in defects:
        # All defects use the reference OG image
        defects_by_og[ref_key].append(d)

    og_annotated = {}
    for k in og_imgs:
        # Use mild-normalized ref for the reference key, raw for others
        img = ref_img_mild if k == ref_key else og_imgs[k]
        ann = img.copy()
        h, w = ann.shape[:2]
        for dd in defects_by_og.get(k, []):
            rect = dd.to_pixel_rect(w, h)
            ann = draw_box(ann, rect, dd.dr_sub_item, pad=30)
        n = len(defects_by_og.get(k, []))
        ann = banner(ann, f"OG: {k} | {n} defect(s){' [contrast-matched]' if k == ref_key else ''}", C_RED if n else C_GREEN)
        og_annotated[k] = ann
        cv2.imwrite(os.path.join(outdir, f"OG_{k}"), ann)
        print(f"  {k}: {n} defects")

    # ---------- 6. Quarantine zone close-ups ----------
    print("\n" + "=" * 60)
    print("STEP 6: Quarantine zone close-ups")
    print("=" * 60)
    for d in defects:
        # Use the mild-normalized reference OG for quarantine zone close-ups
        img = ref_img_mild
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
            f.write(f"\n  Process-by-process verdicts (newest → oldest):\n")
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
