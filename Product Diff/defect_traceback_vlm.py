"""
FLOW — VLM-Assisted Defect Origin Traceback
=============================================
Traces back defects across manufacturing process images using a
Vision Language Model (VLM) instead of algorithmic metrics (SSIM, NCC, etc.).

Pipeline:
1. Parses defect bounding boxes from DVI CSV
2. Uses Frame2 as primary alignment anchor
3. Aligns process images via AxisAligner (same as algorithmic version)
4. Maps defect coordinates with adaptive padding
5. Sends OG crop + ALL process crops in a single VLM call per defect
6. VLM determines per-image PRESENT / ABSENT / INCONCLUSIVE + origin
7. Generates visual traceback report with VLM-based origin callout

Supports: Gemini, OpenAI, Ollama (via .env configuration)
"""

import cv2
import numpy as np
import pandas as pd
import os
import re
import io
import json
import base64
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

try:
    from PIL import Image
except ImportError:
    Image = None

from alignment_validation import AxisAligner, AxisAffine

# ============================================================
# Logging
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vlm_traceback")

# ============================================================
# Configuration
# ============================================================
ORIG_WIDTH = 3000
ORIG_HEIGHT = 5500

PAD_BASE = 30
PAD_ERROR_MULT = 5.0

# Colors (BGR)
C_RED     = (0, 0, 255)
C_GREEN   = (0, 200, 0)
C_CYAN    = (255, 200, 0)
C_ORANGE  = (0, 165, 255)
C_YELLOW  = (0, 255, 255)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0, 0, 0)

# VLM patch context: how much extra padding around the defect zone
# to include in the image sent to the VLM (in process-space pixels)
VLM_CONTEXT_PAD = 60

# ============================================================
# VLM Service Layer (multi-provider)
# ============================================================
class VLMService:
    """Abstract base for VLM providers."""
    def analyze_images(self, images: list, prompt: str) -> str:
        raise NotImplementedError


class GeminiVLM(VLMService):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash",
                 base_url: Optional[str] = None):
        from google import genai
        from google.genai import types
        self._types = types
        http_options = {'baseUrl': base_url} if base_url else None
        self.client = genai.Client(api_key=api_key, http_options=http_options)
        self.model_name = model_name

    def analyze_images(self, images: list, prompt: str) -> str:
        contents = []
        for img in images:
            contents.append(img)  # PIL Image — Gemini accepts natively
        contents.append(prompt)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=self._types.GenerateContentConfig(temperature=0.2),
            )
            return response.text
        except Exception as e:
            log.error(f"Gemini VLM error: {e}")
            return f"ERROR: {e}"


class OpenAIVLM(VLMService):
    def __init__(self, api_key: str, model_name: str = "gpt-4o",
                 base_url: Optional[str] = None):
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model_name = model_name

    def _pil_to_b64(self, img) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def analyze_images(self, images: list, prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._pil_to_b64(img), "detail": "high"}
            })
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception as e:
            log.error(f"OpenAI VLM error: {e}")
            return f"ERROR: {e}"


class OllamaVLM(VLMService):
    def __init__(self, model_name: str = "qwen3-vl:4b"):
        import ollama as _ollama
        self._ollama = _ollama
        self.model_name = model_name

    def analyze_images(self, images: list, prompt: str) -> str:
        img_bytes_list = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes_list.append(buf.getvalue())
        try:
            resp = self._ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": img_bytes_list,
                }],
            )
            return resp["message"]["content"]
        except Exception as e:
            log.error(f"Ollama VLM error: {e}")
            return f"ERROR: {e}"


def create_vlm_service() -> VLMService:
    """Factory: reads .env / environment variables to create the right VLM."""
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        pass

    provider = os.getenv("VLM_PROVIDER", "gemini").lower()

    if provider == "gemini":
        return GeminiVLM(
            api_key=os.getenv("GEMINI_API_KEY", ""),
            model_name=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
            base_url=os.getenv("GEMINI_ENDPOINT"),
        )
    elif provider == "openai":
        return OpenAIVLM(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            base_url=os.getenv("OPENAI_ENDPOINT"),
        )
    elif provider == "ollama":
        return OllamaVLM(
            model_name=os.getenv("OLLAMA_MODEL", "qwen3-vl:4b"),
        )
    else:
        raise ValueError(f"Unknown VLM_PROVIDER: {provider}")


# ============================================================
# Data Classes (shared with algorithmic version)
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
    filename: str
    status: str          # PRESENT, ABSENT, INCONCLUSIVE, OUT_OF_VIEW, ALIGN_FAIL, VLM_ERROR
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
# VLM-Based Origin Detector
# ============================================================
class VLMOriginDetector:
    """Uses a VLM to trace defect origins across process images.

    Sends the OG defect crop + all process crops in a single VLM call,
    asking the model to determine where the defect originated.
    """

    def __init__(self, vlm: VLMService):
        self.vlm = vlm

    # ---- image preparation helpers ----

    def _cv2_to_pil(self, img: np.ndarray) -> "Image.Image":
        """Convert OpenCV BGR image to PIL RGB."""
        if img.ndim == 2:
            return Image.fromarray(img)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _extract_zone(self, img: np.ndarray, rect: Tuple[int, int, int, int],
                      context_pad: int = VLM_CONTEXT_PAD) -> np.ndarray:
        """Crop a zone with context padding, clamped to image bounds."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = rect
        x1 = max(0, x1 - context_pad)
        y1 = max(0, y1 - context_pad)
        x2 = min(w, x2 + context_pad)
        y2 = min(h, y2 + context_pad)
        return img[y1:y2, x1:x2].copy()

    def _label_crop(self, img: np.ndarray, label: str) -> np.ndarray:
        """Draw a label banner on a crop image."""
        out = img.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 22), C_BLACK, -1)
        cv2.putText(out, label, (4, 16),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_CYAN, 1, cv2.LINE_AA)
        return out

    def _draw_defect_box(self, img: np.ndarray, rect: Tuple[int, int, int, int],
                         zone_origin: Tuple[int, int]) -> np.ndarray:
        """Draw the tight defect bounding box on the cropped zone."""
        out = img.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        ox, oy = zone_origin
        x1, y1, x2, y2 = rect
        cv2.rectangle(out, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), C_RED, 2)
        return out

    # ---- VLM response parsing ----

    def _extract_json(self, response: str) -> str:
        """Extract JSON string from a VLM response that may contain markdown fences."""
        text = response
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return text[start:end]
        return text

    # ---- batch analysis: single VLM call per defect ----

    def analyze_all_zones(self, og_img: np.ndarray, og_rect: Tuple[int, int, int, int],
                          proc_zones: List[Dict], defect_info: str,
                          outdir: Optional[str] = None,
                          defect_id: str = "") -> Tuple[List[OriginVerdict], str]:
        """Send OG crop + all process crops in one VLM call.

        Args:
            og_img:      Full OG image (BGR, contrast-matched)
            og_rect:     (x1,y1,x2,y2) defect box in OG pixel coords
            proc_zones:  List of dicts with keys:
                           'filename', 'image' (cropped BGR zone), 'rect' (in full img coords)
                         Ordered from newest to oldest process step.
            defect_info: Defect type description (e.g. DR_SUB_ITEM)
            outdir:      Directory to save debug images
            defect_id:   Unique defect identifier for filenames

        Returns:
            (verdicts, origin_filename)
        """
        # Crop OG zone
        og_zone = self._extract_zone(og_img, og_rect, context_pad=VLM_CONTEXT_PAD)
        og_origin = (max(0, og_rect[0] - VLM_CONTEXT_PAD),
                     max(0, og_rect[1] - VLM_CONTEXT_PAD))
        og_vis = self._draw_defect_box(og_zone, og_rect, og_origin)
        og_vis = self._label_crop(og_vis, "OG (Reference - defect confirmed)")

        # Build labeled process crops
        pil_images = [self._cv2_to_pil(og_vis)]
        image_listing = []

        for i, pz in enumerate(proc_zones, 1):
            zone = pz['image']
            fname = pz['filename']
            vis = self._label_crop(zone, fname)
            pil_images.append(self._cv2_to_pil(vis))
            image_listing.append(f"  Image {i}: **{fname}**")

            # Save debug crop
            if outdir:
                tag = f"_{defect_id}" if defect_id else ""
                dbg_path = os.path.join(outdir, f"DBG_{fname.replace('.jpg', '')}{tag}_CROP.jpg")
                cv2.imwrite(dbg_path, vis)

        filenames_str = "\n".join(image_listing)

        prompt = f"""You are a semiconductor defect inspection expert. You are given cropped defect zone images from a manufacturing process timeline.

## Context
- **Defect type**: {defect_info}
- **Image 0 (first image)**: OG reference where the defect is confirmed PRESENT. The red bounding box marks the defect.
- **Images 1–{len(proc_zones)}**: Process step crops of the same spatial zone, ordered from latest to earliest:
{filenames_str}
- Process filenames follow the pattern `<step_number>_In.jpg` (before step) and `<step_number>_Out.jpg` (after step). Higher step numbers are later in the process.

## What to do
- Given these images, help me find which picture is the origin of the defect (similar to the image on the leftmost column on every pic), give me the name of that picture.

## Response format — respond with ONLY this JSON, no other text:
{{
    "per_image": [
        {{"filename": "<filename>", "status": "PRESENT" or "ABSENT" or "INCONCLUSIVE", "confidence": 0.0 to 1.0}},
        ...
    ],
    "origin": "<filename where defect first appears, or 'DVI' if absent in all>",
    "reasoning": "Brief explanation of how you traced the defect origin"
}}"""

        log.info(f"  VLM batch analyzing {len(proc_zones)} process crops for defect {defect_id}...")
        vlm_response = self.vlm.analyze_images(pil_images, prompt)
        log.info(f"  VLM response: {vlm_response[:300]}")

        return self._parse_batch_response(vlm_response, proc_zones)

    def _parse_batch_response(self, response: str,
                               proc_zones: List[Dict]) -> Tuple[List[OriginVerdict], str]:
        """Parse the batch VLM response into per-image verdicts + origin."""
        filenames = [pz['filename'] for pz in proc_zones]

        if response.startswith("ERROR:"):
            verdicts = [OriginVerdict(f, "VLM_ERROR", 0, response, {}) for f in filenames]
            return verdicts, "UNKNOWN"

        json_str = self._extract_json(response)
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"VLM batch parse error: {e}. Raw: {response[:400]}")
            verdicts = [OriginVerdict(f, "INCONCLUSIVE", 0.3,
                        f"VLM parse error: {e}", {"vlm_reasoning": response[:300]})
                        for f in filenames]
            return verdicts, "UNKNOWN"

        # Build per-image verdicts
        per_image = data.get("per_image", [])
        verdicts_map = {}
        for entry in per_image:
            fname = entry.get("filename", "")
            status = entry.get("status", "INCONCLUSIVE").upper()
            if status not in ("PRESENT", "ABSENT", "INCONCLUSIVE"):
                status = "INCONCLUSIVE"
            conf = max(0.0, min(1.0, float(entry.get("confidence", 0.5))))
            verdicts_map[fname] = (status, conf)

        reasoning = data.get("reasoning", "No reasoning provided")
        origin = data.get("origin", "UNKNOWN")

        verdicts = []
        for fname in filenames:
            if fname in verdicts_map:
                status, conf = verdicts_map[fname]
            else:
                status, conf = "INCONCLUSIVE", 0.3
            detail = f"VLM: {status} ({conf:.0%}) — {reasoning}"
            metrics = {
                "vlm_status": status,
                "vlm_confidence": round(conf, 3),
                "vlm_reasoning": reasoning[:500],
                "method": "VLM-batch",
            }
            verdicts.append(OriginVerdict(fname, status, conf, detail, metrics))

        return verdicts, origin


# ============================================================
# Drawing Helpers (same as algorithmic version)
# ============================================================
def draw_box(img, rect, label, color=C_RED, pad=0, thickness=2):
    out = img.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = rect

    qx1 = max(0, x1 - pad); qy1 = max(0, y1 - pad)
    qx2 = min(w-1, x2 + pad); qy2 = min(h-1, y2 + pad)
    cv2.rectangle(out, (qx1, qy1), (qx2, qy2), C_ORANGE, 1)

    bx1 = max(0, x1); by1 = max(0, y1)
    bx2 = min(w-1, x2); by2 = min(h-1, y2)
    cv2.rectangle(out, (bx1, by1), (bx2, by2), color, thickness)

    cx, cy = (bx1+bx2)//2, (by1+by2)//2
    arm = max(pad, 15)
    cv2.line(out, (cx-arm, cy), (cx+arm, cy), color, 1)
    cv2.line(out, (cx, cy-arm), (cx, cy+arm), color, 1)

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
    cv2.rectangle(out, (0, 0), (w, height), C_BLACK, -1)
    cv2.putText(out, text, (5, height-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def hstack_padded(imgs, target_h):
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
# Image Preprocessing (same as algorithmic version)
# ============================================================
def normalize_contrast(img: np.ndarray, clip_limit=3.0, grid=(8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def enhance_process_image(img: np.ndarray, clip_limit=4.0, gamma=0.6) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    p2, p98 = np.percentile(l, (2, 98))
    if p98 - p2 > 10:
        l = np.clip((l.astype(float) - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    l = cv2.LUT(l, lut)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    return enhanced


def auto_match_og_to_process(og: np.ndarray, proc_sample: np.ndarray):
    og_gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
    pr_gray = cv2.cvtColor(proc_sample, cv2.COLOR_BGR2GRAY)
    og_mean = float(og_gray.mean())
    pr_mean = float(pr_gray.mean())
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


def proc_sort_key(fname):
    m = re.match(r'(\d+)_(In|Out)', fname, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        is_out = m.group(2).lower() == 'out'
        return (num, 0 if is_out else 1)
    return (0, 0)


# ============================================================
# Main Pipeline
# ============================================================
def main():
    uploads = './U65E35A201073'
    outdir  = './U65E35A201073/output'
    os.makedirs(outdir, exist_ok=True)

    # ---------- 0. Initialize VLM ----------
    print("=" * 60)
    print("STEP 0: Initializing VLM service")
    print("=" * 60)
    vlm = create_vlm_service()
    provider = os.getenv("VLM_PROVIDER", "gemini")
    print(f"  Provider: {provider}")
    print(f"  VLM class: {vlm.__class__.__name__}")

    # ---------- 1. Parse CSV ----------
    print("\n" + "=" * 60)
    print("STEP 1: Parsing defect CSV")
    print("=" * 60)
    defects = parse_csv(os.path.join(uploads, 'DVI_box_data.csv'))
    for d in defects:
        print(f"  {d.dr_sub_item}: ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f}) size=({d.box_side_x:.0f}x{d.box_side_y:.0f})")

    # ---------- 2. Load images ----------
    print("\n" + "=" * 60)
    print("STEP 2: Loading images")
    print("=" * 60)

    og_imgs = {}
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

    # ---------- 2b. Preprocess ----------
    print("\n  Preprocessing: normalizing OG contrast to match process images...")
    proc_sample = proc_imgs[proc_sorted[0]]
    ref_img, og_was_inverted = auto_match_og_to_process(ref_img_raw, proc_sample)
    print(f"  OG inverted: {og_was_inverted}")

    proc_imgs_mild = {}
    for fname in proc_sorted:
        proc_imgs_mild[fname] = normalize_contrast(proc_imgs[fname], clip_limit=3.0)
    ref_img_mild = cv2.bitwise_not(ref_img_raw) if og_was_inverted else ref_img_raw.copy()
    ref_img_mild = normalize_contrast(ref_img_mild, clip_limit=3.0)

    print(f"  Mild-normalized {len(proc_imgs_mild)} process images + OG for VLM comparison")

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

    # ---------- 4. VLM-based traceback ----------
    print("\n" + "=" * 60)
    print("STEP 4: VLM-based traceback + origin detection")
    print("=" * 60)

    detector = VLMOriginDetector(vlm)
    all_results = []

    for d in defects:
        print(f"\n  --- Defect: {d.dr_sub_item}  ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f}) ---")

        og_img = ref_img_mild
        h_og, w_og = og_img.shape[:2]
        og_rect = d.to_pixel_rect(w_og, h_og)
        ref_rect = og_rect
        defect_id = f"{d.dr_sub_item}_ctr{int(d.box_ctr_x)}_{int(d.box_ctr_y)}"

        # --- Collect all valid process crops ---
        proc_zones = []     # For VLM batch call
        skip_verdicts = []  # Pre-filled verdicts for failed alignments etc.
        annotated_imgs = []
        proc_rects = {}     # fname -> (proc_rect, pad)

        for fname in proc_sorted:
            ar = alignments[fname]
            proc = proc_imgs[fname]
            proc_n = proc_imgs_mild[fname]
            ph, pw = proc.shape[:2]

            if not ar.ok or ar.inliers < 15:
                v = OriginVerdict(fname, "ALIGN_FAIL", 0,
                                  f"Alignment failed ({ar.inliers} inliers)", {})
                skip_verdicts.append(v)
                ann = banner(proc, f"{fname} | ALIGN FAIL | INCONCLUSIVE", C_ORANGE)
                annotated_imgs.append((fname, ann))
                continue

            pad = ar.adaptive_pad
            proc_rect = aligner.map_rect(ref_rect, ar, pad=pad)
            if proc_rect is None:
                v = OriginVerdict(fname, "INCONCLUSIVE", 0, "Box mapping failed", {})
                skip_verdicts.append(v)
                continue

            px1, py1, px2, py2 = proc_rect
            in_bounds = (px1 >= -pad and py1 >= -pad and px2 < pw + pad and py2 < ph + pad)

            if not in_bounds:
                v = OriginVerdict(fname, "OUT_OF_VIEW", 0,
                                  f"Mapped box ({px1},{py1})-({px2},{py2}) outside image ({pw}x{ph})", {})
                skip_verdicts.append(v)
                ann = banner(proc, f"{fname} | OUT OF VIEW | N/A", C_ORANGE)
                annotated_imgs.append((fname, ann))
                continue

            # Extract crop for VLM
            zone = detector._extract_zone(proc_n, proc_rect, context_pad=VLM_CONTEXT_PAD)
            if zone.size < 100:
                v = OriginVerdict(fname, "OUT_OF_VIEW", 0, "Cropped zone too small", {})
                skip_verdicts.append(v)
                continue

            proc_zones.append({'filename': fname, 'image': zone, 'rect': proc_rect})
            proc_rects[fname] = (proc_rect, pad)

        # --- Single VLM call for all zones ---
        if proc_zones:
            vlm_verdicts, origin = detector.analyze_all_zones(
                og_img, og_rect, proc_zones,
                defect_info=d.dr_sub_item,
                outdir=outdir, defect_id=defect_id)
        else:
            vlm_verdicts = []
            origin = "UNKNOWN"

        # Merge verdicts (skipped + VLM results) in process order
        vlm_map = {v.filename: v for v in vlm_verdicts}
        verdicts = []
        for fname in proc_sorted:
            skip_v = next((v for v in skip_verdicts if v.filename == fname), None)
            if skip_v:
                verdicts.append(skip_v)
            elif fname in vlm_map:
                verdicts.append(vlm_map[fname])

        # --- Annotate process images for panel ---
        for v in verdicts:
            fname = v.filename
            if v.status in ("ALIGN_FAIL", "OUT_OF_VIEW"):
                continue  # Already added above
            if fname not in proc_rects:
                continue
            proc_rect, pad = proc_rects[fname]
            ar = alignments[fname]
            proc = proc_imgs[fname]
            ph, pw = proc.shape[:2]

            tight_rect = aligner.map_rect(ref_rect, ar, pad=0)
            ann = draw_box(proc, proc_rect, d.dr_sub_item, color=C_RED, pad=0, thickness=2)
            if tight_rect is not None:
                tx1, ty1, tx2, ty2 = tight_rect
                tx1 = max(0, tx1); ty1 = max(0, ty1)
                tx2 = min(pw-1, tx2); ty2 = min(ph-1, ty2)
                cv2.rectangle(ann, (tx1, ty1), (tx2, ty2), C_YELLOW, 1)

            scol = {
                "PRESENT": C_RED, "ABSENT": C_GREEN,
                "INCONCLUSIVE": C_ORANGE, "OUT_OF_VIEW": C_ORANGE,
                "VLM_ERROR": C_ORANGE,
            }.get(v.status, C_CYAN)
            ann = banner(ann,
                f"{fname} | {ar.method} {ar.inliers}inl pad={pad}px | VLM: {v.status} ({v.confidence:.0%})",
                scol)
            annotated_imgs.append((fname, ann))

            cv2.imwrite(os.path.join(outdir, f"TB_{d.dr_sub_item}_{fname}"), ann)
            print(f"    {fname}: {v.status} ({v.confidence:.0%}) — {v.detail[:120]}")

        # --- Validate/fallback origin ---
        if origin == "DVI":
            origin = "DVI (defect first appears at final inspection)"
        elif origin == "UNKNOWN" or origin not in [pz['filename'] for pz in proc_zones]:
            # Fallback: use per-image verdicts
            present = [v for v in verdicts if v.status == "PRESENT"]
            if present:
                origin = max(present, key=lambda v: v.confidence).filename
            elif all(v.status == "ABSENT" for v in verdicts
                     if v.status not in ("ALIGN_FAIL", "OUT_OF_VIEW", "VLM_ERROR")):
                origin = "DVI (defect first appears at final inspection)"
            else:
                inc = [v for v in verdicts if v.status == "INCONCLUSIVE"]
                if inc:
                    origin = f"INCONCLUSIVE (possibly {inc[0].filename})"

        print(f"    >>> ORIGIN: {origin}")

        # --- Build traceback panel ---
        THUMB_H = 600
        panel_imgs = []

        og_ann = draw_box(og_img.copy(), og_rect, d.dr_sub_item, pad=30)
        og_ann = banner(og_ann, f"OG: {ref_key}", C_RED)
        panel_imgs.append(og_ann)

        arrow = np.zeros((THUMB_H, 50, 3), dtype=np.uint8)
        cv2.arrowedLine(arrow, (45, THUMB_H//2), (5, THUMB_H//2), C_WHITE, 2, tipLength=0.25)
        panel_imgs.append(arrow)

        for fname, ann in annotated_imgs:
            panel_imgs.append(ann)
            arr = np.zeros((THUMB_H, 50, 3), dtype=np.uint8)
            cv2.arrowedLine(arr, (45, THUMB_H//2), (5, THUMB_H//2), C_WHITE, 2, tipLength=0.25)
            panel_imgs.append(arr)
        panel_imgs = panel_imgs[:-1]

        panel = hstack_padded(panel_imgs, THUMB_H)

        title = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(title,
            f"VLM Traceback: {d.dr_sub_item} | ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f}) size=({d.box_side_x:.0f}x{d.box_side_y:.0f}) | ORIGIN: {origin}",
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
        defects_by_og[ref_key].append(d)

    og_annotated = {}
    for k in og_imgs:
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

    title = np.zeros((55, W, 3), dtype=np.uint8)
    cv2.putText(title, "DEFECT TRACEBACK REPORT (VLM)", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, C_WHITE, 2, cv2.LINE_AA)
    sections.append(title)

    info = np.zeros((35, W, 3), dtype=np.uint8)
    lot = defects[0].lot if defects else "N/A"
    vid = defects[0].visual_id if defects else "N/A"
    cv2.putText(info, f"LOT: {lot}  |  VID: {vid}  |  Defects: {len(defects)}  |  Process: {len(proc_imgs)}  |  Method: VLM ({provider})",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)
    sections.append(info)

    sep = np.full((3, W, 3), 80, dtype=np.uint8)
    sections.append(sep)

    og_row = hstack_padded(list(og_annotated.values()), 500)
    if og_row.shape[1] < W:
        og_row = np.hstack([og_row, np.zeros((og_row.shape[0], W-og_row.shape[1], 3), dtype=np.uint8)])
    elif og_row.shape[1] > W:
        og_row = og_row[:, :W]
    sections.append(og_row)
    sections.append(sep.copy())

    # Origin summary table
    tbl_h = 35 + 45 * len(all_results)
    tbl = np.zeros((tbl_h, W, 3), dtype=np.uint8)
    cv2.putText(tbl, "ORIGIN SUMMARY (VLM)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 2, cv2.LINE_AA)
    for i, (d, verdicts, origin, _) in enumerate(all_results):
        y = 55 + i * 45
        ocol = C_RED if "UNKNOWN" not in origin and "DVI" not in origin else (C_GREEN if "DVI" in origin else C_ORANGE)
        text = f"{d.dr_sub_item}  ctr=({d.box_ctr_x:.0f},{d.box_ctr_y:.0f})  -->  ORIGIN: {origin}"
        cv2.putText(tbl, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ocol, 1, cv2.LINE_AA)
        chain = "  |  ".join(f"{v.filename}:{v.status}({v.confidence:.0%})" for v in verdicts)
        cv2.putText(tbl, chain, (20, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        # VLM reasoning snippet for the origin image
        origin_v = next((v for v in verdicts if v.filename == origin), None)
        if origin_v and origin_v.metrics.get("vlm_reasoning"):
            reason_text = f"VLM: {origin_v.metrics['vlm_reasoning'][:120]}"
            cv2.putText(tbl, reason_text, (20, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 180, 120), 1, cv2.LINE_AA)
    sections.append(tbl)
    sections.append(sep.copy())

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
    spath = os.path.join(outdir, "SUMMARY_REPORT_VLM.jpg")
    cv2.imwrite(spath, summary, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Summary: {spath}")

    # ---------- 8. Origin text report ----------
    rpath = os.path.join(outdir, "ORIGIN_REPORT_VLM.txt")
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("DEFECT ORIGIN TRACEBACK REPORT (VLM-Assisted)\n")
        f.write("=" * 70 + "\n")
        f.write(f"LOT:       {lot}\n")
        f.write(f"VISUAL_ID: {vid}\n")
        f.write(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Reference: {ref_key}\n")
        f.write(f"VLM:       {provider} ({vlm.__class__.__name__})\n")
        f.write(f"Process images ({len(proc_sorted)}): {', '.join(proc_sorted)}\n")
        f.write("\n")

        for d, verdicts, origin, _ in all_results:
            f.write("-" * 70 + "\n")
            f.write(f"DEFECT: {d.dr_sub_item}\n")
            f.write(f"  Location:  ctr=({d.box_ctr_x:.0f}, {d.box_ctr_y:.0f})  "
                    f"size=({d.box_side_x:.0f} x {d.box_side_y:.0f})\n")
            f.write(f"  OG Image:  {d.og_frame}\n")
            f.write(f"  ORIGIN:    {origin}\n")
            f.write(f"\n  Process-by-process VLM verdicts (newest -> oldest):\n")
            for v in verdicts:
                f.write(f"    {v.filename:20s}  {v.status:15s}  conf={v.confidence:.0%}  {v.detail[:100]}\n")
                if v.metrics:
                    reasoning = v.metrics.get("vlm_reasoning", "")
                    if reasoning:
                        words = reasoning.split()
                        line = ""
                        for word in words:
                            if len(line) + len(word) + 1 > 80:
                                f.write(f"{'':26s}  {line}\n")
                                line = word
                            else:
                                line = f"{line} {word}" if line else word
                        if line:
                            f.write(f"{'':26s}  {line}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("LEGEND:\n")
        f.write("  PRESENT      - VLM detected defect pattern in process image\n")
        f.write("  ABSENT       - VLM determined zone is clean / no defect\n")
        f.write("  INCONCLUSIVE - VLM could not determine with confidence\n")
        f.write("  OUT_OF_VIEW  - Defect zone outside process image field of view\n")
        f.write("  ALIGN_FAIL   - Could not align process image to reference\n")
        f.write("  VLM_ERROR    - VLM service returned an error\n")
        f.write("\n  ORIGIN = earliest process image where defect is determined PRESENT.\n")
        f.write("  If all ABSENT -> defect introduced at DVI/final inspection.\n")
        f.write("=" * 70 + "\n")

    print(f"  Origin report: {rpath}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return outdir


if __name__ == "__main__":
    main()
