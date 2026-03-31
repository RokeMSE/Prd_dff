"""
Stain Detective — Streamlit UI
================================
Integrates the VLM-assisted defect traceback backend into a web UI.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import os
import re
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image

# Backend imports (same directory)
from defect_traceback_vlm import run_traceback, DefectBox, parse_csv

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Stain Detective",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(90deg, #0071C5, #0096FF);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    /* Verdict badges */
    .badge-present  { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-absent   { background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-inconc   { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-error    { background:#f3f4f6; color:#6b7280; padding:2px 8px; border-radius:4px; font-weight:600; }
    .origin-tag     { background:#0071C5; color:white; padding:4px 12px; border-radius:6px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Helpers
# ============================================================
DATA_ROOT_DEFAULT = "."  # relative to Product Diff/

OG_PAT = re.compile(r'X\w+_\d+_\d+_.*FRAME\d+.*\.(jpg|jpeg|png)', re.IGNORECASE)
PROC_PAT = re.compile(r'(?:\w+_)?\d+_(In|Out)\.(jpg|jpeg|png)', re.IGNORECASE)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_date(s):
    for f in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y",
              "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), f)
        except Exception:
            pass
    return datetime.min


def safe_vid(s):
    return "".join(c for c in s if c.isalnum() or c in "_-")


def map_module(t):
    t = t.upper().strip()
    if t.startswith("HBI"):
        return "HDBI"
    if t.startswith("HXV"):
        return "HDMx"
    if t.startswith("DOB"):
        return "OLB"
    return "Module"


def read_vid_csv(path, vid):
    """Read vid_data.csv and return rows matching *vid*."""
    rows = []
    with open(path, encoding="utf-8") as f:
        sample = f.read(512)
        delim = ";" if ";" in sample.split("\n")[0] else ","
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            if r["VISUAL_ID"].strip().lower() == vid.lower():
                tester = r["TESTER_ID"].strip()
                rows.append({
                    "OPERATION": r["OPERATION"].strip(),
                    "MODULE": map_module(tester),
                    "LOT": r["LOT"].strip(),
                    "TESTER": tester,
                    "END": r["TEST_END_DATE"].strip(),
                    "KEY": parse_date(r["TEST_END_DATE"].strip()),
                })
    rows.sort(key=lambda r: r["KEY"])
    return rows


def discover_data_dir(data_root, vid):
    """Find the directory for a Visual ID.

    Searches in:
      1. {data_root}/{vid}/
      2. {data_root}/Frontend/output/{vid}/
      3. {data_root}/output/{vid}/
    Returns the first that exists and contains images, or None.
    """
    candidates = [
        os.path.join(data_root, vid),
        os.path.join(data_root, "Frontend", "output", vid),
        os.path.join(data_root, "output", vid),
    ]
    for d in candidates:
        if os.path.isdir(d):
            has_images = any(
                f.lower().endswith(tuple(IMG_EXTS))
                for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
            )
            if has_images:
                return d
    return None


def classify_images(directory):
    """Classify files into OG images and process images."""
    og, proc, other = {}, {}, {}
    for f in sorted(os.listdir(directory)):
        fp = os.path.join(directory, f)
        if not os.path.isfile(fp):
            continue
        if not f.lower().endswith(tuple(IMG_EXTS)):
            continue
        if OG_PAT.match(f):
            og[f] = fp
        elif PROC_PAT.match(f):
            proc[f] = fp
        else:
            other[f] = fp
    return og, proc, other


def cv2_to_pil(img):
    """BGR numpy -> RGB PIL."""
    if img.ndim == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def verdict_badge(status):
    cls = {
        "PRESENT": "badge-present", "ABSENT": "badge-absent",
        "INCONCLUSIVE": "badge-inconc",
    }.get(status, "badge-error")
    return f'<span class="{cls}">{status}</span>'


# ============================================================
# Session-state defaults
# ============================================================
_defaults = {
    "vid": "",
    "data_dir": None,
    "vid_rows": None,
    "dvi_rows": None,
    "og_images": {},
    "proc_images": {},
    "other_images": {},
    "traceback_results": None,
    "manual_boxes": [],
    "canvas_key": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    if os.path.exists("Frontend/assets/logo.png"):
        st.image("Frontend/assets/logo.png", width=48)
    st.markdown("### Stain Detective")
    st.caption("VLM-Assisted Defect Origin Traceback")

    st.divider()
    data_root = st.text_input("Data root directory", value=DATA_ROOT_DEFAULT,
                              help="Base directory containing Visual-ID folders")

    vid_input = st.text_input("Visual ID", placeholder="e.g. U65E35A201073",
                              value=st.session_state.vid)
    search_clicked = st.button("Search", type="primary", use_container_width=True)

    st.divider()
    mode = st.radio("Traceback mode", ["Auto (DVI CSV)", "Manual (Draw boxes)"],
                    help="**Auto**: uses DVI_box_data.csv bounding boxes.\n\n"
                         "**Manual**: draw defect boxes on the OG image.")

    st.divider()
    run_clicked = st.button("Run Traceback", type="primary",
                            use_container_width=True,
                            disabled=st.session_state.data_dir is None)


# ============================================================
# Search logic
# ============================================================
if search_clicked and vid_input.strip():
    vid = safe_vid(vid_input.strip())
    st.session_state.vid = vid
    st.session_state.traceback_results = None
    st.session_state.manual_boxes = []
    st.session_state.canvas_key += 1

    data_dir = discover_data_dir(data_root, vid)
    if data_dir is None:
        st.error(f"No data directory found for **{vid}** under `{data_root}`")
        st.session_state.data_dir = None
        st.stop()

    st.session_state.data_dir = data_dir

    # Read vid_data.csv if present
    vid_csv = os.path.join(data_dir, "vid_data.csv")
    if os.path.isfile(vid_csv):
        st.session_state.vid_rows = read_vid_csv(vid_csv, vid)
    else:
        st.session_state.vid_rows = None

    # Read DVI_box_data.csv if present
    dvi_csv = os.path.join(data_dir, "DVI_box_data.csv")
    if os.path.isfile(dvi_csv):
        df = pd.read_csv(dvi_csv)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=["IMAGE_FULL_PATH"], errors="ignore")
        st.session_state.dvi_rows = df
    else:
        st.session_state.dvi_rows = None

    # Classify images
    og, proc, other = classify_images(data_dir)
    st.session_state.og_images = og
    st.session_state.proc_images = proc
    st.session_state.other_images = other

    st.rerun()


# ============================================================
# Main content
# ============================================================
st.markdown('<div class="main-header">Stain Detective</div>', unsafe_allow_html=True)

if st.session_state.data_dir is None:
    st.info("Enter a **Visual ID** in the sidebar and click **Search** to begin.")
    st.stop()

vid = st.session_state.vid
data_dir = st.session_state.data_dir
st.caption(f"Visual ID: **{vid}**  |  Data: `{data_dir}`")

# ---------- Process info table ----------
if st.session_state.vid_rows:
    with st.expander("Process Information", expanded=True):
        df_proc = pd.DataFrame(st.session_state.vid_rows)
        df_proc = df_proc.drop(columns=["KEY"], errors="ignore")
        st.dataframe(df_proc, use_container_width=True, hide_index=True)

# ---------- DVI records ----------
if st.session_state.dvi_rows is not None:
    with st.expander("DVI Box Records (Auto mode data)", expanded=False):
        st.dataframe(st.session_state.dvi_rows, use_container_width=True, hide_index=True)
else:
    if "Auto" in mode:
        st.warning("No `DVI_box_data.csv` found — Auto mode unavailable. Switch to **Manual**.")

# ---------- Image gallery ----------
og_imgs = st.session_state.og_images
proc_imgs = st.session_state.proc_images
all_imgs = {**og_imgs, **proc_imgs, **st.session_state.other_images}

if all_imgs:
    with st.expander("Image Gallery", expanded=True):
        # OG images first, then process
        ordered = list(og_imgs.keys()) + sorted(proc_imgs.keys()) + list(st.session_state.other_images.keys())
        cols = st.columns(min(len(ordered), 5))
        for idx, fname in enumerate(ordered):
            fp = all_imgs[fname]
            col = cols[idx % len(cols)]
            with col:
                img = Image.open(fp)
                tag = "OG" if fname in og_imgs else ("Process" if fname in proc_imgs else "Other")
                st.image(img, caption=f"{tag}: {fname}", use_container_width=True)

# ---------- Manual mode: canvas ----------
if "Manual" in mode:
    st.subheader("Manual Defect Box Drawing")

    if not HAS_CANVAS:
        st.error(
            "**streamlit-drawable-canvas** is not installed.\n\n"
            "Run: `pip install streamlit-drawable-canvas`\n\n"
            "Falling back to coordinate input below."
        )

    # Pick the OG image for drawing (prefer latest FRAME number)
    if og_imgs:
        og_keys = sorted(og_imgs.keys(), reverse=True)  # highest frame first
        selected_og = st.selectbox("Reference image for drawing", og_keys)
        ref_path = og_imgs[selected_og]
    elif proc_imgs:
        st.info("No OG (FRAME) images found. Selecting latest process image as reference.")
        proc_keys = sorted(proc_imgs.keys(), reverse=True)
        selected_og = st.selectbox("Reference image for drawing", proc_keys)
        ref_path = proc_imgs[selected_og]
    else:
        st.error("No images available for drawing.")
        st.stop()

    ref_pil = Image.open(ref_path)
    orig_w, orig_h = ref_pil.size
    CANVAS_W = 700
    scale = CANVAS_W / orig_w
    canvas_h = int(orig_h * scale)

    if HAS_CANVAS:
        st.caption(
            "Draw rectangles on the image to mark defect regions. "
            "Each rectangle becomes a defect box for traceback."
        )

        col_canvas, col_info = st.columns([3, 1])
        with col_canvas:
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.15)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=ref_pil,
                drawing_mode="rect",
                height=canvas_h,
                width=CANVAS_W,
                key=f"canvas_{st.session_state.canvas_key}",
            )

        # Parse drawn rectangles
        boxes = []
        if canvas_result and canvas_result.json_data:
            for obj in canvas_result.json_data.get("objects", []):
                if obj.get("type") != "rect":
                    continue
                # Canvas coords -> image pixel coords
                left = obj["left"] / scale
                top = obj["top"] / scale
                w = obj["width"] * obj.get("scaleX", 1) / scale
                h = obj["height"] * obj.get("scaleY", 1) / scale
                if w < 5 or h < 5:
                    continue
                boxes.append({
                    "cx": left + w / 2,
                    "cy": top + h / 2,
                    "w": w,
                    "h": h,
                })

        with col_info:
            st.markdown(f"**Boxes drawn:** {len(boxes)}")
            for i, b in enumerate(boxes):
                st.caption(
                    f"DEFECT\\_{i+1}: "
                    f"ctr=({b['cx']:.0f}, {b['cy']:.0f}) "
                    f"size=({b['w']:.0f}x{b['h']:.0f})"
                )
            if st.button("Clear canvas"):
                st.session_state.canvas_key += 1
                st.rerun()

        st.session_state.manual_boxes = boxes

    else:
        # Fallback: manual coordinate entry
        st.image(ref_pil, caption=f"Reference: {selected_og}", use_container_width=True)
        st.markdown("Enter defect box coordinates manually (in image pixels):")
        num_boxes = st.number_input("Number of defect boxes", min_value=1, max_value=20, value=1)
        boxes = []
        for i in range(int(num_boxes)):
            st.markdown(f"**Defect {i+1}**")
            c1, c2, c3, c4 = st.columns(4)
            cx = c1.number_input(f"Center X", key=f"cx_{i}", min_value=0, max_value=orig_w, value=orig_w // 2)
            cy = c2.number_input(f"Center Y", key=f"cy_{i}", min_value=0, max_value=orig_h, value=orig_h // 2)
            bw = c3.number_input(f"Width", key=f"bw_{i}", min_value=10, max_value=orig_w, value=200)
            bh = c4.number_input(f"Height", key=f"bh_{i}", min_value=10, max_value=orig_h, value=200)
            boxes.append({"cx": cx, "cy": cy, "w": bw, "h": bh})
        st.session_state.manual_boxes = boxes

# ============================================================
# Run traceback
# ============================================================
if run_clicked:
    data_dir = st.session_state.data_dir
    outdir = os.path.join(data_dir, "output")

    # Build defect boxes based on mode
    if "Auto" in mode:
        dvi_csv = os.path.join(data_dir, "DVI_box_data.csv")
        if not os.path.isfile(dvi_csv):
            st.error("DVI_box_data.csv not found. Switch to **Manual** mode.")
            st.stop()
        defect_boxes = None  # run_traceback will parse CSV
    else:
        boxes = st.session_state.manual_boxes
        if not boxes:
            st.error("No defect boxes drawn. Please draw at least one box on the image.")
            st.stop()

        # Determine lot/vid from DVI CSV or fallback
        lot = vid
        if st.session_state.vid_rows:
            lot = st.session_state.vid_rows[0].get("LOT", vid)

        # Which image was used as reference for drawing
        if og_imgs:
            ref_og_name = sorted(og_imgs.keys(), reverse=True)[0]
            ref_og_path = og_imgs[ref_og_name]
        elif proc_imgs:
            ref_og_name = sorted(proc_imgs.keys(), reverse=True)[0]
            ref_og_path = proc_imgs[ref_og_name]
        else:
            st.error("No reference image.")
            st.stop()

        defect_boxes = []
        for i, b in enumerate(boxes):
            defect_boxes.append(DefectBox(
                lot=lot,
                visual_id=vid,
                dr_result="MANUAL",
                dr_sub_item=f"DEFECT_{i+1}",
                box_ctr_x=b["cx"],
                box_ctr_y=b["cy"],
                box_side_x=b["w"],
                box_side_y=b["h"],
                image_path=ref_og_path,
                coord_space="PIXEL",
            ))

    # Run with progress
    progress_bar = st.progress(0, text="Starting traceback...")
    status_text = st.empty()

    def on_progress(step, total, msg):
        progress_bar.progress(step / total, text=msg)
        status_text.caption(msg)

    with st.spinner("Running VLM-assisted defect traceback..."):
        results = run_traceback(
            uploads_dir=data_dir,
            outdir=outdir,
            defect_boxes=defect_boxes,
            progress_callback=on_progress,
        )

    progress_bar.progress(1.0, text="Complete!")

    if "error" in results:
        st.error(f"Traceback failed: {results['error']}")
    else:
        st.session_state.traceback_results = results
        st.rerun()


# ============================================================
# Display results
# ============================================================
if st.session_state.traceback_results:
    results = st.session_state.traceback_results
    st.divider()
    st.subheader("Traceback Results")

    # ---------- Origin summary ----------
    st.markdown("#### Origin Summary")
    for defect_name, origin in results.get("origin_summary", []):
        st.markdown(
            f'**{defect_name}** &rarr; <span class="origin-tag">{origin}</span>',
            unsafe_allow_html=True,
        )

    # ---------- Per-defect verdicts ----------
    st.markdown("#### Per-Image Verdicts")
    for d, verdicts, origin, panel in results.get("all_results", []):
        with st.expander(f"{d.dr_sub_item} — Origin: {origin}", expanded=True):
            cols_v = st.columns(min(len(verdicts), 4)) if verdicts else []
            for idx, v in enumerate(verdicts):
                col = cols_v[idx % len(cols_v)]
                with col:
                    st.markdown(
                        f"**{v.filename}**<br>"
                        f"{verdict_badge(v.status)} {v.confidence:.0%}",
                        unsafe_allow_html=True,
                    )
                    if v.metrics and v.metrics.get("vlm_reasoning"):
                        st.caption(v.metrics["vlm_reasoning"][:200])

            # Show panel image
            if panel is not None:
                panel_pil = cv2_to_pil(panel)
                st.image(panel_pil, caption=f"Traceback panel: {d.dr_sub_item}",
                         use_container_width=True)

    # ---------- Output images ----------
    output_images = results.get("output_images", [])
    if output_images:
        with st.expander("All Output Images", expanded=False):
            img_cols = st.columns(3)
            for idx, img_path in enumerate(output_images):
                if not os.path.isfile(img_path):
                    continue
                col = img_cols[idx % 3]
                with col:
                    st.image(Image.open(img_path),
                             caption=os.path.basename(img_path),
                             use_container_width=True)

    # ---------- Report download ----------
    report_text = results.get("report_text", "")
    if report_text:
        st.markdown("#### Report")
        with st.expander("View full report", expanded=False):
            st.code(report_text, language=None)
        st.download_button(
            label="Download Report (.txt)",
            data=report_text,
            file_name=f"TRACEBACK_REPORT_{vid}_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain",
        )

    # Offer to clear results
    if st.button("Clear results"):
        st.session_state.traceback_results = None
        st.rerun()
