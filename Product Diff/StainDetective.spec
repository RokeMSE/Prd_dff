# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Stain Detective (PySide6 desktop app).

Build:
    cd "Product Diff"
    pyinstaller StainDetective.spec

Output:  dist/StainDetective/StainDetective.exe  (--one)
"""

import os

block_cipher = None

# All paths are relative to this spec file's directory
SPEC_DIR = os.path.abspath(SPECPATH)
SRC_DIR = os.path.join(SPEC_DIR, "src")
FRONTEND_DIR = os.path.join(SRC_DIR, "Frontend")

a = Analysis(
    [os.path.join(FRONTEND_DIR, "UI", "main.py")],
    pathex=[SRC_DIR],
    binaries=[],
    datas=[
        # Assets (logo, loading gif)
        (os.path.join(FRONTEND_DIR, "assets", "logo.png"), os.path.join("Frontend", "assets")),
        (os.path.join(FRONTEND_DIR, "assets", "logo.svg"), os.path.join("Frontend", "assets")),
        (os.path.join(FRONTEND_DIR, "assets", "loading.gif"), os.path.join("Frontend", "assets")),
        # Backend modules
        (os.path.join(SRC_DIR, "defect_traceback_vlm.py"), "."),
        (os.path.join(SRC_DIR, "alignment_validation.py"), "."),
    ],
    hiddenimports=[
        # PySide6
        "PySide6.QtWidgets",
        "PySide6.QtGui",
        "PySide6.QtCore",
        # VLM providers (imported dynamically)
        "google.genai",
        "google.genai.types",
        "openai",
        "dotenv",
        # Image / numeric
        "cv2",
        "numpy",
        "pandas",
        "PIL",
        "PIL.Image",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude Streamlit and related web-only packages
        "streamlit",
        "streamlit_drawable_canvas",
        "tornado",
        "matplotlib",
        "tkinter",
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StainDetective",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # No console window (GUI app)
    icon=os.path.join(FRONTEND_DIR, "assets", "logo.png"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="StainDetective",
)
