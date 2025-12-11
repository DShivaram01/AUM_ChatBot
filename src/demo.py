#!/usr/bin/env python
"""
demo.py

One-shot launcher for your AUM_ChatBot project.

Usage in Colab:

    !git clone https://github.com/<you>/AUM_ChatBot.git
    %cd AUM_ChatBot
    !python demo.py

This script will:
  1. Install requirements from requirements.txt or src/requirements.txt
  2. Launch the main Gradio app (src/main_app.py or src/main_aoo.py) via `python -m`
"""

import sys
import subprocess
from pathlib import Path


def run(cmd):
    """Run a subprocess command, printing it first."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}: {' '.join(cmd)}")
        raise SystemExit(result.returncode)


def main():
    root = Path(__file__).resolve().parent
    print(f"📂 Repo root: {root}")

    # ----------------------------------------------------
    # 1) Find requirements.txt (prefer root, then src/)
    # ----------------------------------------------------
    req_candidates = [
        root / "requirements.txt",
        root / "src" / "requirements.txt",
    ]
    req_path = next((p for p in req_candidates if p.exists()), None)

    if req_path is None:
        print("⚠️ No requirements.txt found in root or src/. Skipping pip install.")
    else:
        print(f"📦 Using requirements file: {req_path}")
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        run([sys.executable, "-m", "pip", "install", "-r", str(req_path)])

    # ----------------------------------------------------
    # 2) Decide which main module to run
    #    Try src.main_app first, then src.main_aoo
    # ----------------------------------------------------
    # We will *run* it via `python -m`, but we use importlib
    # first just to detect which one actually exists.
    import importlib

    main_modules = ["src.main_app", "src.main_aoo"]
    chosen_module = None

    for mod_name in main_modules:
        try:
            importlib.import_module(mod_name)
            chosen_module = mod_name
            print(f"✅ Found main module: {mod_name}")
            break
        except ModuleNotFoundError:
            print(f"ℹ️ Module not found: {mod_name} (trying next...)")

    if chosen_module is None:
        print("❌ Could not find src.main_app or src.main_aoo.")
        print("   Please make sure one of these exists and is importable.")
        raise SystemExit(1)

    # ----------------------------------------------------
    # 3) Launch the main app via `python -m`
    # ----------------------------------------------------
    print(f"🚀 Launching Gradio app from {chosen_module} ...")
    # This will block and run until you stop the Gradio server.
    run([sys.executable, "-m", chosen_module])


if __name__ == "__main__":
    main()
