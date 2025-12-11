#!/usr/bin/env python
"""
demo.py  (lives in: AUM_ChatBot/src)

One-shot launcher for your AUM_ChatBot project.

Usage in Colab:

    !git clone https://github.com/DShivaram01/AUM_ChatBot.git
    %cd AUM_ChatBot/src
    !python demo.py

This script will:
  1. Install requirements from src/requirements.txt
  2. Launch the main Gradio app (main_app.py or main_aoo.py) via `python -m`
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
    # We are inside src/
    root = Path(__file__).resolve().parent
    print(f"📂 Src root: {root}")

    # ----------------------------------------------------
    # 1) Install requirements from requirements.txt in src/
    # ----------------------------------------------------
    req_path = root / "requirements.txt"
    if not req_path.exists():
        print("⚠️ requirements.txt not found in src/. Skipping pip install.")
    else:
        print(f"📦 Using requirements file: {req_path}")
        run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        run([sys.executable, "-m", "pip", "install", "-r", str(req_path)])

    # ----------------------------------------------------
    # 2) Decide which main module to run
    #    Try main_app first, then main_aoo
    # ----------------------------------------------------
    import importlib

    mod_name = "main_app"
    try:
        importlib.import_module(mod_name)
        chosen_module = mod_name
        print(f"✅ Found main module: {mod_name}")
        
    except ModuleNotFoundError:
        print(f"ℹ️ Module not found: {mod_name} (trying next...)")

    if chosen_module is None:
        print("❌ Could not find main_app.py or main_aoo.py in src/.")
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
