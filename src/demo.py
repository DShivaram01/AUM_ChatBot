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
  2. Launch the main Gradio app from main_app.py via `python -m main_app`
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
    # 2) Launch the main app via `python -m main_app`
    # ----------------------------------------------------
    main_py = root / "main_app.py"
    if not main_py.exists():
        print("❌ main_app.py not found in src/.")
        print("   Make sure AUM_ChatBot/src/main_app.py exists.")
        raise SystemExit(1)

    print("✅ Found main_app.py")
    print("🚀 Launching Gradio app from main_app ...")
    # This will block and run until you stop the Gradio server.
    run([sys.executable, "-m", "main_app"])


if __name__ == "__main__":
    main()
