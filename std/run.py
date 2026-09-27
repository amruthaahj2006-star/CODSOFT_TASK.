"""
run.py
One-click error-free launcher for Credit Card Fraud Detection Web Application.
Automatically verifies/installs dependencies and starts Streamlit.
"""

import os
import sys
import subprocess

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    app_path = os.path.join(base_dir, "app.py")

    print("=" * 60)
    print("  🛡️ CREDIT CARD FRAUD DETECTION AI - DASHBOARD LAUNCHER")
    print("=" * 60)

    # 1. Ensure dependencies are installed
    packages = {
        "streamlit": "streamlit",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
    }

    for mod, pkg in packages.items():
        try:
            __import__(mod)
        except ImportError:
            print(f"[*] Installing required library '{pkg}'...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "--no-warn-script-location"]
            )

    print("[+] All dependencies verified.")
    print("[+] Starting Streamlit web server...")
    print(f"[+] Running: {sys.executable} -m streamlit run {app_path}\n")

    # 2. Launch Streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n[+] Server stopped.")

if __name__ == "__main__":
    main()
