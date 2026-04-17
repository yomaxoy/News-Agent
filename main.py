"""
This file exists for Railway to auto-detect the FastAPI app.
Railway looks for main.py or app.py in the project root.
"""
import subprocess
import sys

if __name__ == "__main__":
    # Run uvicorn from backend directory
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd="backend")
