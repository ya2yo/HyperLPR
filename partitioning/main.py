import argparse
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["server"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    os.environ["HYPERLPR_DEVICE"] = args.device
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Prj-Python"))
    import uvicorn
    uvicorn.run("partitioning.server.api:app", host=args.host, port=args.port)
