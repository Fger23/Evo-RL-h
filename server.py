"""Foreground model server. Use your own terminal or tmux."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lerobot.scripts.hotswitch_server import main

if __name__ == "__main__":
    main()
