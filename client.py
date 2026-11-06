"""SO101 bimanual hardware client; configuration is checked before driver import."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lerobot.scripts.hotswitch_so101_client import main

if __name__ == "__main__":
    main()
