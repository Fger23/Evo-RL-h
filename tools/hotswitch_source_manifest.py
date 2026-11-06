"""Record exact deployed source when distributing an uncommitted working tree."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    files = sorted([*root.joinpath("src").rglob("*.py"), *root.joinpath("src").rglob("*.proto"),
                    *root.joinpath("tools").glob("*.py"), *root.joinpath("tests/hotswitch").glob("*.py")])
    contents = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    result = {"repository": "https://github.com/Fger23/Evo-RL", "source_files_sha256": contents}
    for name, args in (("head", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        result[name] = subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    result["source_tree_sha256"] = hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()
    root.joinpath("source_provenance.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
