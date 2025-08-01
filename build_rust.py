#!/usr/bin/env python3
import platform
import subprocess
from pathlib import Path
import shutil
import sys


def main():
    repo_root = Path(__file__).resolve().parent
    manifest = repo_root / "spectrogram-rs" / "Cargo.toml"
    try:
        subprocess.run([
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
        ], check=True)
    except FileNotFoundError:
        print(
            "cargo not found. Install Rust from https://www.rust-lang.org/.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_dir = repo_root / "spectrogram-rs" / "target" / "release"
    exe_name = "spectrogram-rs"
    if platform.system() == "Windows":
        exe_name += ".exe"

    built = target_dir / exe_name
    if not built.exists():
        print(f"Expected build output {built} not found", file=sys.stderr)
        sys.exit(1)

    dist = repo_root / "dist"
    dist.mkdir(exist_ok=True)
    shutil.copy2(built, dist / built.name)
    print(f"Built executable: {dist / built.name}")


if __name__ == "__main__":
    main()
