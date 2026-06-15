"""Sync release metadata into CITATION.cff and README.md."""

from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
README = ROOT / "README.md"


def get_version() -> str:
    """Return version from setuptools-scm."""
    result = subprocess.run(
        ["python", "-m", "setuptools_scm"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().removeprefix("v")


def main() -> None:
    """Sync release metadata files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=get_version())
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    version = args.version.removeprefix("v")

    citation_text = CITATION.read_text()
    citation_text = re.sub(
        r'^version:\s*".*"$',
        f'version: "{version}"',
        citation_text,
        flags=re.MULTILINE,
    )
    citation_text = re.sub(
        r"^date-released:\s*.*$",
        f"date-released: {args.date}",
        citation_text,
        flags=re.MULTILINE,
    )
    CITATION.write_text(citation_text)

    readme_text = README.read_text()
    readme_text = re.sub(
        r"(version\s*=\s*\{)[^}]+(\})",
        rf"\g<1>{version}\2",
        readme_text,
        count=1,
    )
    README.write_text(readme_text)

    print(f"Synced CITATION.cff and README.md to {version} dated {args.date}")


if __name__ == "__main__":
    main()
