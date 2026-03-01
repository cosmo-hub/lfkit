"""Download Poggianti (1997) correction tables from VizieR.

This script retrieves the Poggianti (1997) catalog
(J/A+AS/122/399) using astroquery and stores the tables as CSV
files inside the LFKit package data directory:

    src/lfkit/data/poggianti1997/

The downloaded files are written in CSV format so they can be
accessed at runtime via ``importlib.resources``.

To run this script, simply execute it from the command line:

    python scripts/download_poggianti97_data.py

This script is intended to be run manually during development
or data refresh.

Paper: https://arxiv.org/abs/astro-ph/9608029.
Catalog: https://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+AS/122/399.
"""

from __future__ import annotations

from pathlib import Path

from astroquery.vizier import Vizier

CATALOG = "J/A+AS/122/399"

# Package data directory where CSVs are stored
PKG_DATADIR = Path("src/lfkit/data/poggianti1997")


def main() -> None:
    """Fetch the Poggianti (1997) catalog and save tables as CSV files.

    The VizieR catalog may contain multiple tables (e.g., k-corrections,
    e-corrections). Each table is written to the LFKit package data
    directory using its short table name as the filename.

    The output directory is created if it does not already exist,
    and minimal ``__init__.py`` files are ensured so the directory
    is importable as a package resource.
    """
    Vizier.ROW_LIMIT = -1  # retrieve all rows

    # Ensure package data directory exists and is importable
    PKG_DATADIR.mkdir(parents=True, exist_ok=True)
    (PKG_DATADIR.parent / "__init__.py").touch(exist_ok=True)
    (PKG_DATADIR / "__init__.py").touch(exist_ok=True)

    tables = Vizier.get_catalogs(CATALOG)
    print(f"Fetched {len(tables)} table(s) from {CATALOG}")

    saved = []
    for key in tables.keys():
        table = tables[key]
        short_name = key.split("/")[-1]
        path = PKG_DATADIR / f"{short_name}.csv"
        table.write(path, format="csv", overwrite=True)
        saved.append(path.name)
        print(f"Saved {path} ({len(table)} rows, {len(table.colnames)} columns)")

    print("\nDone. Wrote:", ", ".join(sorted(saved)))
    print(f"Package data directory: {PKG_DATADIR.resolve()}")


if __name__ == "__main__":
    main()
