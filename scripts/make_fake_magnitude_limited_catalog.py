"""Create a small fake magnitude-limited galaxy catalog for examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    """Create a deterministic fake catalog with redshifts and apparent magnitudes."""
    rng = np.random.default_rng(42)

    n_gal = 200
    z = np.sort(rng.uniform(0.01, 1.2, n_gal))

    # Fake magnitude-redshift trend plus scatter.
    m_app = 15.0 + 6.0 * np.log10(1.0 + 8.0 * z) + rng.normal(0.0, 0.35, n_gal)

    catalog = pd.DataFrame(
        {
            "galaxy_id": [f"fake-{i:05d}" for i in range(n_gal)],
            "ra_deg": rng.uniform(0.0, 360.0, n_gal),
            "dec_deg": rng.uniform(-30.0, 30.0, n_gal),
            "z": z,
            "m_app": m_app,
            "band": "r",
        }
    )

    repo_root = Path(__file__).resolve().parents[1]
    output = repo_root / "src/lfkit/data/demo_catalogs/fake_magnitude_limited_catalog.csv"

    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)


if __name__ == "__main__":
    main()