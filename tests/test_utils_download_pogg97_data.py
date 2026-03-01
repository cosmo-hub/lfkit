"""Tests for the `lfkit.utils.download_poggianti97_data` module."""

from __future__ import annotations

from pathlib import Path

import lfkit.utils.download_poggianti97_data as dl


class DummyTable:
    """Dummy class for testing."""
    def __init__(self, nrows=3, ncols=2):
        """Initialize the table."""
        self._nrows = nrows
        self.colnames = [f"col{i}" for i in range(ncols)]

    def __len__(self):
        """Return the number of rows."""
        return self._nrows

    def write(self, path, format="csv", overwrite=True):
        """Write the table to a CSV file."""
        Path(path).write_text("dummy,data\n1,2\n")


def test_main_creates_package_directory(tmp_path, monkeypatch):
    """Tests that main creates the package data directory and __init__.py files."""
    monkeypatch.setattr(dl, "PKG_DATADIR", tmp_path / "poggianti1997")

    dummy_tables = {"J/A+AS/122/399/table1": DummyTable()}

    monkeypatch.setattr(
        dl.Vizier,
        "get_catalogs",
        lambda catalog: dummy_tables,
    )

    dl.main()

    assert (tmp_path / "poggianti1997").exists()
    assert (tmp_path / "poggianti1997" / "__init__.py").exists()
    assert (tmp_path / "__init__.py").exists()


def test_main_writes_csv_files(tmp_path, monkeypatch):
    """Tests that main writes one CSV per returned VizieR table."""
    monkeypatch.setattr(dl, "PKG_DATADIR", tmp_path / "poggianti1997")

    dummy_tables = {
        "J/A+AS/122/399/table1": DummyTable(),
        "J/A+AS/122/399/table2": DummyTable(),
    }

    monkeypatch.setattr(
        dl.Vizier,
        "get_catalogs",
        lambda catalog: dummy_tables,
    )

    dl.main()

    written = list((tmp_path / "poggianti1997").glob("*.csv"))
    assert len(written) == 2


def test_main_uses_correct_catalog(monkeypatch):
    """Tests that main calls Vizier.get_catalogs with the expected catalog identifier."""
    called = {}

    def fake_get_catalogs(catalog):
        called["catalog"] = catalog
        return {}

    monkeypatch.setattr(dl.Vizier, "get_catalogs", fake_get_catalogs)
    monkeypatch.setattr(dl, "PKG_DATADIR", Path("unused"))

    dl.main()

    assert called["catalog"] == dl.CATALOG
