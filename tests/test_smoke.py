"""Verifie que l'environnement et le paquet repondent."""

from __future__ import annotations


def test_import_cli() -> None:
    from src.cli import app

    assert app is not None


def test_dependances_principales() -> None:
    import numpy
    import pandas
    import sklearn

    assert numpy.__version__
    assert pandas.__version__
    assert sklearn.__version__
