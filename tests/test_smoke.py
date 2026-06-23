"""Smoke test: the package and its subpackages import cleanly."""

import importlib

import pals


def test_version_is_exposed():
    assert pals.__version__ == "0.1.0"


def test_subpackages_import():
    for name in ("core", "oracles", "shielding", "envs", "bench"):
        importlib.import_module(f"pals.{name}")
