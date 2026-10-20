"""Tests for the names the moderndid namespace exports."""

import moderndid


def test_every_optional_import_is_listed_in_all():
    missing = sorted(set(moderndid._optional_imports) - set(moderndid.__all__))

    assert missing == []
    assert set(moderndid._optional_imports) <= set(dir(moderndid))
