from pathlib import Path

from soma_retargeter.utils.newton_asset_utils import as_newton_usd_source


def test_as_newton_usd_source_converts_path_objects_to_strings():
    source = Path("/tmp/robot.usda")

    normalized = as_newton_usd_source(source)

    assert normalized == str(source)
    assert isinstance(normalized, str)


def test_as_newton_usd_source_leaves_strings_unchanged():
    source = "/tmp/robot.usda"

    normalized = as_newton_usd_source(source)

    assert normalized == source
