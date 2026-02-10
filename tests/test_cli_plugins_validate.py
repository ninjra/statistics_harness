from statistic_harness.cli import cmd_plugins_validate


def test_cli_plugins_validate_profile_basic():
    # Keep this fast: validate a single known-good plugin.
    cmd_plugins_validate("profile_basic")

