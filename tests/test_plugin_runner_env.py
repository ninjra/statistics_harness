from pathlib import Path

from statistic_harness.core.plugin_runner import _deterministic_env


def test_deterministic_env_includes_repo_and_src_paths() -> None:
    cwd = Path('/workspace/statistics_harness')
    env = _deterministic_env(123, cwd=cwd)
    py_path = env.get('PYTHONPATH', '')
    parts = py_path.split(':') if py_path else []
    assert str(cwd) in parts
    assert str(cwd / 'src') in parts
    assert env['PYTHONHASHSEED'] == '123'
