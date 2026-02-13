import socket
import subprocess
import os

import pytest

from statistic_harness.core.plugin_runner import (
    FileSandbox,
    _should_block_eval_for_path,
    _install_sqlite_guard,
    _install_eval_guard,
    _install_network_guard,
    _install_pickle_guard,
    _install_shell_guard,
)


def test_file_sandbox_blocks_disallowed_paths(tmp_path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    allowed_file = allowed_dir / "ok.txt"
    allowed_file.write_text("ok", encoding="utf-8")

    blocked_dir = tmp_path / "blocked"
    blocked_dir.mkdir()
    blocked_file = blocked_dir / "no.txt"
    blocked_file.write_text("no", encoding="utf-8")

    with FileSandbox([str(allowed_dir)], [str(allowed_dir)], cwd=tmp_path):
        assert allowed_file.read_text(encoding="utf-8") == "ok"
        allowed_file.write_text("updated", encoding="utf-8")
        with pytest.raises(PermissionError):
            blocked_file.read_text(encoding="utf-8")
        with pytest.raises(PermissionError):
            blocked_file.write_text("nope", encoding="utf-8")

        import os

        fd = os.open(str(allowed_dir / "os_open.txt"), os.O_CREAT | os.O_WRONLY, 0o600)
        os.write(fd, b"ok")
        os.close(fd)
        with pytest.raises(PermissionError):
            os.open(str(blocked_dir / "nope.txt"), os.O_CREAT | os.O_WRONLY, 0o600)


def test_file_sandbox_allows_fd_open(tmp_path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    rfd, wfd = os.pipe()
    try:
        with FileSandbox([str(allowed_dir)], [str(allowed_dir)], cwd=tmp_path):
            with open(wfd, "wb", closefd=False) as w:
                w.write(b"x")
                w.flush()
            with open(rfd, "rb", closefd=False) as r:
                assert r.read(1) == b"x"
    finally:
        os.close(rfd)
        os.close(wfd)


def test_file_sandbox_accepts_bytes_paths(tmp_path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    target = allowed_dir / "bytes-path.txt"
    path_bytes = os.fsencode(str(target))
    with FileSandbox([str(allowed_dir)], [str(allowed_dir)], cwd=tmp_path):
        fd = os.open(path_bytes, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        os.write(fd, b"ok")
        os.close(fd)
        with open(path_bytes, "rb") as handle:
            assert handle.read() == b"ok"


def test_network_guard_blocks_socket():
    orig_socket = socket.socket
    orig_create = socket.create_connection
    try:
        _install_network_guard()
        with pytest.raises(RuntimeError):
            socket.socket()
    finally:
        socket.socket = orig_socket
        socket.create_connection = orig_create


def test_eval_guard_blocks_eval():
    import builtins
    from pathlib import Path

    orig_eval = builtins.eval
    try:
        root = Path(__file__).resolve().parents[1]
        _install_eval_guard(root)
        with pytest.raises(RuntimeError):
            eval("1 + 1")  # noqa: S307 - intentional for guard test
    finally:
        builtins.eval = orig_eval


def test_eval_guard_allows_site_packages_paths(tmp_path):
    from pathlib import Path

    root = tmp_path / "repo"
    root.mkdir()
    caller = root / ".venv" / "lib" / "python3.12" / "site-packages" / "mod.py"
    caller.parent.mkdir(parents=True)
    caller.write_text("# stub", encoding="utf-8")
    assert _should_block_eval_for_path(caller, root) is False
    assert _should_block_eval_for_path(Path("<frozen importlib._bootstrap>"), root) is False


def test_sqlite_guard_blocks_other_databases(tmp_path):
    import sqlite3

    allowed = tmp_path / "state.sqlite"
    allowed.write_bytes(b"")  # file must exist for connect on some platforms

    other = tmp_path / "other.sqlite"
    other.write_bytes(b"")

    orig_connect = sqlite3.connect
    try:
        _install_sqlite_guard(allowed)
        with pytest.raises(RuntimeError):
            sqlite3.connect(other)
        conn = sqlite3.connect(allowed)
        conn.close()
    finally:
        sqlite3.connect = orig_connect



def test_pickle_guard_blocks_pickling():
    import pickle

    orig_load = pickle.load
    orig_loads = pickle.loads
    try:
        _install_pickle_guard()
        with pytest.raises(RuntimeError):
            pickle.loads(b"test")
        assert pickle.dumps({"a": 1})  # serialization remains available
        assert isinstance(pickle.Pickler, type)
    finally:
        pickle.load = orig_load
        pickle.loads = orig_loads


def test_shell_guard_blocks_subprocess_and_os():
    import os
    from pathlib import Path

    orig_system = os.system
    orig_popen = os.popen
    orig_run = subprocess.run
    orig_call = subprocess.call
    orig_check_call = subprocess.check_call
    orig_check_output = subprocess.check_output
    orig_popen_cls = subprocess.Popen
    try:
        root = Path(__file__).resolve().parents[1]
        _install_shell_guard(root)
        with pytest.raises(RuntimeError):
            os.system("echo blocked")
        with pytest.raises(RuntimeError):
            os.popen("echo blocked")
        with pytest.raises(RuntimeError):
            subprocess.run(["echo", "blocked"])  # noqa: S603 - intentional
    finally:
        os.system = orig_system
        os.popen = orig_popen
        subprocess.run = orig_run
        subprocess.call = orig_call
        subprocess.check_call = orig_check_call
        subprocess.check_output = orig_check_output
        subprocess.Popen = orig_popen_cls


def test_shell_guard_allows_fc_list_probe():
    import subprocess
    from pathlib import Path

    orig_check_output = subprocess.check_output
    try:
        root = Path(__file__).resolve().parents[1]
        _install_shell_guard(root)
        try:
            subprocess.check_output(["fc-list", "--help"])  # noqa: S603 - guard allowlist test
        except FileNotFoundError:
            # Command not available in environment; still proves guard did not block it.
            pass
    finally:
        subprocess.check_output = orig_check_output


def test_shell_guard_allows_site_packages_caller(tmp_path):
    import subprocess

    orig_check_output = subprocess.check_output
    try:
        root = tmp_path / "repo"
        root.mkdir()
        _install_shell_guard(root)
        fake = root / ".venv" / "lib" / "python3.12" / "site-packages" / "fake_mod.py"
        fake.parent.mkdir(parents=True)
        code = compile(
            "import subprocess\nresult = subprocess.check_output(['/bin/echo', 'ok'])",
            str(fake),
            "exec",
        )
        scope: dict[str, object] = {}
        exec(code, scope)  # noqa: S102 - intentional for guard test
        assert scope["result"] == b"ok\n"
    finally:
        subprocess.check_output = orig_check_output


def test_shell_guard_reinstall_is_idempotent_for_trusted_caller(tmp_path):
    import subprocess

    orig_run = subprocess.run
    base_run = getattr(orig_run, "__stat_harness_orig__", orig_run)
    try:
        root = tmp_path / "repo"
        root.mkdir()
        _install_shell_guard(root)
        _install_shell_guard(root)
        assert getattr(subprocess.run, "__stat_harness_orig__", None) is base_run
    finally:
        subprocess.run = orig_run
