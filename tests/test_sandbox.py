import socket
import subprocess

import pytest

from statistic_harness.core.plugin_runner import (
    FileSandbox,
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
    orig_dump = pickle.dump
    orig_dumps = pickle.dumps
    orig_pickler = pickle.Pickler
    orig_unpickler = pickle.Unpickler
    try:
        _install_pickle_guard()
        with pytest.raises(RuntimeError):
            pickle.dumps({"a": 1})
        with pytest.raises(RuntimeError):
            pickle.loads(b"test")
    finally:
        pickle.load = orig_load
        pickle.loads = orig_loads
        pickle.dump = orig_dump
        pickle.dumps = orig_dumps
        pickle.Pickler = orig_pickler
        pickle.Unpickler = orig_unpickler


def test_shell_guard_blocks_subprocess_and_os():
    import os

    orig_system = os.system
    orig_popen = os.popen
    orig_run = subprocess.run
    orig_call = subprocess.call
    orig_check_call = subprocess.check_call
    orig_check_output = subprocess.check_output
    orig_popen_cls = subprocess.Popen
    try:
        _install_shell_guard()
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
