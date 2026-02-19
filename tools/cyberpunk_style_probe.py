#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys


def _enabled(force: bool, disable: bool) -> bool:
    if disable:
        return False
    if force:
        return True
    if str(os.getenv("NO_COLOR") or "").strip():
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


class C:
    def __init__(self, on: bool) -> None:
        self.on = on
        self.reset = "\033[0m"
        self.head = "\033[38;5;177m"
        self.label = "\033[38;5;111m"
        self.value = "\033[38;5;150m"
        self.acct = "\033[38;5;117m"
        self.static = "\033[38;5;81m"
        self.dynamic = "\033[38;5;183m"
        self.sep = "\033[97m"
        self.dim = "\033[90m"
        self.warn = "\033[38;5;214m"
        self.bad = "\033[38;5;203m"

    def c(self, text: str, code: str) -> str:
        if not self.on:
            return text
        return f"{code}{text}{self.reset}"


def _kv(cs: C, k: str, v: str, v_code: str) -> str:
    return cs.c(k, cs.label) + cs.c("=", cs.sep) + cs.c(v, v_code)


def _triplet(cs: C, a: str, s: str, d: str) -> str:
    return (
        cs.c(a, cs.acct)
        + cs.c("/", cs.sep)
        + cs.c(s, cs.static)
        + cs.c("/", cs.sep)
        + cs.c(d, cs.dynamic)
    )


def _semi_join(cs: C, parts: list[str]) -> str:
    return cs.c(" ; ", cs.sep).join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-color", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    cs = C(_enabled(force=bool(args.force_color), disable=bool(args.no_color)))
    print(cs.c("# Cyberpunk Style Probe", cs.head))
    print(cs.c("legend ", cs.head) + _triplet(cs, "acct", "static", "dyn"))
    print("")

    print(cs.c("[A] x=y contrast + semicolon separators", cs.head))
    row_a = _semi_join(
        cs,
        [
            _kv(cs, "process", "qpec", cs.value),
            _kv(cs, "score", "9.95", cs.value),
            _kv(cs, "scope", "close_specific", cs.value),
        ],
    )
    print("  " + row_a)
    print("  " + cs.c("delta_h ", cs.label) + _triplet(cs, "17.35", "17.35", "17.35"))
    print("  " + cs.c("eff_idx ", cs.label) + _triplet(cs, "0.001442", "0.002896", "0.001860"))
    print("")

    print(cs.c("[B] low-signal row", cs.head))
    row_b = _semi_join(
        cs,
        [
            _kv(cs, "process", "chkregupdt", cs.value),
            _kv(cs, "score", "0.30", cs.warn),
            _kv(cs, "status", "deprioritize", cs.dim),
        ],
    )
    print("  " + row_b)
    print("  " + cs.c("delta_h ", cs.label) + _triplet(cs, "0.01", "0.01", "0.01"))
    print("  " + cs.c("eff_idx ", cs.label) + _triplet(cs, "0.000001", "0.000001", "0.000001"))
    print("")

    print(cs.c("[C] N/A and warning state", cs.head))
    row_c = _semi_join(
        cs,
        [
            _kv(cs, "process", "close_window_model", cs.value),
            _kv(cs, "score", "N/A", cs.warn),
            _kv(cs, "reason", "NO_DYNAMIC_WINDOW", cs.dim),
        ],
    )
    print("  " + row_c)
    print(
        "  "
        + cs.c("delta_h ", cs.label)
        + _triplet(cs, "N/A(NO_DYN)", "0.12", "N/A(NO_MONTH)")
    )
    print(
        "  "
        + cs.c("eff_idx ", cs.label)
        + _triplet(cs, "N/A(NO_DYN)", "0.000010", "N/A(NO_MONTH)")
    )
    print("")

    print(cs.c("[D] negative/regression case", cs.head))
    row_d = _semi_join(
        cs,
        [
            _kv(cs, "process", "queue_policy_v2", cs.value),
            _kv(cs, "score", "0.42", cs.bad),
            _kv(cs, "flag", "review_required", cs.bad),
        ],
    )
    print("  " + row_d)
    print("  " + cs.c("delta_h ", cs.label) + _triplet(cs, "-0.30", "-0.12", "-0.18"))
    print("  " + cs.c("eff_idx ", cs.label) + _triplet(cs, "-0.000025", "-0.000011", "-0.000016"))
    print("")

    print(cs.c("Probe complete.", cs.head))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
