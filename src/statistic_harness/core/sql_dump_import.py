from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_UNSAFE_PREFIXES = (
    "DROP ",
    "ALTER ",
    "ATTACH ",
    "PRAGMA ",
    "VACUUM ",
    "CREATE VIEW",
    "CREATE TRIGGER",
    "DELETE ",
    "UPDATE ",
)


def _dequote_ident(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        return token
    if token[0] in {'"', "`", "["}:
        if token[0] == "[" and token.endswith("]"):
            return token[1:-1]
        if token.endswith(token[0]):
            return token[1:-1]
    return token


def _split_sql_csv(text: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    in_single = False
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            buf.append(ch)
            if in_single:
                if i + 1 < len(text) and text[i + 1] == "'":
                    buf.append("'")
                    i += 1
                else:
                    in_single = False
            else:
                in_single = True
            i += 1
            continue
        if in_single:
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                items.append(token)
            buf = []
        else:
            buf.append(ch)
        i += 1
    token = "".join(buf).strip()
    if token:
        items.append(token)
    return items


def _parse_scalar(token: str) -> Any:
    t = str(token).strip()
    if not t:
        return None
    up = t.upper()
    if up == "NULL":
        return None
    if t.startswith("'") and t.endswith("'") and len(t) >= 2:
        inner = t[1:-1]
        inner = inner.replace("''", "'")
        inner = inner.replace("\\'", "'")
        return inner
    if re.fullmatch(r"[+-]?\d+", t):
        try:
            return int(t)
        except Exception:
            return t
    if re.fullmatch(r"[+-]?\d+\.\d+", t) or re.fullmatch(r"[+-]?\.\d+", t):
        try:
            return float(t)
        except Exception:
            return t
    return t


def _parse_values_rows(text: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    current: list[str] = []
    token: list[str] = []
    in_single = False
    in_row = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            token.append(ch)
            if in_single:
                if i + 1 < len(text) and text[i + 1] == "'":
                    token.append("'")
                    i += 1
                else:
                    in_single = False
            else:
                in_single = True
            i += 1
            continue
        if in_single:
            token.append(ch)
            i += 1
            continue
        if ch == "(" and not in_row:
            in_row = True
            current = []
            token = []
            i += 1
            continue
        if ch == "," and in_row:
            current.append("".join(token))
            token = []
            i += 1
            continue
        if ch == ")" and in_row:
            current.append("".join(token))
            token = []
            rows.append([_parse_scalar(part) for part in current])
            current = []
            in_row = False
            i += 1
            continue
        token.append(ch)
        i += 1
    return rows


@dataclass(frozen=True)
class ParsedCreateTable:
    table_name: str
    columns: list[str]


@dataclass(frozen=True)
class ParsedInsert:
    table_name: str
    columns: list[str] | None
    rows: list[list[Any]]


def iter_sql_statements(path: Path, *, encoding: str = "utf-8"):
    buf: list[str] = []
    in_single = False
    with path.open("r", encoding=encoding, errors="replace") as handle:
        for line in handle:
            i = 0
            while i < len(line):
                ch = line[i]
                buf.append(ch)
                if ch == "'":
                    if in_single:
                        if i + 1 < len(line) and line[i + 1] == "'":
                            buf.append("'")
                            i += 1
                        else:
                            in_single = False
                    else:
                        in_single = True
                elif ch == ";" and not in_single:
                    stmt = "".join(buf).strip()
                    buf = []
                    if stmt:
                        yield stmt[:-1].strip() if stmt.endswith(";") else stmt
                i += 1
    tail = "".join(buf).strip()
    if tail:
        yield tail


def statement_kind(statement: str) -> str:
    head = str(statement or "").strip().upper()
    if head.startswith("CREATE TABLE"):
        return "create_table"
    if head.startswith("INSERT INTO"):
        return "insert"
    return "other"


def parse_create_table(statement: str) -> ParsedCreateTable:
    m = re.match(
        r"(?is)^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<table>[`\"\[]?[A-Za-z0-9_\.]+[`\"\]]?)\s*\((?P<body>.*)\)\s*$",
        statement.strip(),
    )
    if not m:
        raise ValueError("Unsupported CREATE TABLE statement")
    table_name = _dequote_ident(m.group("table"))
    body = m.group("body")
    cols: list[str] = []
    for part in _split_sql_csv(body):
        chunk = part.strip()
        if not chunk:
            continue
        first = chunk.split()[0].upper()
        if first in {"CONSTRAINT", "PRIMARY", "FOREIGN", "UNIQUE", "CHECK"}:
            continue
        col = _dequote_ident(chunk.split()[0])
        if col:
            cols.append(col)
    if not cols:
        raise ValueError("CREATE TABLE contains no usable columns")
    return ParsedCreateTable(table_name=table_name, columns=cols)


def parse_insert(statement: str) -> ParsedInsert:
    m = re.match(
        r"(?is)^\s*INSERT\s+INTO\s+(?P<table>[`\"\[]?[A-Za-z0-9_\.]+[`\"\]]?)\s*(?:\((?P<cols>.*?)\))?\s*VALUES\s*(?P<vals>.*)\s*$",
        statement.strip(),
    )
    if not m:
        raise ValueError("Unsupported INSERT INTO statement")
    table_name = _dequote_ident(m.group("table"))
    cols_raw = m.group("cols")
    vals_raw = (m.group("vals") or "").strip()
    columns: list[str] | None = None
    if cols_raw is not None and cols_raw.strip():
        columns = [_dequote_ident(x) for x in _split_sql_csv(cols_raw)]
    rows = _parse_values_rows(vals_raw)
    if not rows:
        raise ValueError("INSERT INTO contains no VALUES rows")
    return ParsedInsert(table_name=table_name, columns=columns, rows=rows)


def import_sql_dump(
    path: Path,
    *,
    encoding: str = "utf-8",
    max_rows: int | None = None,
    chunk_rows: int = 50_000,
    on_create_table: Callable[[ParsedCreateTable], None] | None = None,
    on_insert_rows: Callable[[str, list[str] | None, list[list[Any]]], None] | None = None,
) -> dict[str, Any]:
    creates: dict[str, list[str]] = {}
    total_rows = 0
    statements = 0
    create_count = 0
    insert_count = 0
    rejected_count = 0
    for statement in iter_sql_statements(path, encoding=encoding):
        statements += 1
        normalized = statement.strip().upper()
        if any(normalized.startswith(prefix) for prefix in _UNSAFE_PREFIXES):
            raise ValueError(f"Unsafe SQL statement blocked: {normalized.split()[0]}")
        kind = statement_kind(statement)
        if kind == "create_table":
            parsed = parse_create_table(statement)
            creates[parsed.table_name] = list(parsed.columns)
            create_count += 1
            if on_create_table:
                on_create_table(parsed)
            continue
        if kind == "insert":
            parsed_insert = parse_insert(statement)
            cols = parsed_insert.columns
            if cols is None:
                cols = creates.get(parsed_insert.table_name)
                if not cols:
                    raise ValueError(
                        f"INSERT INTO {parsed_insert.table_name} omitted columns and no CREATE TABLE columns were parsed"
                    )
            rows = parsed_insert.rows
            start = 0
            while start < len(rows):
                end = min(start + max(1, int(chunk_rows)), len(rows))
                chunk = rows[start:end]
                start = end
                if max_rows is not None:
                    remaining = int(max_rows) - total_rows
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                if not chunk:
                    break
                if on_insert_rows:
                    on_insert_rows(parsed_insert.table_name, cols, chunk)
                total_rows += len(chunk)
            insert_count += 1
            if max_rows is not None and total_rows >= int(max_rows):
                break
            continue
        rejected_count += 1
        raise ValueError("Unsupported SQL statement; only CREATE TABLE and INSERT INTO are allowed")
    return {
        "statements": statements,
        "create_statements": create_count,
        "insert_statements": insert_count,
        "rejected_statements": rejected_count,
        "rows_inserted": total_rows,
        "tables_detected": sorted(creates.keys()),
    }

