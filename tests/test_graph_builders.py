from __future__ import annotations

import sqlite3

from statistic_harness.core.graph_builders import transition_bigrams


def test_transition_bigrams_builds_deterministic_edges() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE tmpl (
          dataset_version_id TEXT NOT NULL,
          case_id TEXT NOT NULL,
          ts INTEGER NOT NULL,
          proc TEXT NOT NULL
        )
        """
    )
    rows = [
        ("d1", "c1", 1, "A"),
        ("d1", "c1", 2, "B"),
        ("d1", "c1", 3, "C"),
        ("d1", "c2", 1, "A"),
        ("d1", "c2", 2, "B"),
        ("d1", "c2", 3, "D"),
    ]
    conn.executemany("INSERT INTO tmpl VALUES (?,?,?,?)", rows)
    edges = transition_bigrams(
        conn,
        template_table="tmpl",
        dataset_version_id="d1",
        case_col="case_id",
        time_col="ts",
        process_col="proc",
        min_edge_weight=1,
    )
    # A->B happens twice; B->C and B->D once.
    assert edges[0] == ("a", "b", 2)
    assert ("b", "c", 1) in edges
    assert ("b", "d", 1) in edges
    conn.close()

