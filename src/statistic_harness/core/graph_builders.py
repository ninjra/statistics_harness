from __future__ import annotations

from typing import Any

from statistic_harness.core.utils import quote_identifier


def transition_bigrams(
    conn: Any,
    *,
    template_table: str,
    dataset_version_id: str,
    case_col: str,
    time_col: str,
    process_col: str,
    min_edge_weight: int = 5,
) -> list[tuple[str, str, int]]:
    """Build a deterministic process->process bigram edge list from a template table.

    Uses SQLite window functions to compute `LEAD(process)` per case ordered by time.
    """

    rows = conn.execute(
        f"""
        WITH ordered AS (
          SELECT
            LOWER(TRIM({quote_identifier(process_col)})) AS proc,
            LEAD(LOWER(TRIM({quote_identifier(process_col)}))) OVER (
              PARTITION BY {quote_identifier(case_col)} ORDER BY {quote_identifier(time_col)}
            ) AS next_proc
          FROM {quote_identifier(template_table)}
          WHERE dataset_version_id = ?
            AND {quote_identifier(case_col)} IS NOT NULL
            AND {quote_identifier(process_col)} IS NOT NULL
            AND {quote_identifier(time_col)} IS NOT NULL
        )
        SELECT proc, next_proc, COUNT(*) AS n
        FROM ordered
        WHERE next_proc IS NOT NULL AND proc != '' AND next_proc != ''
        GROUP BY proc, next_proc
        HAVING n >= ?
        ORDER BY n DESC
        """,
        (dataset_version_id, int(min_edge_weight)),
    ).fetchall()
    return [(str(r["proc"]), str(r["next_proc"]), int(r["n"])) for r in rows]

