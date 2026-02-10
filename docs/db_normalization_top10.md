# DB Normalization: Top 10 Performance Optimizations (Plugin-Driven)

This doc summarizes the highest-leverage optimizations to apply during the **raw -> normalized DB preparation phase** (ingest + normalization), before most plugins run.

## Why Focus Here

From `docs/plugin_data_access_matrix.json`:
- 139 plugins total
- 121 plugins call `ctx.dataset_loader()` unbounded (full dataset load)
- 0 plugins use `ctx.dataset_iter_batches()` today
- 2 plugins do direct SQL

So the harness spends most of its time repeatedly scanning the same SQLite tables across many subprocesses. The best ROI is making DB scans fast and reusing full-dataset derived facts.

## Top 10 (Ordered By ROI)

1. Ensure a `row_index` index exists after ingest completes
   - Reason: most reads are `ORDER BY row_index` / range pagination
   - Implemented: `plugins/ingest_tabular/plugin.py`, `src/statistic_harness/core/storage.py`

2. Run `ANALYZE` after bulk ingest and after bulk template normalization
   - Reason: improves query planning for subsequent plugin scans
   - Implemented: `plugins/ingest_tabular/plugin.py`, `plugins/transform_normalize_mixed/plugin.py`, `src/statistic_harness/core/storage.py`

3. Add a composite index on normalized template tables: `(dataset_version_id, row_index)`
   - Reason: template reads are typically row-ordered within one dataset_version_id
   - Implemented: `src/statistic_harness/core/storage.py`

4. Compute a full-dataset column statistics snapshot during ingest (avoid later re-scans)
   - Stats stored per column:
     - `n`, `nulls`
     - numeric: `min`, `max`, `mean`
     - text: `min_len`, `max_len`, `numeric_like_ratio`
   - Implemented: `plugins/ingest_tabular/plugin.py`, `src/statistic_harness/core/storage.py`, `src/statistic_harness/core/migrations.py`

5. Drive normalization decisions from stored full-dataset stats (no sampling)
   - Example: numeric coercion uses `numeric_like_ratio` (computed at ingest over the full dataset)
   - Implemented: `plugins/transform_normalize_mixed/plugin.py`

6. Increase default bulk ingest / normalization chunk size
   - Reason: reduces SQLite + Python overhead per row for multi-million-row datasets
   - Implemented (defaults): `plugins/ingest_tabular/plugin.py`, `plugins/transform_normalize_mixed/plugin.py`

7. Create a small, capped set of additional indexes for common key columns (best-effort)
   - Heuristic role hints from column names/types (e.g. `timestamp`, `id`, `event`, `variant`, `status`)
   - Index creation is capped to avoid index explosion
   - Implemented: `plugins/ingest_tabular/plugin.py`, `src/statistic_harness/core/storage.py`

8. Optional: materialize a numeric column cache for repeated scans across many plugins
   - Reason: avoids re-reading/parsing numeric columns from SQLite N times
   - Implemented (opt-in): `src/statistic_harness/core/dataset_cache.py`, `src/statistic_harness/core/dataset_io.py`, `scripts/materialize_dataset_cache.py`, `scripts/inspect_dataset_cache.py`

9. Use cursor-based pagination for batch iteration (`row_index > last ORDER BY row_index LIMIT n`)
   - Reason: works even if `row_index` is not perfectly contiguous and avoids large `WHERE row_index BETWEEN ...` assumptions
   - Implemented: `src/statistic_harness/core/dataset_io.py`

10. Next improvements (recommended; not yet fully implemented everywhere)
   - Add durable summary tables computed once per dataset_version_id, such as:
     - exact distinct counts per column (`COUNT(DISTINCT ...)`) where feasible
     - top-k categorical values per column
     - numeric quantiles via deterministic multi-pass scans
   - Use execution telemetry (`duration_ms`, `max_rss`) to decide which additional indexes/summaries are worth building.

