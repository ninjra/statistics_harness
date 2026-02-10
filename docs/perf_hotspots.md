# Performance Hotspots

This doc is a living scratchpad for identifying the slowest / most memory-intensive plugins on real datasets.

## How To Generate A Hotspot List

1. Run a full pipeline run (all plugins).
2. Query the run execution telemetry (duration + RSS) from SQLite.

Example SQL (adjust `run_id`):

```sql
SELECT plugin_id, status, duration_ms, max_rss, exit_code
FROM plugin_executions
WHERE run_id = '<run_id>'
ORDER BY max_rss DESC, duration_ms DESC;
```

Notes:
- `max_rss` is `ru_maxrss` from Linux (KB on Linux). The report embeds both `max_rss_kb` and derived `max_rss_bytes`.
- Newer runs also include a `report.json -> hotspots` block with top duration/RSS lists.

## What To Do With Hotspots (4 Pillars)

- Performant: migrate hotspots to `ctx.dataset_iter_batches()` and/or SQL aggregation.
- Accurate: avoid row sampling; prefer multi-pass scans with bounded complexity caps.
- Secure: keep all artifacts under `./appdata/`; avoid shelling out; validate inputs.
- Citable: ensure each recommendation references the plugin ID and the evidence fields (duration/RSS/budget).
