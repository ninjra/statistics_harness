# Close-cycle revenue compression

Summary:
- close_window_mode: override
- close_window_source: override
- target_days: 7.0
- basis: queue_wait
- anchor_basis: queue
- months_evaluated: 1

Result:
| process | scale_median | scale_p90 | worst_month | worst_month_scale |
| --- | --- | --- | --- | --- |
| revenue | 1.167 | 1.167 | 2026-01 | 1.167 |

Month details:
| month | rows | baseline_days | modeled_days | required_scale | reason |
| --- | --- | --- | --- | --- | --- |
| 2026-01 | 1 | 8.00 | 7.00 | 1.167 | ok |