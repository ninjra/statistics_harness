# Evaluation Harness

The evaluator compares `report.json` against a ground truth YAML file to assert
expected findings (feature discovery, changepoints, dependence shifts, and
anomalies) within configured tolerances. `strict` defaults to `true`; set
`strict: false` in the YAML to allow unexpected findings.

You can also specify `expected_findings` for plugin-specific assertions:

```
expected_findings:
  - plugin_id: analysis_process_sequence
    kind: process_variant
    contains:
      variant: qemail
    min_count: 1
```

`where` enforces exact matches, `contains` checks substring/list membership.

You can also specify expected numeric metrics:

```
expected_metrics:
  - plugin_id: analysis_queue_delay_decomposition
    metric: eligible_wait.p95
    value: 120.0
    tolerance:
      absolute: 5
      relative: 0.1
```

Tolerance supports `absolute` and `relative` (fractional) thresholds. When `strict`
is true, unexpected findings for a known kind will fail evaluation.
