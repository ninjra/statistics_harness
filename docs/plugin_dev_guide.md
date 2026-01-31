# Plugin Development Guide

Plugins live in `plugins/<plugin_id>` and provide a `plugin.yaml` manifest with
an entrypoint class that implements `run(ctx)`.
