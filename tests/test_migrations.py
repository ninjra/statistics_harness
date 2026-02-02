from pathlib import Path
import shutil
import sqlite3

from statistic_harness.core.migrations import MIGRATIONS, run_migrations


FIXTURE_DIR = Path("tests/fixtures/db")


def _expected_tables(version: int) -> dict[str, int]:
    expected = {
        "runs": 1,
        "plugin_results": 1,
        "plugin_results_v2": 1,
    }
    if version >= 2:
        expected.update(
            {
                "projects": 1,
                "datasets": 1,
                "dataset_versions": 1,
                "dataset_columns": 1,
            }
        )
    if version >= 4:
        expected.update(
            {
                "parameter_entities": 1,
                "parameter_kv": 1,
                "row_parameter_link": 1,
                "entities": 2,
                "edges": 1,
            }
        )
    if version >= 5:
        expected.update(
            {
                "analysis_jobs": 1,
                "deliveries": 1,
            }
        )
    if version >= 6:
        expected.update(
            {
                "raw_formats": 1,
                "raw_format_notes": 1,
                "templates": 1,
                "template_fields": 1,
                "dataset_templates": 1,
                "template_conversions": 1,
            }
        )
    if version >= 7:
        expected.update({"raw_format_mappings": 1})
    if version >= 8:
        expected.update({"plugin_executions": 1})
    if version >= 10:
        expected.update({"known_issue_sets": 1, "known_issues": 1})
    if version >= 11:
        expected.update({"dataset_role_candidates": 1, "project_role_overrides": 1})
    if version >= 13:
        expected.update({"project_plugin_settings": 0})
    if version >= 15:
        expected.update({"pii_salts": 0, "pii_entities": 0})
    if version >= 16:
        expected.update({"tenants": 1})
    if version >= 17:
        expected.update(
            {
                "users": 0,
                "tenant_memberships": 0,
                "user_sessions": 0,
                "api_keys": 0,
            }
        )
    if version >= 18:
        expected.update({"vector_collections": 0})
    return expected


def test_migrations_from_golden(tmp_path):
    total_versions = len(MIGRATIONS)
    for version in range(1, total_versions + 1):
        fixture = FIXTURE_DIR / f"v{version}.sqlite"
        assert fixture.exists()
        db_path = tmp_path / f"v{version}.sqlite"
        shutil.copy(fixture, db_path)
        conn = sqlite3.connect(db_path)
        run_migrations(conn)
        cur = conn.execute("PRAGMA user_version")
        assert int(cur.fetchone()[0]) == total_versions
        expected = _expected_tables(version)
        for table, min_count in expected.items():
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = int(cur.fetchone()[0])
            assert count >= min_count
        conn.close()
