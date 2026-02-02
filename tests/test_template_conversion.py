from pathlib import Path

import pandas as pd

from statistic_harness.core.pipeline import Pipeline
from tests.conftest import make_context


def test_template_conversion(run_dir):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    ctx = make_context(run_dir, df, {})

    template_id = ctx.storage.create_template(
        name="Basic",
        fields=[
            {"name": "alpha", "dtype": "int"},
            {"name": "beta", "dtype": "text"},
        ],
        description=None,
        version=None,
        created_at="now",
    )

    mapping = {"alpha": "a", "beta": "b"}
    pipeline = Pipeline(run_dir, Path("plugins"))
    run_id = pipeline.run(
        None,
        ["transform_template"],
        {"transform_template": {"template_id": template_id, "mapping": mapping}},
        0,
        dataset_version_id=ctx.dataset_version_id,
    )
    assert run_id
    dataset_template = ctx.storage.fetch_dataset_template(ctx.dataset_version_id)
    assert dataset_template
    assert dataset_template["status"] == "ready"
