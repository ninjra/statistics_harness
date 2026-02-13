from statistic_harness.core.stat_plugins.contract import validate_contract


def _base_result():
    return {
        "plugin_id": "analysis_example",
        "version": "0.1.0",
        "status": "ok",
        "summary": {"headline": "Example"},
        "findings": [
            {
                "id": "f1",
                "severity": "info",
                "confidence": 0.5,
                "title": "Example",
                "what": "Something happened",
                "why": "Because of reasons",
            }
        ],
        "artifacts": [],
        "metrics": {},
        "references": {},
        "debug": {},
    }


def test_validate_contract_ok():
    errors = validate_contract(_base_result())
    assert errors == []


def test_validate_contract_missing_key():
    result = _base_result()
    result.pop("metrics")
    errors = validate_contract(result)
    assert any("missing_top_keys" in err for err in errors)


def test_validate_contract_invalid_status():
    result = _base_result()
    result["status"] = "unknown"
    errors = validate_contract(result)
    assert "invalid_status" in errors


def test_validate_contract_findings_not_list():
    result = _base_result()
    result["findings"] = "oops"
    errors = validate_contract(result)
    assert "findings_not_list" in errors


def test_validate_contract_missing_finding_fields():
    result = _base_result()
    result["findings"] = [{"id": "x"}]
    errors = validate_contract(result)
    assert "finding_0_missing_severity" in errors
    assert "finding_0_missing_confidence" in errors
    assert "finding_0_missing_title" in errors
    assert "finding_0_missing_what" in errors
    assert "finding_0_missing_why" in errors


def test_validate_contract_invalid_severity():
    result = _base_result()
    result["findings"][0]["severity"] = "nope"
    errors = validate_contract(result)
    assert "finding_0_invalid_severity" in errors


def test_validate_contract_max_findings():
    result = _base_result()
    result["findings"].append(
        {
            "id": "f2",
            "severity": "warn",
            "confidence": 0.2,
            "title": "Second",
            "what": "Other",
            "why": "Because",
        }
    )
    errors = validate_contract(result, max_findings=1)
    assert "findings_exceed_max" in errors
