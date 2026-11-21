import copy
import json

import pytest


def test_research_policy_retains_findings_without_blocking(project_root):
    from scripts.audit_lock import assess_audit

    policy = json.loads((project_root / "configs/research-acceptance.json").read_text())
    report = {
        "dependencies": [
            {
                "name": "diskcache",
                "version": "5.6.3",
                "vulns": [
                    {"id": "PYSEC-2026-2447", "aliases": ["CVE-2025-69872"], "fix_versions": []}
                ],
            }
        ]
    }
    original = copy.deepcopy(report)
    result = assess_audit(report, 1, policy)
    assert result["exit_code"] == 0
    assert result["status"] == "findings_accepted_for_private_research"
    assert result["vulnerable_packages"] == 1
    assert report == original
    assert policy["human_review"]["performed"] is False


def test_strict_audit_still_rejects_findings_without_research_policy():
    from scripts.audit_lock import assess_audit

    report = {"dependencies": [{"name": "package", "vulns": [{"id": "example"}]}]}
    assert assess_audit(report, 1, None)["exit_code"] == 1


@pytest.mark.parametrize(
    "report,exit_code",
    [
        ({}, 1),
        ({"dependencies": []}, 2),
        ({"dependencies": [{"name": "package", "skip_reason": "unavailable"}]}, 1),
        ({"dependencies": [{"name": "package", "vulns": []}]}, 1),
    ],
)
def test_incomplete_or_failed_audit_is_not_waived(project_root, report, exit_code):
    from scripts.audit_lock import assess_audit

    policy = json.loads((project_root / "configs/research-acceptance.json").read_text())
    with pytest.raises(ValueError):
        assess_audit(report, exit_code, policy)


def test_research_exception_cannot_be_reused_for_production(project_root):
    from scripts.audit_lock import assess_audit

    policy = json.loads((project_root / "configs/research-acceptance.json").read_text())
    policy["production"] = True
    with pytest.raises(ValueError):
        assess_audit({"dependencies": [{"name": "package", "vulns": []}]}, 0, policy)
