import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from packaging.version import Version


def assess_audit(
    report: dict[str, Any], return_code: int, policy: dict[str, Any] | None
) -> dict[str, Any]:
    dependencies = report.get("dependencies")
    if return_code not in {0, 1} or not isinstance(dependencies, list) or not dependencies:
        raise ValueError("The dependency audit did not complete successfully.")
    if any(
        not isinstance(entry, dict)
        or entry.get("skip_reason")
        or not isinstance(entry.get("vulns"), list)
        for entry in dependencies
    ):
        raise ValueError("The dependency audit skipped or failed to inspect a dependency.")
    vulnerable = sum(bool(entry["vulns"]) for entry in dependencies)
    if return_code != int(vulnerable > 0):
        raise ValueError("Audit exit status is inconsistent with its findings.")
    informational = False
    if policy is not None:
        if (
            policy.get("schema_version") != "1.0.0"
            or policy.get("scope") != "private_scientific_research"
            or policy.get("production") is not False
            or policy.get("authorized_by") != "project_owner"
            or not policy.get("authorized_on")
            or policy.get("dependency_audit", {}).get("enforcement") != "informational"
            or policy.get("dependency_audit", {}).get("preserve_raw_findings") is not True
            or policy.get("dependency_audit", {}).get("fail_on_execution_error") is not True
        ):
            raise ValueError(
                "The research exception requires explicit non-production authorization."
            )
        informational = True
    return {
        "status": "clean"
        if not vulnerable
        else "findings_accepted_for_private_research"
        if informational
        else "blocked_by_findings",
        "exit_code": 0 if informational else return_code,
        "scanner_exit_code": return_code,
        "dependencies": len(dependencies),
        "vulnerable_packages": vulnerable,
        "raw_findings_preserved": True,
        "scope": policy["scope"] if policy else "strict",
        "authorized_on": policy["authorized_on"] if policy else None,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=root / "configs/research-acceptance.json")
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    policy = None if arguments.strict else json.loads(arguments.policy.read_text(encoding="utf-8"))
    with (root / "uv.lock").open("rb") as source:
        lock = tomllib.load(source)
    coordinates = set()
    variants = []
    for package in lock["package"]:
        if package["source"].get("editable") == ".":
            continue
        version = Version(package["version"])
        if version.local:
            if package["name"] != "torch" or version.local != "cpu":
                raise ValueError(
                    "Unrecognized local version requires an explicit advisory mapping."
                )
            variants.append(
                {
                    "name": package["name"],
                    "locked_version": str(version),
                    "advisory_version": version.public,
                    "reason": "Official PyTorch CPU build of the same upstream release.",
                }
            )
        coordinates.add((package["name"], version.public))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    requirements = arguments.output.with_suffix(".requirements.txt")
    requirements.write_text(
        "".join(f"{name}=={version}\n" for name, version in sorted(coordinates)), encoding="utf-8"
    )
    arguments.output.with_suffix(".mapping.json").write_text(
        json.dumps(
            {
                "source": "uv.lock",
                "normalized_variants": variants,
                "excluded": [
                    {
                        "name": "road-accidents",
                        "reason": "Local application, not a third-party registry dependency.",
                    }
                ],
                "scope": "All locked versions across platforms, including development dependencies.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--requirement",
            str(requirements),
            "--no-deps",
            "--disable-pip",
            "--strict",
            "--progress-spinner",
            "off",
            "--cache-dir",
            str(arguments.output.parent / "audit-cache"),
            "--format",
            "json",
            "--output",
            str(arguments.output),
            "--timeout",
            "30",
        ],
        cwd=root,
        check=False,
    )
    if not arguments.output.is_file():
        raise ValueError("The audit did not produce a findings report.")
    report = json.loads(arguments.output.read_text(encoding="utf-8"))
    outcome = assess_audit(report, result.returncode, policy)
    arguments.output.with_suffix(".policy.json").write_text(
        json.dumps(outcome, indent=2), encoding="utf-8"
    )
    print(json.dumps(outcome))
    return int(outcome["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
