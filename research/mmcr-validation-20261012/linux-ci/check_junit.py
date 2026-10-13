"""Fail unless a JUnit report shows only passed tests (no skips, failures or errors)."""

import json
import sys
import xml.etree.ElementTree as ET


def main(path: str) -> None:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    cases = [case for suite in suites for case in suite.iter("testcase")]
    summary = {
        "tests": len(cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "passed": [
            f"{case.get('classname')}::{case.get('name')}"
            for case in cases
            if all(case.find(tag) is None for tag in ("skipped", "failure", "error"))
        ],
    }
    print(json.dumps(summary, indent=2))
    if not cases or summary["skipped"] or summary["failures"] or summary["errors"]:
        sys.exit("JUnit report is not an all-PASSED result")


if __name__ == "__main__":
    main(sys.argv[1])
