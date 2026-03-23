from __future__ import annotations

from collections.abc import Iterable


def pytest_terminal_summary(terminalreporter, exitstatus: int, config) -> None:
    passed_reports: Iterable = terminalreporter.stats.get("passed", [])
    passed_nodeids = [report.nodeid for report in passed_reports if getattr(report, "when", None) == "call"]
    if not passed_nodeids:
        return

    terminalreporter.section("successful tests")
    for nodeid in passed_nodeids:
        terminalreporter.write_line(f"PASS: {nodeid}")
