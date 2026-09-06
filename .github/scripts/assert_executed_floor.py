#!/usr/bin/env python3
"""Refuse a green test run that executed almost nothing.

Roughly half of this suite is gated on CUDA or on a compiled ``sweep._C``
symbol, and a skip is not a failure: a run with no GPU, or with a binding that
silently failed to build, prints a green summary over half the suite. Without a
floor, "the tests passed" and "the tests did not run" look identical.

    assert_executed_floor.py report.xml 300
"""
import sys
import xml.etree.ElementTree as ET


def counts(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = skipped = failed = 0
    for s in suites:
        total += int(s.get("tests", 0))
        skipped += int(s.get("skipped", 0))
        failed += int(s.get("failures", 0)) + int(s.get("errors", 0))
    return total, skipped, failed


def main():
    if len(sys.argv) != 3:
        print(__doc__.strip())
        return 2
    path, floor = sys.argv[1], int(sys.argv[2])
    total, skipped, failed = counts(path)
    executed = total - skipped
    print(f"collected {total}, skipped {skipped}, executed {executed}, failed {failed}")
    if failed:
        print(f"FAIL: {failed} test(s) failed")
        return 1
    if executed < floor:
        print(
            f"FAIL: only {executed} tests actually ran, below the floor of {floor}.\n"
            "      Something is skipping wholesale -- a missing dependency, or a\n"
            "      compiled binding that did not build. Raise the floor only when\n"
            "      the suite genuinely got smaller, and say why in the commit."
        )
        return 1
    print(f"ok: {executed} executed, floor {floor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
