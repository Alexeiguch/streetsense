#!/usr/bin/env python3
"""Summarise a StreetSense event-trace JSONL file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from streetsense.event_trace import replay_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    args = ap.parse_args()
    print(json.dumps(replay_summary(args.trace), indent=2))


if __name__ == "__main__":
    main()
