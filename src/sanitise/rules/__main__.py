"""Run the rule pass on its own: python -m sanitise.rules "text" | -f file.txt"""

import argparse
import json
import sys

from sanitise.rules import rule_detect


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m sanitise.rules")
    p.add_argument("text", nargs="?", help="text to scan (default: stdin)")
    p.add_argument("-f", "--file", help="read text from a file")
    args = p.parse_args()
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = args.text if args.text is not None else sys.stdin.read()
    spans = rule_detect(text)
    json.dump([s.model_dump(mode="json") for s in spans], sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
