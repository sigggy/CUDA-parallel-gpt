#!/usr/bin/env python3
"""Sort words in a file by descending length."""

import sys


def sort_words_by_length(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        words = [line.rstrip("\n") for line in f if line.strip()]

    sorted_words = sorted(words, key=len, reverse=True)

    with open(output_path, "w") as f:
        f.write("\n".join(sorted_words) + "\n")

    print(f"Wrote {len(sorted_words)} words to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)

    sort_words_by_length(sys.argv[1], sys.argv[2])