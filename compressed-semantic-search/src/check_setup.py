"""Check the Python/configuration scaffold."""
import sys

from utils.commands import main

if __name__ == "__main__":
    raise SystemExit(main(["check", *sys.argv[1:]]))
