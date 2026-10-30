"""Display pinned model and experiment configuration."""
import sys

from utils.commands import main

if __name__ == "__main__":
    raise SystemExit(main(["show-config", *sys.argv[1:]]))
