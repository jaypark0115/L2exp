from __future__ import annotations

import sys

from train_l2_standalone import main


def has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


if not has_arg("--modes"):
    sys.argv.extend(["--modes", "sen"])

if not has_arg("--run-name"):
    sys.argv.extend(["--run-name", "server_run_l2_sen_001"])


if __name__ == "__main__":
    main()
