from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPOSITORY_ROOT / "run_smoke_train.sh"


class SmokeScriptTests(unittest.TestCase):
    def test_tmp_capacity_check_fails_before_mktemp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            marker = root / "mktemp-was-called"
            available = 39 * 1024 * 1024 * 1024

            fake_df = fake_bin / "df"
            fake_df.write_text(
                "#!/bin/sh\n"
                "printf 'Filesystem 1B-blocks Used Available Capacity Mounted on\\n'\n"
                f"printf 'fake 1 1 {available} 1%% /tmp\\n'\n",
                encoding="utf-8",
            )
            fake_df.chmod(0o755)

            fake_mktemp = fake_bin / "mktemp"
            fake_mktemp.write_text(
                "#!/bin/sh\n"
                ': > "$SMOKE_TEST_MKTEMP_MARKER"\n'
                "exit 99\n",
                encoding="utf-8",
            )
            fake_mktemp.chmod(0o755)

            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["SMOKE_TEST_MKTEMP_MARKER"] = str(marker)
            environment.pop("JMOSHIVIS_SMOKE_MIN_TMP_FREE_BYTES", None)
            result = subprocess.run(
                ["bash", str(SMOKE_SCRIPT)],
                cwd=REPOSITORY_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Insufficient free space in /tmp", result.stderr)
            self.assertIn("40 GiB by default", result.stderr)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
