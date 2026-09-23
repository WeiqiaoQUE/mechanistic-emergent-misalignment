import json
import tempfile
import unittest
from pathlib import Path

from phase_transition.utils.runtime import load_condition_manifest


class RuntimeTests(unittest.TestCase):
    def test_condition_manifest_accepts_list_format(self):
        rows = [
            {"condition_id": "Cstart", "adapter_dir": "checkpoint-100"},
            {"condition_id": "C0_real", "adapter_dir": "checkpoint-200"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            loaded = load_condition_manifest(path)

        self.assertEqual(set(loaded), {"Cstart", "C0_real"})
        self.assertEqual(loaded["C0_real"]["adapter_dir"], "checkpoint-200")


if __name__ == "__main__":
    unittest.main()
