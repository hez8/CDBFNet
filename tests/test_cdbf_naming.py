import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CDBFNamingTest(unittest.TestCase):
    def test_package_and_model_names(self):
        package = ROOT / "cdbf_net"
        self.assertTrue((package / "architecture.py").is_file())
        source = (package / "architecture.py").read_text(encoding="utf-8")
        self.assertIn("class CDBFNet", source)

    def test_entrypoint_uses_new_package(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertIn("from cdbf_net import CDBFNet", source)

    def test_release_text_has_no_legacy_name(self):
        legacy_name = "paid" + "net"
        text_suffixes = {".py", ".md", ".yaml", ".yml", ".txt"}
        for path in ROOT.rglob("*"):
            if path.suffix.lower() not in text_suffixes:
                continue
            if ".git" in path.parts or "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8").lower()
            self.assertNotIn(legacy_name, source, str(path))


if __name__ == "__main__":
    unittest.main()
