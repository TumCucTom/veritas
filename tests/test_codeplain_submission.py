from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]

PROJECTS = {
    "contract-clients": {
        "plain": "veritas_contract_clients.plain",
        "template": "veritas-typescript-package-template.plain",
        "import": "veritas-typescript-package-template",
    },
    "bank-connectors": {
        "plain": "veritas_bank_connectors.plain",
        "template": "veritas-python-package-template.plain",
        "import": "veritas-python-package-template",
    },
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class CodeplainSubmissionTest(unittest.TestCase):
    def test_codeplain_projects_keep_plain_specs_and_config(self) -> None:
        base = ROOT / "codeplain"
        self.assertTrue((base / "README.md").is_file())
        self.assertTrue((base / ".gitignore").is_file())

        for project, expected in PROJECTS.items():
            with self.subTest(project=project):
                folder = base / project
                plain_file = folder / expected["plain"]
                config_file = folder / "config.yaml"
                template_file = folder / expected["template"]

                self.assertTrue(plain_file.is_file())
                self.assertTrue(config_file.is_file())
                self.assertTrue(template_file.is_file())

                template = read(template_file)
                for built_in_concept in (":Implementation:", ":UnitTests:", ":ConformanceTests:"):
                    self.assertNotRegex(
                        template,
                        rf"(?m)^- {re.escape(built_in_concept)} is ",
                        msg=f"{template_file} redefines Codeplain concept {built_in_concept}",
                    )

                plain = read(plain_file)
                self.assertIn(f"- {expected['import']}", plain)
                self.assertIn("***definitions***", plain)
                self.assertIn("***implementation reqs***", plain)
                self.assertIn("***test reqs***", plain)
                self.assertIn("***functional specs***", plain)
                self.assertNotRegex(plain, r"(?m)^\*\*\*acceptance tests\*\*\*$")

                functional_items = re.findall(
                    r"(?ms)^- .+?(?=^\- |\Z)",
                    plain.split("***functional specs***", 1)[1].strip(),
                )
                self.assertGreaterEqual(len(functional_items), 3)
                for item in functional_items:
                    self.assertIn("***acceptance tests***", item)

                config = read(config_file)
                self.assertIn("headless: true", config)
                self.assertIn("copy-build: true", config)
                self.assertIn("build-folder: plain_modules", config)
                self.assertIn("build-dest: build", config)
                self.assertIn("unittests-script:", config)
                self.assertIn("conformance-tests-script:", config)

                for script in (folder / "scripts").glob("run_conformance_tests_*.sh"):
                    script_text = read(script)
                    self.assertIn("SOURCE_DIR_ABS=", script_text)
                    self.assertIn("CONFORMANCE_DIR_ABS=", script_text)

    def test_generated_codeplain_outputs_are_ignored_but_sources_are_not(self) -> None:
        ignore_file = ROOT / "codeplain" / ".gitignore"
        self.assertTrue(ignore_file.is_file())
        ignore = read(ignore_file)

        for generated_folder in ("plain_modules/", "conformance_tests/", "build/"):
            self.assertIn(generated_folder, ignore)

        self.assertNotIn("*.plain", ignore)
        self.assertNotIn("config.yaml", ignore)


if __name__ == "__main__":
    unittest.main()
