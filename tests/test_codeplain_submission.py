from pathlib import Path
import re
import subprocess
import tempfile
import textwrap
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
    def test_submission_materials_map_codeplain_work_to_judging_criteria(self) -> None:
        submission = ROOT / "codeplain" / "SUBMISSION.md"
        root_readme = ROOT / "README.md"

        self.assertTrue(submission.is_file())

        submission_text = read(submission)
        for phrase in (
            "spec-driven development setup",
            "presentation",
            "innovation and creativity",
            "charm",
            "contract-clients",
            "bank-connectors",
            "python3 -m unittest tests.test_codeplain_submission",
        ):
            self.assertIn(phrase, submission_text)

        self.assertIn("codeplain/SUBMISSION.md", read(root_readme))

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

    def test_codeplain_specs_target_real_veritas_paths_and_runnable_packages(self) -> None:
        contract_plain = read(
            ROOT / "codeplain" / "contract-clients" / "veritas_contract_clients.plain"
        )
        for expected in (
            "package.json",
            "tsconfig.json",
            "vitest.config.ts",
            "`contract/types.ts`",
            "`web/src/lib/`",
            "`edge-sdk/src/`",
            "`StateSnapshot`",
            "at least four banks",
        ):
            self.assertIn(expected, contract_plain)

        for stale_path in ("apps/console", "packages/sdk", "services/api"):
            self.assertNotIn(stale_path, contract_plain)

        template = read(
            ROOT
            / "codeplain"
            / "contract-clients"
            / "veritas-typescript-package-template.plain"
        )
        for expected in ("package.json", "tsconfig.json", "vitest.config.ts"):
            self.assertIn(expected, template)

    def test_typescript_runner_requires_real_package_or_tsconfig(self) -> None:
        script = (
            ROOT
            / "codeplain"
            / "contract-clients"
            / "scripts"
            / "run_unittests_typescript.sh"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "index.ts").write_text("export const ok = true;\n", encoding="utf-8")

            result = subprocess.run(
                [str(script), str(source)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("package.json or tsconfig.json", result.stderr)

    def test_python_conformance_runner_supports_nested_generated_modules(self) -> None:
        script = (
            ROOT
            / "codeplain"
            / "bank-connectors"
            / "scripts"
            / "run_conformance_tests_python.sh"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            conformance = Path(tmp) / "conformance"
            nested = conformance / "connector_fixtures_and_conformance_test_generation"
            source.mkdir()
            nested.mkdir(parents=True)
            (nested / "fixtures_generator.py").write_text(
                "VALUE = 'synthetic-fixture'\n",
                encoding="utf-8",
            )
            (nested / "test_conformance.py").write_text(
                textwrap.dedent(
                    """
                    import unittest
                    import fixtures_generator


                    class NestedImportTest(unittest.TestCase):
                        def test_sibling_module_imports(self):
                            self.assertEqual(fixtures_generator.VALUE, "synthetic-fixture")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [str(script), str(source), str(conformance)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_python_conformance_runner_isolates_sibling_test_packages(self) -> None:
        script = (
            ROOT
            / "codeplain"
            / "bank-connectors"
            / "scripts"
            / "run_conformance_tests_python.sh"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            conformance = Path(tmp) / "conformance"
            source.mkdir()
            source_tests = source / "tests"
            source_tests.mkdir()
            (source_tests / "__init__.py").write_text("", encoding="utf-8")
            (source_tests / "fixtures_generator.py").write_text(
                "VALUE = 'source-package-should-not-win'\n",
                encoding="utf-8",
            )

            for name, value in (("alpha_connector", "alpha"), ("beta_connector", "beta")):
                tests_dir = conformance / name / "tests"
                tests_dir.mkdir(parents=True)
                (tests_dir / "fixtures_generator.py").write_text(
                    f"VALUE = {value!r}\n",
                    encoding="utf-8",
                )
                (tests_dir / f"test_{value}.py").write_text(
                    textwrap.dedent(
                        f"""
                        import unittest
                        import tests.fixtures_generator as fixtures_generator


                        class IsolatedPackageTest(unittest.TestCase):
                            def test_imports_own_sibling_package(self):
                                self.assertEqual(fixtures_generator.VALUE, {value!r})
                        """
                    ).strip()
                    + "\n",
                    encoding="utf-8",
                )

            result = subprocess.run(
                [str(script), str(source), str(conformance)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_python_conformance_runner_links_root_fixture_generators_for_tests_package(self) -> None:
        script = (
            ROOT
            / "codeplain"
            / "bank-connectors"
            / "scripts"
            / "run_conformance_tests_python.sh"
        )
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            conformance = Path(tmp) / "conformance"
            child = conformance / "connector_fixtures"
            tests_dir = child / "tests"
            source.mkdir()
            tests_dir.mkdir(parents=True)
            (child / "fixtures_generator.py").write_text(
                "VALUE = 'root-fixture-generator'\n",
                encoding="utf-8",
            )
            (tests_dir / "test_conformance.py").write_text(
                textwrap.dedent(
                    """
                    import unittest
                    import tests.fixtures_generator as fixtures_generator


                    class RootFixtureShimTest(unittest.TestCase):
                        def test_tests_package_can_import_root_fixture_generator(self):
                            self.assertEqual(fixtures_generator.VALUE, "root-fixture-generator")
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [str(script), str(source), str(conformance)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_bank_connector_spec_names_current_contract_failures_explicitly(self) -> None:
        bank_plain = read(
            ROOT / "codeplain" / "bank-connectors" / "veritas_bank_connectors.plain"
        )

        for expected in (
            "`iter_records()`",
            "must not expose `fetch_records()`",
            "`generate_conformance_fixtures(base_dir)`",
        ):
            self.assertIn(expected, bank_plain)

    def test_generated_codeplain_outputs_are_ignored_but_sources_are_not(self) -> None:
        ignore_file = ROOT / "codeplain" / ".gitignore"
        self.assertTrue(ignore_file.is_file())
        ignore = read(ignore_file)

        for generated_folder in ("plain_modules/", "conformance_tests/", "build/"):
            self.assertIn(generated_folder, ignore)

        self.assertNotIn("*.plain", ignore)
        self.assertNotIn("config.yaml", ignore)

    def test_contract_is_single_source_of_truth_for_predict_and_events(self) -> None:
        """schema.py (canonical) and types.ts (mirror) must declare the same
        contract surface: the State/Bank/Detection/Counters/Provenance core, the
        PredictRequest/PredictResponse pair, and the fraud_propagated event."""
        schema = read(ROOT / "contract" / "schema.py")
        types = read(ROOT / "contract" / "types.ts")

        # schema.py is documented as the source of truth.
        self.assertRegex(schema.lower(), r"single source of truth")
        self.assertRegex(types.lower(), r"source of truth")

        # Core contract types appear in both files.
        for name in ("Detection", "Bank", "RegimeCounters", "Counters", "Provenance"):
            self.assertIn(name, schema, msg=f"{name} missing from schema.py")
            self.assertIn(name, types, msg=f"{name} missing from types.ts")

        # /predict contract is now declared in both single-source files.
        for name in ("PredictRequest", "PredictResponse"):
            self.assertIn(name, schema, msg=f"{name} missing from schema.py")
            self.assertIn(name, types, msg=f"{name} missing from types.ts")

        # PredictResponse must carry label/confidence/indicators in both files
        # (mirrors the real node /predict endpoint).
        for field in ("label", "confidence", "indicators"):
            self.assertIn(field, schema)
            self.assertIn(field, types)

        # The fraud_propagated event the web subscribes to must be declared.
        self.assertIn("fraud_propagated", types)

        # Both express the camelCase privacy invariant field.
        self.assertIn("customerRecordsTransmitted", schema)
        self.assertIn("customerRecordsTransmitted", types)

    def test_generated_contract_client_holds_conformance_invariants(self) -> None:
        """If the contract-clients package has been generated, the generated
        validators/conformance suite must satisfy the basic invariants the spec
        requires: a real 0..1 confidence check, domain-aliased runtime exports,
        and no Jasmine .withContext() calls (which throw under Vitest)."""
        base = ROOT / "codeplain" / "contract-clients"
        src = base / "build" / "src"
        validators = src / "validators.ts"
        types = src / "types.ts"
        if not validators.is_file() or not types.is_file():
            self.skipTest("contract-clients build/ not generated in this checkout")

        validators_text = read(validators)
        types_text = read(types)

        # Real 0..1 confidence validation lives in isPredictResponse.
        self.assertIn("isPredictResponse", validators_text)
        self.assertRegex(
            validators_text,
            r"confidence\s*>=\s*0",
            msg="isPredictResponse must reject confidence < 0",
        )
        self.assertRegex(
            validators_text,
            r"confidence\s*<=\s*1",
            msg="isPredictResponse must reject confidence > 1",
        )

        # Domain-specific aliases are exported at runtime (declaration merging).
        for runtime_export in ("StateSnapshot", "BankSummary", "ProvenanceEntry"):
            self.assertRegex(
                types_text,
                rf"export const {runtime_export}\b",
                msg=f"{runtime_export} must be a runtime export, not just a type",
            )

        # No Jasmine .withContext() anywhere in the generated conformance suite.
        conformance_dir = base / "build_conformance_tests"
        if conformance_dir.is_dir():
            for test_file in conformance_dir.rglob("*.test.ts"):
                self.assertNotIn(
                    ".withContext",
                    read(test_file),
                    msg=f"{test_file} uses Jasmine .withContext (throws under Vitest)",
                )


if __name__ == "__main__":
    unittest.main()
