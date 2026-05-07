from click.testing import CliRunner

from open_standard_evaluation.cli import main


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Open Standard Evaluation" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--langfuse-host" in result.output
        assert "--model" in result.output
        assert "--output-format" in result.output

    def test_status_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_clean_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["clean", "--help"])
        assert result.exit_code == 0
