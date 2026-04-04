import pytest
from click.testing import CliRunner

from paloalto.cli import main


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "PaloAlto" in result.output

    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--embedding" in result.output

    def test_metrics_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output

    def test_metrics_command(self, mock_adata_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "metrics",
            "--input", mock_adata_path,
            "--embedding", "X_pca",
            "--batch-key", "batch",
            "--label-key", "cell_type",
        ])
        assert result.exit_code == 0, result.output
        assert "scib_overall" in result.output
