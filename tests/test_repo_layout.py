"""Dependency-free checks: every script parses, and the README does not lie.

These run without torch/diffusers installed, so they stay green even when the
heavy ML stack is unavailable.
"""
import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(REPO_ROOT.glob("*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_is_syntactically_valid(script):
    ast.parse(script.read_text(), filename=str(script))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_hardcoded_secrets(script):
    """Guard against a pasted W&B / HF token making it into a commit."""
    text = script.read_text()
    patterns = [
        r"hf_[A-Za-z0-9]{34,}",            # HuggingFace user token
        r"\bwandb\.login\([^)]*key\s*=\s*[\"'][^\"']+[\"']",
        r"[\"'][0-9a-f]{40}[\"']",         # bare 40-hex W&B key
    ]
    for pattern in patterns:
        assert not re.search(pattern, text), f"possible secret in {script.name}: {pattern}"


def test_readme_project_structure_files_exist():
    """Every path listed in the README's Project Structure block must exist."""
    readme = (REPO_ROOT / "README.md").read_text()
    block = re.search(r"## Project Structure\s*```(.*?)```", readme, re.S)
    assert block, "README is missing its Project Structure block"

    # Paths may contain spaces, so take everything up to the comment column.
    listed = [
        match.group(1).strip()
        for match in re.finditer(r"[├└]──\s+(.+?)(?:\s{2,}#.*)?$", block.group(1), re.M)
    ]
    assert listed, "Project Structure block lists no files"

    missing = [name for name in listed if not (REPO_ROOT / name).exists()]
    assert not missing, f"README lists files that do not exist: {missing}"


def test_repo_ships_the_usual_metadata():
    for name in ("LICENSE", "requirements.txt", ".gitignore", ".env.example"):
        assert (REPO_ROOT / name).exists(), f"missing {name}"


def test_env_example_has_no_filled_in_values():
    """.env.example must stay a template - keys only, never real values."""
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        assert value == "" or value.islower() or "-" in value, (
            f".env.example appears to contain a real value for {key}"
        )
