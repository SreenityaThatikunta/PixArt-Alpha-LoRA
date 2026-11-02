"""Dependency-free checks: every script parses, and the README does not lie.

These run without torch/diffusers installed, so they stay green even when the
heavy ML stack is unavailable.
"""
import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((REPO_ROOT / "scripts").glob("*.py"))


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


def _parse_tree_block(block: str) -> list[str]:
    """Turn an indented ``tree``-style block into repo-relative paths.

    Indentation gives nesting, so a child under ``scripts/`` resolves to
    ``scripts/train.py`` rather than a bare ``train.py`` at the root.
    """
    paths: list[str] = []
    parents: dict[int, str] = {}

    for line in block.splitlines():
        match = re.match(r"^(?P<indent>[\s│]*)[├└]──\s+(?P<name>.+?)(?:\s{2,}#.*)?$", line)
        if not match:
            continue

        depth = len(match.group("indent")) // 4
        name = match.group("name").strip()
        prefix = parents.get(depth - 1, "") if depth else ""
        path = f"{prefix}{name}"

        if name.endswith("/"):
            parents[depth] = path
        paths.append(path.rstrip("/"))

    return paths


def test_readme_project_structure_files_exist():
    """Every path listed in the README's Project Structure block must exist."""
    readme = (REPO_ROOT / "README.md").read_text()
    block = re.search(r"## Project Structure\s*```(.*?)```", readme, re.S)
    assert block, "README is missing its Project Structure block"

    listed = _parse_tree_block(block.group(1))
    assert listed, "Project Structure block lists no files"

    missing = [name for name in listed if not (REPO_ROOT / name).exists()]
    assert not missing, f"README lists paths that do not exist: {missing}"


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
