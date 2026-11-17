"""Every text read/write must name encoding="utf-8".

Without it Windows decodes files with cp1252, so '¬' in 03_logic/rules.yaml becomes 'Â¬' and rule parsing fails.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_IO = {"read_text", "write_text", "open"}


def _offenders():
    for py in sorted(ROOT.glob("**/*.py")):
        if ".venv" in py.parts:
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name not in TEXT_IO:
                continue
            mode = next((a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)), "")
            if "b" in mode:
                continue
            if not any(k.arg == "encoding" for k in node.keywords):
                yield f"{py.relative_to(ROOT)}:{node.lineno}"


def test_all_text_io_is_utf8():
    assert list(_offenders()) == []
