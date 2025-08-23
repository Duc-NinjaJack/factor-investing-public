from pathlib import Path
import re


def test_no_deprecated_config_imports():
    root = Path('.')
    bad_refs = []
    pattern = re.compile(r"^(from\s+production\.tests\.phase31_add_fscore\s+import\s+07_QVM_flat_config|import\s+07_QVM_flat_config)\b", re.M)
    for p in root.rglob('*.py'):
        # Exclude this test file itself
        if p.name == 'test_no_deprecated_imports.py':
            continue
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if pattern.search(text):
            bad_refs.append(str(p))
    assert not bad_refs, f"Deprecated config imports found in: {bad_refs}"


