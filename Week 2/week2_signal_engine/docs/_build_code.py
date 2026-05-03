"""Build all_code.md — combines every source file into one markdown document."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the parent of docs/ as the project root
OUT = os.path.join(BASE, "docs", "all_code.md")

FILES = [
    "configs/params_example.yaml",
    "conftest.py",
    "requirements.txt",
    "src/data/loaders.py",
    "src/data/validation.py",
    "src/signals/spread.py",
    "src/signals/zscore.py",
    "src/signals/state_machine.py",
    "src/signals/kalman.py",
    "src/analytics/characterize.py",
    "src/analytics/diagnostics.py",
    "src/analytics/sensitivity.py",
    "src/utils/config.py",
    "src/utils/dates.py",
    "src/utils/io.py",
    "src/visuals/plots.py",
    "src/pipeline/run_week2.py",
    "scripts/run_all_pairs_diagnostics.py",
    "tests/test_spread.py",
    "tests/test_zscore.py",
    "tests/test_state_machine.py",
    "tests/test_kalman.py",
]

LANG = {".yaml": "yaml", ".py": "python", ".txt": "text"}
FENCE = chr(96) * 3  # three backticks

with open(OUT, "w", encoding="utf-8") as out:
    out.write("# Week 2 Signal Engine \u2014 Complete Source Code\n\n")
    out.write("All source files from `week2_signal_engine/`, ordered by module.\n\n")
    out.write("---\n\n")

    count = 0
    for rel in FILES:
        fpath = os.path.join(BASE, rel)
        if not os.path.exists(fpath):
            print(f"  SKIP (not found): {rel}")
            continue
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        ext = os.path.splitext(rel)[1]
        lang = LANG.get(ext, "text")

        out.write(f"\n## `{rel}`\n\n")
        out.write(f"{FENCE}{lang}\n")
        out.write(code)
        if not code.endswith("\n"):
            out.write("\n")
        out.write(f"{FENCE}\n\n")
        out.write("---\n\n")
        count += 1

print(f"Written {count} files to {OUT}")
print(f"Size: {os.path.getsize(OUT):,} bytes")

# Verify fences
with open(OUT, "r", encoding="utf-8") as f:
    text = f.read()
n_fences = text.count(FENCE)
print(f"Code fence markers: {n_fences} (expected {count * 2})")
