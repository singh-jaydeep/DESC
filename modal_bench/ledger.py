"""Append-only run ledger.

Every measurement is written to disk the moment it arrives, so a sweep that is
interrupted, rate-limited, or killed still leaves a complete record of what ran,
on what hardware, and what it produced. Two files:

* ``ledger.md``   -- human-readable, append-only, one block per batch
* ``ledger.jsonl`` -- one JSON object per measurement, for later analysis

Nothing here ever rewrites history: results are appended, never replaced.
"""

import json
import pathlib
import subprocess
from datetime import datetime, timezone

LEDGER_DIR = pathlib.Path(__file__).parent
MD = LEDGER_DIR / "ledger.md"
JSONL = LEDGER_DIR / "ledger.jsonl"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def git_sha():
    """Short SHA of HEAD, or 'unknown'."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=LEDGER_DIR.parent,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def dirty():
    """True if desc/ has uncommitted changes."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", "desc/"],
            capture_output=True,
            text=True,
            cwd=LEDGER_DIR.parent,
        ).stdout.strip()
        return bool(out)
    except Exception:
        return None


def open_section(title, meta=None):
    """Start a ledger block. Returns the header string that was written."""
    meta = dict(meta or {})
    meta.setdefault("git", git_sha())
    meta.setdefault("desc_dirty", dirty())
    head = [f"\n\n## {title}", f"*{_now()}*", ""]
    head += [f"- **{k}**: {v}" for k, v in meta.items()]
    head += [""]
    text = "\n".join(head)
    with MD.open("a") as fh:
        fh.write(text)
    return text


def row(record, line):
    """Append one measurement: a formatted line to the md, the object to jsonl."""
    with MD.open("a") as fh:
        fh.write(line.rstrip() + "\n")
    with JSONL.open("a") as fh:
        fh.write(json.dumps({"ts": _now(), **record}) + "\n")


def note(text):
    """Append a raw line to the markdown ledger."""
    with MD.open("a") as fh:
        fh.write(text.rstrip() + "\n")


def table_header(cols):
    """Write a markdown table header (kept inside a fenced block for alignment)."""
    note("```")
    note(cols)
    note("-" * len(cols))


def table_end():
    """Close the fenced block."""
    note("```")
