"""Deterministic context collector for context-first coding mode.

Measured motivation (evidence/2026-06-11_opus_vs_raidho): the waste of a pure
LLM tool-loop is not reading code — it is the LOOP, where the growing context
is re-paid on every iteration (x6 cost, x16 tokens on a real audit task).
Handing the model the same evidence in ONE call closed the quality gap at a
fraction of the price — and the single holistic read found issues the
iterative loop missed.

collect_context() gathers, for $0 and milliseconds, what the model would
otherwise discover through tool iterations: the file tree plus the contents
of task-relevant files, packed into a character budget. Relevance is a cheap
deterministic heuristic (task keywords in path/content), not a model call —
the boundary stays "structure → procedure, meaning → model".
"""
from __future__ import annotations

from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             ".idea", ".vscode", "dist", "build", ".pytest_cache", ".mypy_cache",
             ".tox", "target", ".cache"}
SKIP_SUFFIXES = {".pyc", ".so", ".dylib", ".o", ".a", ".bin", ".lock",
                 ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf",
                 ".zip", ".gz", ".tar", ".whl", ".mp3", ".mp4", ".ogg", ".npz"}
MAX_FILE_BYTES = 200_000        # skip giants — they are read targeted, via tools
MAX_FILES_SCANNED = 400


def _iter_files(workdir: Path):
    """Walk the tree, pruning noise directories; bounded by MAX_FILES_SCANNED."""
    n = 0
    stack = [workdir]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for p in entries:
            if p.is_dir():
                if p.name not in SKIP_DIRS and not p.name.endswith(".egg-info"):
                    stack.append(p)
            elif p.is_file() and p.suffix.lower() not in SKIP_SUFFIXES:
                yield p
                n += 1
                if n >= MAX_FILES_SCANNED:
                    return


def _read_text(p: Path) -> str | None:
    """File text, or None for binaries/giants/unreadables."""
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return None
        raw = p.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:4096]:
        return None
    return raw.decode("utf-8", "replace")


def _keywords(task: str) -> list[str]:
    return [w for w in {t.strip(".,;:!?()[]{}'\"`").lower() for t in task.split()}
            if len(w) > 3]


def _score(rel_path: str, text: str, kws: list[str]) -> float:
    """Cheap deterministic relevance: keyword hits in path (heavy) + content."""
    path_l, text_l = rel_path.lower(), text.lower()
    s = 0.0
    for kw in kws:
        if kw in path_l:
            s += 5.0
        s += min(text_l.count(kw), 20) * 0.5
    return s


def collect_context(workdir: str | Path, task: str,
                    char_budget: int = 24_000) -> tuple[str, dict]:
    """Deterministic context block for the task: file tree + the most relevant
    file contents packed into char_budget. Returns (block, stats).

    Files with a positive relevance score go first (best score, then smaller
    first); if budget remains, small files are added score-free — on small
    projects the model simply gets everything."""
    workdir = Path(workdir).resolve()
    kws = _keywords(task)

    candidates = []                       # (score, size, rel, text)
    tree_lines = []
    for p in _iter_files(workdir):
        rel = str(p.relative_to(workdir))
        try:
            size = p.stat().st_size
        except OSError:
            continue
        tree_lines.append(f"{rel} ({size}B)")
        text = _read_text(p)
        if text is not None:
            candidates.append((_score(rel, text, kws), size, rel, text))

    candidates.sort(key=lambda c: (-c[0], c[1]))
    scored = [c for c in candidates if c[0] > 0]
    fillers = sorted((c for c in candidates if c[0] == 0), key=lambda c: c[1])

    parts, used, included = [], 0, []
    for _, _, rel, text in scored + fillers:
        piece = f"===== {rel} =====\n{text}\n"
        if used + len(piece) > char_budget:
            continue
        parts.append(piece)
        used += len(piece)
        included.append(rel)

    omitted = len(candidates) - len(included)
    block = (
        "\n\n## Workspace context (collected deterministically — trust the file "
        "tree and contents below; use tools only for actions and for files NOT "
        "included here)\n\n"
        "### File tree\n" + "\n".join(tree_lines) +
        f"\n\n### File contents ({len(included)} files"
        + (f", {omitted} omitted by budget — read them via tools if needed" if omitted else "")
        + ")\n\n" + "".join(parts)
    )
    return block, {"files_included": len(included), "files_omitted": omitted,
                   "chars": len(block)}
