"""Verify every captioned code listing in the lesson guide still matches source.

A guide whose code has drifted from the repository is worse than no guide: the
reader hits a wall and cannot tell whether they mistyped or the book is stale.
This checks each line of every ```python block that sits under a
`Filename: <strong>x.py</strong>` caption against that file.

Line-wrapping and comment differences produce false positives, so the output is
a review list, not a gate. Exits non-zero only when a captioned file is missing.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "lesson/content/chapters"
CAPTION = re.compile(r'<p class="filename">Filename: <strong>(.*?)</strong></p>', re.S)

# Part II teaches the pre-unfreeze build; the repo carries the finished form.
STAGED = {
    "torch.save(self.head.state_dict(), self.checkpoint_file)",
    "torch.load(path or self.checkpoint_file, map_location=self.device))",
    "self.optimizer = Adam(self.model.head.parameters(), lr=learning_rate)",
    "self.dataset = Dataset(max_size=max_buffer_size,",
}


def listings():
    for chapter in sorted(CHAPTERS.glob("*.md")):
        filename = None
        for chunk in re.split(r'(<p class="filename">.*?</p>)', chapter.read_text(), flags=re.S):
            caption = CAPTION.match(chunk)
            if caption:
                filename = caption.group(1).strip()
                continue
            for block in re.finditer(r"```python\n(.*?)```", chunk, re.S):
                if filename:
                    yield chapter.name, filename, block.group(1)


TEACHES = ["model.py", "agent.py"]
SKIP_PREFIX = ("#", "import ", "from ")


def coverage():
    """How much of each taught file appears in some listing.

    A line the guide never shows is a line the reader has to invent. Staged
    differences (Part II teaches the pre-unfreeze build) are excluded.
    """
    shown = set()
    for _, _, code in listings():
        shown.update(line.strip() for line in code.splitlines() if line.strip())

    for src in TEACHES:
        total = gaps = 0
        for line in (ROOT / src).read_text().splitlines():
            text = line.strip()
            if (not text or text.startswith(SKIP_PREFIX) or text[:3] in ('"""', "'''")
                    or text in STAGED):
                continue
            total += 1
            gaps += text not in shown
        pct = 100 * (total - gaps) // total if total else 100
        print(f"coverage {src:<12} {total - gaps:>3}/{total} lines shown ({pct}%)")


def main():
    drift, missing = [], []
    for chapter, filename, code in listings():
        source = ROOT / filename
        if not source.exists():
            missing.append(f"{chapter}: no such file {filename}")
            continue
        text = source.read_text()
        for line in code.splitlines():
            line = line.strip()
            if (not line or line.startswith("#") or line.startswith('"""')
                    or line.endswith("...)") or line in ("...", ")") or line in STAGED):
                continue
            if line not in text:
                drift.append(f"{chapter} [{filename}]: {line[:70]}")

    for m in missing:
        print(f"MISSING  {m}")
    for d in drift:
        print(f"review   {d}")
    print()
    coverage()
    print(f"\n{len(drift)} lines to review, {len(missing)} missing files")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
