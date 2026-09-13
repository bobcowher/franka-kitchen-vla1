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
    print(f"\n{len(drift)} lines to review, {len(missing)} missing files")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
