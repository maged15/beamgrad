#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write pip find-links pages for the wheels attached to beamgrad's GitHub releases.

Each wheel's local version label names the PyTorch minor version and CUDA
variant it was built for (`2.1.0+pt214cu126`). This writes one page per label,
`whl/<label>.html`, linking every release's wheels with that label, so that

    pip install beamgrad -f https://<owner>.github.io/beamgrad/whl/pt214cu126.html

installs the wheel for the platform, and `index.html` with the variants and a
command that picks the page from the installed PyTorch. The pages only link to
the release assets; the wheels stay on GitHub releases.

    GH_TOKEN=... python scripts/wheel_index.py --repo maged15/beamgrad --out site
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

WHEEL = re.compile(r"^beamgrad-(?P<version>[^-+]+)\+(?P<label>pt\d+(?:cpu|cu\d+))-[^/]+\.whl$")


def releases(repo: str) -> list[dict]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    out, page = [], 1
    while True:
        request = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}")
        request.add_header("Accept", "application/vnd.github+json")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request) as response:
            batch = json.load(response)
        out.extend(r for r in batch if not r["draft"])
        if len(batch) < 100:
            return out
        page += 1


def page(title: str, body: str) -> str:
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head><meta charset="utf-8"><title>{html.escape(title)}</title></head>\n'
        f"<body>\n{body}\n</body>\n</html>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    by_label: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for release in releases(args.repo):
        for asset in release["assets"]:
            match = WHEEL.match(asset["name"])
            if match:
                # %2B for the "+" of the local version, as download.pytorch.org writes it.
                url = asset["browser_download_url"].replace("+", "%2B")
                by_label[match["label"]].append((asset["name"], url))

    owner, name = args.repo.split("/")
    base = f"https://{owner}.github.io/{name}"
    (args.out / "whl").mkdir(parents=True, exist_ok=True)
    for label, wheels in sorted(by_label.items()):
        links = "\n".join(f'<a href="{html.escape(url)}">{html.escape(file)}</a><br>' for file, url in sorted(wheels))
        (args.out / "whl" / f"{label}.html").write_text(page(f"beamgrad wheels: {label}", links))

    rows = "\n".join(
        f'<li><a href="whl/{label}.html">{label}</a>: {len(wheels)} wheels</li>'
        for label, wheels in sorted(by_label.items())
    )
    label_command = (
        "python -c \"import torch; v = torch.__version__.split('+')[0].split('.'); c = torch.version.cuda; "
        "print(f'pt{v[0]}{v[1]}' + ('cu' + c.replace('.', '') if c else 'cpu'))\""
    )
    body = f"""<h1>beamgrad wheels</h1>
<p>Prebuilt wheels of <a href="https://github.com/{args.repo}">beamgrad</a>, one page per PyTorch minor version and
CUDA variant. Install PyTorch first, then:</p>
<pre>pip install beamgrad -f {base}/whl/$({html.escape(label_command)}).html</pre>
<p>If there is no page for your PyTorch, pip builds beamgrad from the source distribution on PyPI instead
(see <a href="https://github.com/{args.repo}/blob/main/docs/installation.md">installation</a>).</p>
<ul>
{rows}
</ul>"""
    (args.out / "index.html").write_text(page("beamgrad wheels", body))
    (args.out / ".nojekyll").write_text("")
    print(f"{sum(len(w) for w in by_label.values())} wheels, {len(by_label)} variants: {', '.join(sorted(by_label))}")


if __name__ == "__main__":
    main()
