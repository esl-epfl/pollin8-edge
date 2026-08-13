"""Download + verify the Bjerge et al. (2022) benchmark from Zenodo.

FAIR: data is fetched from the original DOI, never redistributed. md5 sums are
the published ones, so a mismatch fails loudly (no silent corruption).
"""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
import urllib.request

ZENODO = "https://zenodo.org/records/7395752/files"
FILES = {  # name: published md5
    "train1201.zip": "6831b05cab0988743a113819eb23be75",
    "val1201.zip":   "88317db11fd10fab4976edb4d8d4a71f",
    "test1201.zip":  "d940cac65cf067a3baf356ecaa9944e3",
    "YOLOv5models.zip": "bc2194e94bfbe0ba93e4a66df6eb6f1b",
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> Path:
    """Idempotent + resumable. Already-verified files are skipped instantly (via a
    `.verified` marker, so we don't re-hash 14 GB on every rerun). A partially
    downloaded file is RESUMED with an HTTP Range request, not restarted."""
    out = dest / name
    marker = out.with_suffix(out.suffix + ".verified")
    url = f"{ZENODO}/{name}?download=1"

    # 1) fast path: verified marker present and file still there -> skip, no re-hash
    if marker.exists() and out.exists():
        print(f"[ok] {name} already verified (skip)")
        return out
    # 2) file present but unmarked: hash once; if good, mark and skip
    if out.exists() and md5(out) == FILES[name]:
        marker.write_text("ok")
        print(f"[ok] {name} present and verified")
        return out

    # 3) (re)download — resume from the partial byte offset if any
    resume = out.stat().st_size if out.exists() else 0
    req = urllib.request.Request(url)
    if resume:
        req.add_header("Range", f"bytes={resume}-")
        print(f"[..] resuming {name} from {resume/1e6:.0f} MB")
    else:
        print(f"[..] downloading {name}")
    with urllib.request.urlopen(req) as r:
        partial = resume and getattr(r, "status", 200) == 206  # 206 => server honoured Range
        with out.open("ab" if partial else "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)

    got = md5(out)
    if got != FILES[name]:
        raise SystemExit(f"[md5 MISMATCH] {name}: got {got} expected {FILES[name]} "
                         f"-- delete {out} and re-run")
    marker.write_text("ok")
    print(f"[ok] {name} verified")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", nargs="*", choices=list(FILES), help="subset to fetch")
    a = ap.parse_args(argv)
    a.dest.mkdir(parents=True, exist_ok=True)
    for name in (a.only or FILES):
        fetch(name, a.dest)
    print("[done] all files verified ->", a.dest)


if __name__ == "__main__":
    main()
