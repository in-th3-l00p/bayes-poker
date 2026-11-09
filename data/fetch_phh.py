"""Selective downloader for the PHH poker hand-history dataset (Zenodo).

The full dataset is a single ~20 GB zip. Zenodo supports HTTP range requests,
so this script reads only the zip's central directory, then downloads just the
members you ask for. Standard library only.

Usage:
    python3 data/fetch_phh.py list [--grep PATTERN] [--limit N]
    python3 data/fetch_phh.py summary            # file counts / sizes per top-level folder
    python3 data/fetch_phh.py get PATTERN [--max-files N] [--out data/raw]

PATTERN is an fnmatch glob on the path inside the zip, e.g.
    'poker-hand-histories/data/acpc/2011/*limit*'
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import io
import pathlib
import sys
import urllib.request
import zipfile

RECORD_API = "https://zenodo.org/api/records/10796885"  # concept DOI -> latest version
DEFAULT_OUT = pathlib.Path(__file__).resolve().parent / "raw"


def resolve_zip_url() -> str:
    import json

    with urllib.request.urlopen(RECORD_API) as r:
        rec = json.load(r)
    for f in rec["files"]:
        if f["key"].endswith(".zip"):
            return f["links"]["self"]
    raise RuntimeError("no zip file found in Zenodo record")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP using Range requests, with a block cache."""

    BLOCK = 1 << 20  # 1 MiB

    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self._cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    def _block(self, idx: int) -> bytes:
        if idx not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            start = idx * self.BLOCK
            self._cache[idx] = self._fetch(start, min(start + self.BLOCK, self.size))
        return self._cache[idx]

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n > 4 * self.BLOCK:  # large read: fetch directly, skip cache
            data = self._fetch(self.pos, self.pos + n)
            self.pos += len(data)
            return data
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.BLOCK)
            chunk = self._block(idx)[off : off + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip() -> zipfile.ZipFile:
    url = resolve_zip_url()
    print(f"[phh] remote zip: {url}", file=sys.stderr)
    return zipfile.ZipFile(HTTPRangeFile(url))


def cmd_list(z: zipfile.ZipFile, grep: str | None, limit: int) -> None:
    shown = 0
    for info in z.infolist():
        if grep and not fnmatch.fnmatch(info.filename, grep):
            continue
        print(f"{info.file_size:>14,d}  {info.filename}")
        shown += 1
        if limit and shown >= limit:
            break


def cmd_summary(z: zipfile.ZipFile, depth: int) -> None:
    count: collections.Counter[str] = collections.Counter()
    size: collections.Counter[str] = collections.Counter()
    for info in z.infolist():
        if info.is_dir():
            continue
        key = "/".join(info.filename.split("/")[:depth])
        count[key] += 1
        size[key] += info.file_size
    for key in sorted(count):
        print(f"{count[key]:>8,d} files  {size[key] / 1e9:>8.2f} GB  {key}")


def cmd_get(z: zipfile.ZipFile, pattern: str, max_files: int, out: pathlib.Path) -> None:
    members = [i for i in z.infolist() if not i.is_dir() and fnmatch.fnmatch(i.filename, pattern)]
    if max_files:
        members = members[:max_files]
    total = sum(m.file_size for m in members)
    print(f"[phh] {len(members)} files, {total / 1e6:.1f} MB uncompressed", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)
    for m in members:
        target = (out / m.filename).resolve()
        if not str(target).startswith(str(out.resolve())):  # zip-slip guard
            raise RuntimeError(f"unsafe path in zip: {m.filename}")
        if target.exists() and target.stat().st_size == m.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(m) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        print(f"  {m.file_size:>12,d}  {m.filename}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list")
    pl.add_argument("--grep")
    pl.add_argument("--limit", type=int, default=200)
    ps = sub.add_parser("summary")
    ps.add_argument("--depth", type=int, default=4)
    pg = sub.add_parser("get")
    pg.add_argument("pattern")
    pg.add_argument("--max-files", type=int, default=0)
    pg.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    a = p.parse_args()

    z = open_remote_zip()
    if a.cmd == "list":
        cmd_list(z, a.grep, a.limit)
    elif a.cmd == "summary":
        cmd_summary(z, a.depth)
    else:
        cmd_get(z, a.pattern, a.max_files, a.out)


if __name__ == "__main__":
    main()
