#!/usr/bin/env python3
"""Send images or PDF pages to an Unlimited-OCR vLLM server.

The server-side decode recipe is not optional. Three pieces must be right or the
model returns empty output or loops forever:

  * the prompt must begin with the literal "<image>"
  * skip_special_tokens must be False (grounding tokens are part of the output)
  * ngram_size / window_size must be passed per request via vllm_xargs

Usage:
    python ocr_client.py page.png
    python ocr_client.py doc.pdf --output-dir out
    python ocr_client.py scans/ --concurrency 8
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Single image uses gundam (crop) mode with a 128-token window; multi-image
# requests fall back to base mode and need the wider 1024 window.
NGRAM_SIZE = 35
WINDOW_SINGLE = 128
WINDOW_MULTI = 1024

# Two grounding shapes occur in the wild. The checkpoint emits paired markers
# (modeling_unlimitedocr.py:45 uses
# r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'), while the README's
# OmniDocBench snippet shows the category inside <|det|> instead. Handle both:
# the block category is the <|ref|> text when present, else the leading word of
# the <|det|> payload.
BLOCK_RE = re.compile(
    r"^(?:<\|ref\|>(.*?)<\|/ref\|>)?\s*<\|det\|>(.*?)<\|/det\|>\s*(.*)$", re.DOTALL
)
# Leftovers appearing mid-line rather than at a block start.
INLINE_REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
INLINE_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)


def _clean(text: str) -> str:
    """Unwrap inline <|ref|> text and drop inline <|det|> coordinate spans."""
    return INLINE_DET_RE.sub("", INLINE_REF_RE.sub(r"\1", text)).strip()


def remove_det(raw: str) -> str:
    """Drop grounding tokens, join lines in a block, blank-line between blocks."""
    blocks: list[list[str]] = []
    cur: list[str] | None = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            # A blank line ends the current block. Without this, ungrounded
            # output (no <|det|> markers at all) collapses into one block.
            if cur:
                blocks.append(cur)
            cur = None
            continue
        m = BLOCK_RE.match(line)
        if m:
            ref, det, rest = m.group(1), m.group(2), m.group(3)
            # Category lives in <|ref|>, or as the first word of <|det|>.
            category = (ref or det.split("[", 1)[0]).strip().lower()
            if category == "image":
                # Figure region: no text to keep, and its caption is a
                # separate block.
                continue
            if cur is not None:
                blocks.append(cur)
            content = _clean(rest)
            cur = [content] if content else []
            continue
        content = _clean(line)
        if not content:
            continue
        if cur is None:
            cur = []
        cur.append(content)
    if cur is not None:
        blocks.append(cur)
    return "\n\n".join("\n".join(b) for b in blocks if b).strip()


def pdf_to_images(pdf_path: Path, dpi: int) -> list[Path]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("PDF input needs PyMuPDF: pip install pymupdf")

    doc = fitz.open(pdf_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="unlimited_ocr_"))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for i, page in enumerate(doc):
        out = tmp_dir / f"page_{i + 1:04d}.png"
        page.get_pixmap(matrix=mat).save(out)
        pages.append(out)
    doc.close()
    return pages


def encode_image(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def ocr_one(
    session: requests.Session,
    base_url: str,
    model: str,
    image: Path,
    window_size: int,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                # The literal "<image>" prefix is mandatory.
                "content": [
                    {"type": "text", "text": "<image>document parsing."},
                    encode_image(image),
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "skip_special_tokens": False,
        "vllm_xargs": {"ngram_size": NGRAM_SIZE, "window_size": window_size},
    }
    resp = session.post(
        f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"] or ""


def collect_inputs(target: Path, dpi: int) -> tuple[list[Path], int]:
    """Return (images, window_size)."""
    if target.is_dir():
        images = sorted(
            p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            sys.exit(f"No images ({', '.join(IMAGE_EXTS)}) under {target}")
        return images, WINDOW_SINGLE
    if target.suffix.lower() == ".pdf":
        return pdf_to_images(target, dpi), WINDOW_MULTI
    if target.suffix.lower() in IMAGE_EXTS:
        return [target], WINDOW_SINGLE
    sys.exit(f"Unsupported input: {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="image, PDF, or directory of images")
    ap.add_argument("--base-url", default=os.environ.get("OCR_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default="baidu/Unlimited-OCR")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--output-dir", type=Path, default=None, help="write .md per page")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--raw", action="store_true", help="keep <|det|> grounding tokens")
    args = ap.parse_args()

    images, window_size = collect_inputs(args.input, args.dpi)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.trust_env = False
    if args.api_key:
        session.headers["Authorization"] = f"Bearer {args.api_key}"

    print(
        f"{len(images)} page(s), window_size={window_size}, "
        f"concurrency={args.concurrency} -> {args.base_url}",
        file=sys.stderr,
    )

    started = time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                ocr_one, session, args.base_url, args.model, img,
                window_size, args.max_tokens, args.timeout,
            ): img
            for img in images
        }
        for future in as_completed(futures):
            img = futures[future]
            try:
                raw = future.result()
            except Exception as exc:  # one bad page must not sink the batch
                failures += 1
                print(f"FAIL {img.name}: {exc}", file=sys.stderr)
                continue
            text = raw if args.raw else remove_det(raw)
            if args.output_dir:
                out = args.output_dir / f"{img.stem}.md"
                out.write_text(text, encoding="utf-8")
                print(f"ok   {img.name} -> {out} ({len(text)} chars)", file=sys.stderr)
            else:
                print(text)

    elapsed = time.time() - started
    print(
        f"{len(images) - failures}/{len(images)} page(s) in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
