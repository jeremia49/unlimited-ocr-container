#!/usr/bin/env python3
"""Send images or PDF pages to an Unlimited-OCR vLLM server.

The server-side decode recipe is not optional. Three pieces must be right or the
model returns empty output or loops forever:

  * the prompt must begin with the literal "<image>"
  * skip_special_tokens must be False (grounding tokens are part of the output)
  * ngram_size / window_size must be passed per request via vllm_xargs

Figure regions are croppable: the model grounds every block with a bounding box,
so --extract-figures saves each image/chart as its own PNG plus a JSON manifest
ready to route to a vision LLM.

Usage:
    python ocr_client.py page.png
    python ocr_client.py doc.pdf --output-dir out
    python ocr_client.py doc.pdf --output-dir out --extract-figures
    python ocr_client.py scans/ --concurrency 8
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


class FatalRequestError(RuntimeError):
    """A 4xx from the server: the request itself is wrong, so do not retry."""


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# One page per request means every request is single-image, i.e. gundam (crop)
# mode with a 128-token n-gram window. The 1024 window only applies when several
# images share one request, which this client never does.
NGRAM_SIZE = 35
WINDOW_SINGLE = 128

# Region categories the model actually emits, censused over every <|det|> marker
# of a 14-page paper: header, title, text, image, image_caption, page_number,
# equation, chart, table, ref_text. Note that "chart" is a figure too -- dropping
# only "image" leaks vector charts into the text stream.
FIGURE_CATS = frozenset({"image", "chart", "figure", "diagram", "graph", "plot"})
CAPTION_CATS = frozenset(
    {"image_caption", "figure_caption", "chart_caption", "table_caption"}
)

# <|det|> boxes are normalized per axis to 0..1000, NOT pixels. Verified by
# denormalizing against PyMuPDF text-block rects on a 595x842 pt page rendered at
# 300 DPI (2481x3508 px): every block with a 1:1 counterpart landed within
# 5-10 pt of ground truth.
DET_SCALE = 1000.0

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
# Any marker anywhere. Several can sit back-to-back on one line: a chart and its
# caption are emitted as "<|det|>chart [...]<|/det|><|det|>image_caption [...]".
ANY_DET_RE = re.compile(r"<\|det\|>([^\[\]]*?)\s*\[([^\]]*)\]\s*<\|/det\|>")


def _box(payload: str) -> tuple[int, int, int, int] | None:
    """Pull the 4 leading integers out of a <|det|> payload."""
    nums = [int(n) for n in re.findall(r"-?\d+", payload)[:4]]
    return (nums[0], nums[1], nums[2], nums[3]) if len(nums) == 4 else None


def _clean(text: str) -> str:
    """Unwrap inline <|ref|> text and drop inline <|det|> coordinate spans."""
    return INLINE_DET_RE.sub("", INLINE_REF_RE.sub(r"\1", text)).strip()


def parse_regions(raw: str) -> list[dict]:
    """Every grounded region in emission order: category, box, trailing text."""
    marks = list(ANY_DET_RE.finditer(raw))
    regions = []
    for i, m in enumerate(marks):
        box = _box(m.group(2))
        if box is None:
            continue
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        regions.append(
            {
                "category": m.group(1).strip().lower(),
                "box": box,
                "text": _clean(raw[m.end() : stop]),
            }
        )
    return regions


def denormalize(box, width: int, height: int) -> tuple[int, int, int, int]:
    """0..1000 per-axis box -> pixel box in an image of the given size."""
    x1, y1, x2, y2 = box
    return (
        round(x1 / DET_SCALE * width),
        round(y1 / DET_SCALE * height),
        round(x2 / DET_SCALE * width),
        round(y2 / DET_SCALE * height),
    )


def remove_det(raw: str, figure_links: dict | None = None) -> str:
    """Drop grounding tokens, join lines in a block, blank-line between blocks.

    A figure region carries no text. When ``figure_links`` maps its box to a
    relative path, emit a Markdown image reference in its place so the document
    keeps the figure's position; otherwise drop it.
    """
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
            if cur is not None:
                blocks.append(cur)
            cur = None
            # A nested caption marker may trail a figure on the same line.
            content = _clean(rest)
            if category in FIGURE_CATS:
                box = _box(det)
                link = (figure_links or {}).get(box)
                if link:
                    alt = content or category
                    blocks.append([f"![{alt}]({link})"])
                if content:
                    cur = [content]
                continue
            if content:
                cur = [content]
            else:
                cur = []
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


# One requests.Session shared across threads corrupts TLS records through a
# Cloudflare tunnel (observed: SSLV3_ALERT_BAD_RECORD_MAC under 4 workers). Give
# every worker its own session and drop a poisoned pool on failure.
_local = threading.local()


def _session(api_key: str) -> requests.Session:
    session = getattr(_local, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        if api_key:
            session.headers["Authorization"] = f"Bearer {api_key}"
        _local.session = session
    return session


def ocr_one(
    base_url: str,
    api_key: str,
    model: str,
    image: Path,
    window_size: int,
    max_tokens: int,
    timeout: int,
    retries: int = 4,
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
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _session(api_key).post(url, json=payload, timeout=timeout)
            if 400 <= resp.status_code < 500:
                # Contract error (bad params, too many tokens); retrying cannot fix it.
                raise FatalRequestError(f"{resp.status_code}: {resp.text[:400]}")
            if resp.status_code >= 500:
                raise RuntimeError(f"{resp.status_code}: {resp.text[:200]}")
            return resp.json()["choices"][0]["message"]["content"] or ""
        except FatalRequestError:
            raise
        except Exception as exc:
            last = exc
        _local.session = None
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gave up after {retries} attempts: {last}")


def crop_figures(
    page_image: Path, regions: list[dict], out_dir: Path, stem: str, pad: int = 6
) -> list[dict]:
    """Save every figure region as its own PNG. Returns manifest entries.

    Figures in a paper are often vector drawings, not embedded raster images, so
    pulling them out of the PDF object tree misses them. Cropping the rendered
    page from the model's own boxes catches both.
    """
    try:
        from PIL import Image
    except ImportError:
        sys.exit("--extract-figures needs Pillow: pip install pillow")

    figures = [r for r in regions if r["category"] in FIGURE_CATS]
    if not figures:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    captions = [r for r in regions if r["category"] in CAPTION_CATS]
    with Image.open(page_image) as src:
        width, height = src.size
        entries = []
        for n, region in enumerate(figures, 1):
            x1, y1, x2, y2 = denormalize(region["box"], width, height)
            box = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(width, x2 + pad),
                min(height, y2 + pad),
            )
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue  # degenerate box, nothing to route to an LLM
            path = out_dir / f"{stem}_fig{n:02d}_{region['category']}.png"
            src.crop(box).save(path)
            entries.append(
                {
                    "path": path,
                    "category": region["category"],
                    "box_norm": list(region["box"]),
                    "box_px": list(box),
                    # Caption text may trail the figure marker, or arrive as a
                    # separate caption region; prefer the inline one.
                    "caption": region["text"] or _nearest_caption(region, captions),
                }
            )
    return entries


def _nearest_caption(figure: dict, captions: list[dict]) -> str:
    """Caption whose box sits closest below the figure, if any."""
    fx1, _, fx2, fy2 = figure["box"]
    best, best_gap = "", 10**9
    for cap in captions:
        cx1, cy1, cx2, _ = cap["box"]
        overlap = min(fx2, cx2) - max(fx1, cx1)
        gap = cy1 - fy2
        if overlap > 0 and 0 <= gap < best_gap:
            best, best_gap = cap["text"], gap
    return best


def collect_inputs(target: Path, dpi: int) -> tuple[list[Path], int]:
    """Return (images, window_size).

    window_size follows how many images go into ONE request, not how many pages
    the job has. This client sends one page per request, so every case is a
    single-image (gundam) request and uses WINDOW_SINGLE. Keeping pages in
    separate requests is what lets them run concurrently.
    """
    if target.is_dir():
        images = sorted(
            p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not images:
            sys.exit(f"No images ({', '.join(IMAGE_EXTS)}) under {target}")
        return images, WINDOW_SINGLE
    if target.suffix.lower() == ".pdf":
        return pdf_to_images(target, dpi), WINDOW_SINGLE
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
    # max_tokens + prompt_tokens must stay under max_model_len (32768). A 300 DPI
    # A4 page costs ~2.7k prompt tokens, so 32768 here is rejected with HTTP 400.
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--raw", action="store_true", help="keep <|det|> grounding tokens")
    ap.add_argument(
        "--extract-figures",
        action="store_true",
        help="crop image/chart regions to PNGs plus figures.json (needs --output-dir)",
    )
    ap.add_argument(
        "--figure-pad",
        type=int,
        default=6,
        help="pixels of padding around each cropped figure",
    )
    args = ap.parse_args()

    if args.extract_figures and not args.output_dir:
        ap.error("--extract-figures requires --output-dir")

    images, window_size = collect_inputs(args.input, args.dpi)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = (args.output_dir / "figures") if args.extract_figures else None

    print(
        f"{len(images)} page(s), window_size={window_size}, "
        f"concurrency={args.concurrency} -> {args.base_url}",
        file=sys.stderr,
    )

    started = time.time()
    # Indexed, not append-order: pages finish out of order under concurrency and
    # the document must stay in page order.
    pages: list[str | None] = [None] * len(images)
    figures: list[list[dict]] = [[] for _ in images]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                ocr_one, args.base_url, args.api_key, args.model, img,
                window_size, args.max_tokens, args.timeout,
            ): i
            for i, img in enumerate(images)
        }
        for future in as_completed(futures):
            i = futures[future]
            img = images[i]
            try:
                raw = future.result()
            except Exception as exc:  # one bad page must not sink the batch
                print(f"FAIL p{i + 1} {img.name}: {exc}", file=sys.stderr)
                continue

            links: dict = {}
            if figures_dir is not None:
                entries = crop_figures(
                    img, parse_regions(raw), figures_dir, f"page_{i + 1:04d}",
                    pad=args.figure_pad,
                )
                for entry in entries:
                    entry["page"] = i + 1
                    # Link relative to the .md files, which sit in output_dir.
                    links[tuple(entry["box_norm"])] = (
                        f"figures/{entry['path'].name}"
                    )
                figures[i] = entries

            text = raw if args.raw else remove_det(raw, links)
            pages[i] = text
            note = f", {len(figures[i])} figure(s)" if figures[i] else ""
            print(f"ok   p{i + 1} {img.name} ({len(text)} chars{note})", file=sys.stderr)

    failures = sum(1 for p in pages if p is None)

    if args.output_dir:
        for i, text in enumerate(pages):
            if text is None:
                continue
            (args.output_dir / f"{images[i].stem}.md").write_text(text, encoding="utf-8")
        # Full document in page order. A failed page leaves a visible marker
        # rather than silently shrinking the output.
        combined = "\n\n".join(
            pages[i] if pages[i] is not None else f"<!-- page {i + 1} FAILED -->"
            for i in range(len(pages))
        )
        full = args.output_dir / f"{args.input.stem}.md"
        full.write_text(combined, encoding="utf-8")
        print(f"full document -> {full} ({len(combined)} chars)", file=sys.stderr)

        if figures_dir is not None:
            flat = [e for page in figures for e in page]
            manifest = args.output_dir / "figures.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "page": e["page"],
                            "category": e["category"],
                            "file": f"figures/{e['path'].name}",
                            "caption": e["caption"],
                            "box_norm": e["box_norm"],
                            "box_px": e["box_px"],
                        }
                        for e in flat
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(
                f"{len(flat)} figure(s) -> {figures_dir} (manifest {manifest})",
                file=sys.stderr,
            )
    else:
        for text in pages:
            if text is not None:
                print(text)

    elapsed = time.time() - started
    print(
        f"{len(images) - failures}/{len(images)} page(s) in {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
