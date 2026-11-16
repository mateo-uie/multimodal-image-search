"""Small web interface for the CLIP semantic image search."""

from __future__ import annotations

import argparse
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from transformers import CLIPModel, CLIPProcessor

from search import DEFAULT_CLIP_MODEL, load_or_build_index, search


PAGE_STYLE = """
:root { font-family: system-ui, sans-serif; color: #172033; background: #f4f6fb; }
body { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem 4rem; }
h1 { margin-bottom: .3rem; }
.subtitle { color: #5b6578; margin-top: 0; }
form { display: flex; gap: .7rem; flex-wrap: wrap; background: white; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 10px #17203318; }
input, select, button { font: inherit; padding: .65rem .8rem; border: 1px solid #ccd3e0; border-radius: 8px; }
input[name=query] { flex: 1 1 320px; }
button { color: white; background: #3157d5; border-color: #3157d5; cursor: pointer; }
.results { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; margin-top: 1.5rem; }
.card { overflow: hidden; background: white; border-radius: 12px; box-shadow: 0 2px 10px #17203318; }
.card img { display: block; width: 100%; height: 190px; object-fit: cover; }
.card-content { padding: 1rem; }
.rank { color: #3157d5; font-weight: 700; }
.score { float: right; font-weight: 700; }
.category { display: inline-block; margin: .5rem 0; padding: .2rem .5rem; color: #3157d5; background: #e9edff; border-radius: 999px; font-size: .85rem; }
.caption { line-height: 1.4; margin-bottom: 0; }
.error { padding: 1rem; color: #9c2c2c; background: #ffe8e8; border-radius: 8px; }
"""


def render_page(query: str, k: int, results: list[tuple[dict, float]] | None, error: str = "") -> bytes:
    cards = ""
    if results is not None:
        cards = "<section class='results'>"
        for rank, (item, score) in enumerate(results, start=1):
            image_url = "/assets/" + item["image"]
            cards += f"""
            <article class="card">
              <img src="{html.escape(image_url, quote=True)}" alt="{html.escape(item['caption'], quote=True)}">
              <div class="card-content">
                <span class="rank">#{rank}</span>
                <span class="score">{score:.4f}</span>
                <span class="category">{html.escape(item['category'])}</span>
                <p class="caption">{html.escape(item['caption'])}</p>
              </div>
            </article>
            """
        cards += "</section>"
    error_html = f"<p class='error'>{html.escape(error)}</p>" if error else ""
    options = "".join(
        f'<option value="{number}"{" selected" if number == k else ""}>{number}</option>'
        for number in range(1, 21)
    )
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CLIP Image Search</title><style>{PAGE_STYLE}</style></head>
<body>
<h1>CLIP Image Search</h1>
<p class="subtitle">Search the 20-image dataset using a natural-language query in English.</p>
<form method="get" action="/">
  <input name="query" value="{html.escape(query, quote=True)}" placeholder="e.g. a small dog" autofocus>
  <label>Results <select name="k">{options}</select></label>
  <button type="submit">Search</button>
</form>
{error_html}
{cards}
</body></html>"""
    return document.encode("utf-8")


class SearchHandler(SimpleHTTPRequestHandler):
    items = []
    image_embeddings = None
    processor = None
    model = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/":
            super().do_GET()
            return

        values = parse_qs(parsed.query)
        query = values.get("query", [""])[0].strip()
        try:
            k = max(1, min(len(self.items), int(values.get("k", ["5"])[0])))
        except ValueError:
            k = 5

        results = None
        error = ""
        if query:
            try:
                results = search(
                    query,
                    self.items,
                    self.image_embeddings,
                    self.processor,
                    self.model,
                    k,
                )
            except Exception as exc:  # pragma: no cover - defensive UI fallback
                error = f"Search failed: {exc}"

        body = render_page(query, k, results, error)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("assets/dataset.json"))
    parser.add_argument("--index", type=Path, default=Path("results/clip_image_index.npz"))
    parser.add_argument("--clip-results", type=Path, default=Path("results/clip_results.json"))
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained(args.model)
    model = CLIPModel.from_pretrained(args.model).eval()
    items, image_embeddings, loaded_from_cache = load_or_build_index(
        args.manifest,
        args.index,
        args.clip_results,
        processor,
        model,
        args.batch_size,
    )
    SearchHandler.items = items
    SearchHandler.image_embeddings = image_embeddings
    SearchHandler.processor = processor
    SearchHandler.model = model

    project_root = Path.cwd()
    handler = lambda *handler_args, **handler_kwargs: SearchHandler(
        *handler_args, directory=str(project_root), **handler_kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    cache_message = "cache" if loaded_from_cache else "new index"
    print(f"CLIP web search ready at http://localhost:{args.port} ({cache_message})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
