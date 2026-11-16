"""Interactive CLIP semantic search over the practice dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def load_items(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = manifest_path.parent
    items = []
    for category in manifest["categories"]:
        for item in category["items"]:
            items.append(
                {
                    "id": item["id"],
                    "category": category["name"],
                    "image": item["image"],
                    "image_path": str(dataset_root / item["image"]),
                    "caption": item["caption"],
                }
            )
    return items


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)


def encode_text(query: str, processor: CLIPProcessor, model: CLIPModel) -> np.ndarray:
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.inference_mode():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        features = model.text_projection(text_outputs.pooler_output)
    return normalize(features.cpu().numpy())[0]


def encode_images(
    items: list[dict], processor: CLIPProcessor, model: CLIPModel, batch_size: int
) -> np.ndarray:
    from PIL import Image

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            images = []
            for item in items[start : start + batch_size]:
                with Image.open(item["image_path"]) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            features = model.visual_projection(vision_outputs.pooler_output)
            embeddings.append(features.cpu().numpy())
    return normalize(np.concatenate(embeddings, axis=0))


def save_index(index_path: Path, items: list[dict], image_embeddings: np.ndarray) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        {
            "id": item["id"],
            "category": item["category"],
            "image": item["image"],
            "image_path": item["image_path"],
            "caption": item["caption"],
        }
        for item in items
    ]
    np.savez_compressed(
        index_path,
        embeddings=image_embeddings,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )


def load_or_build_index(
    manifest_path: Path,
    index_path: Path,
    clip_results_path: Path,
    processor: CLIPProcessor,
    model: CLIPModel,
    batch_size: int,
) -> tuple[list[dict], np.ndarray, bool]:
    items = load_items(manifest_path)
    expected_ids = [item["id"] for item in items]

    if index_path.is_file():
        cached = np.load(index_path, allow_pickle=False)
        cached_items = json.loads(str(cached["metadata"]))
        cached_ids = [item["id"] for item in cached_items]
        if cached_ids == expected_ids:
            return cached_items, normalize(cached["embeddings"]), True

    if clip_results_path.is_file():
        results = json.loads(clip_results_path.read_text(encoding="utf-8"))
        result_ids = [item["id"] for item in results.get("items", [])]
        if result_ids == expected_ids and "image_embeddings" in results:
            image_embeddings = normalize(np.asarray(results["image_embeddings"], dtype=np.float32))
            save_index(index_path, items, image_embeddings)
            return items, image_embeddings, False

    image_embeddings = encode_images(items, processor, model, batch_size)
    save_index(index_path, items, image_embeddings)
    return items, image_embeddings, False


def search(query: str, items: list[dict], image_embeddings: np.ndarray, processor, model, k: int):
    query_embedding = encode_text(query, processor, model)
    scores = image_embeddings @ query_embedding
    indices = np.argsort(-scores)[:k]
    return [(items[index], float(scores[index])) for index in indices]


def print_results(query: str, results: list[tuple[dict, float]]) -> None:
    print(f"\nQuery: {query}")
    print(f"Top {len(results)} results:")
    for rank, (item, score) in enumerate(results, start=1):
        print(f"{rank}. [{score:.4f}] {item['category']} — {item['image']}")
        print(f"   {item['caption']}")


def interactive_loop(items, image_embeddings, processor, model, k: int) -> None:
    print("\nCLIP semantic image search")
    print("Write an English query, ':k N' to change k, or 'exit' to finish.")
    while True:
        try:
            query = input("\nQuery> ").strip()
        except EOFError:
            print()
            return
        if query.lower() in {"exit", "quit"}:
            return
        if query.lower().startswith(":k "):
            try:
                k = max(1, min(len(items), int(query.split(maxsplit=1)[1])))
                print(f"k set to {k}")
            except (IndexError, ValueError):
                print("Use :k followed by an integer between 1 and 20.")
            continue
        if not query:
            print("Please enter a non-empty English query.")
            continue
        print_results(query, search(query, items, image_embeddings, processor, model, k))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("assets/dataset.json"))
    parser.add_argument("--index", type=Path, default=Path("results/clip_image_index.npz"))
    parser.add_argument("--clip-results", type=Path, default=Path("results/clip_results.json"))
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--query", help="Run one query and exit instead of opening the prompt.")
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
    print(
        "Image index loaded from cache."
        if loaded_from_cache
        else "Image index created and saved for reuse."
    )

    k = max(1, min(len(items), args.k))
    if args.query:
        print_results(args.query, search(args.query, items, image_embeddings, processor, model, k))
    else:
        interactive_loop(items, image_embeddings, processor, model, k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
