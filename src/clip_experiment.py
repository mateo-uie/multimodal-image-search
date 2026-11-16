"""Run the CLIP image-caption matching experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
                    "image_path": dataset_root / item["image"],
                    "caption": item["caption"],
                }
            )
    return items


def normalize(embeddings: np.ndarray) -> np.ndarray:
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)


def encode_images(
    items: list[dict], processor: CLIPProcessor, model: CLIPModel, batch_size: int
) -> np.ndarray:
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


def encode_texts(
    items: list[dict], processor: CLIPProcessor, model: CLIPModel, batch_size: int
) -> np.ndarray:
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            texts = [item["caption"] for item in items[start : start + batch_size]]
            inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            features = model.text_projection(text_outputs.pooler_output)
            embeddings.append(features.cpu().numpy())
    return normalize(np.concatenate(embeddings, axis=0))


def matching_metrics(similarities: np.ndarray) -> dict[str, float]:
    reciprocal_ranks = []
    top1 = 0
    ranks = []
    for image_index, row in enumerate(similarities):
        order = np.argsort(-row)
        rank = int(np.where(order == image_index)[0][0]) + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
        top1 += rank == 1
    return {
        "top1_accuracy": top1 / len(similarities),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "mean_correct_caption_rank": float(np.mean(ranks)),
    }


def category_metrics(items: list[dict], similarities: np.ndarray) -> dict[str, float]:
    categories = sorted({item["category"] for item in items})
    return {
        category: float(
            np.mean(
                [
                    similarities[index, index]
                    for index, item in enumerate(items)
                    if item["category"] == category
                ]
            )
        )
        for category in categories
    }


def load_preclip_comparison(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    preclip = json.loads(path.read_text(encoding="utf-8"))
    analysis = preclip.get("analysis", {})
    return {
        "preclip_within_modality_similarity_spearman": analysis.get(
            "within_modality_similarity_spearman"
        ),
        "preclip_nearest_neighbor_overlap_at_3": analysis.get(
            "nearest_neighbor_overlap_at_3"
        ),
    }


def run_experiment(
    manifest_path: Path,
    output_path: Path,
    model_name: str,
    preclip_path: Path,
    batch_size: int,
) -> None:
    items = load_items(manifest_path)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).eval()
    image_embeddings = encode_images(items, processor, model, batch_size)
    text_embeddings = encode_texts(items, processor, model, batch_size)
    similarities = image_embeddings @ text_embeddings.T

    metrics = matching_metrics(similarities)
    metrics["mean_correct_pair_similarity"] = float(np.mean(np.diag(similarities)))
    metrics["mean_correct_pair_similarity_by_category"] = category_metrics(items, similarities)
    comparison = load_preclip_comparison(preclip_path)
    if comparison is not None:
        metrics["preclip_comparison"] = comparison

    results = {
        "experiment": "clip",
        "model": model_name,
        "embedding_dimension": int(image_embeddings.shape[1]),
        "metrics": metrics,
        "items": [
            {
                "id": item["id"],
                "category": item["category"],
                "image": item["image"],
                "caption": item["caption"],
            }
            for item in items
        ],
        "image_embeddings": image_embeddings.tolist(),
        "text_embeddings": text_embeddings.tolist(),
        "image_text_similarity_matrix": similarities.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved CLIP results to {output_path}")
    print(f"Embedding dimension: {image_embeddings.shape[1]}")
    print(f"Top-1 matching accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"Mean reciprocal rank: {metrics['mean_reciprocal_rank']:.4f}")
    print(f"Mean correct-pair similarity: {metrics['mean_correct_pair_similarity']:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("assets/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("results/clip_results.json"))
    parser.add_argument("--preclip", type=Path, default=Path("results/preclip_results.json"))
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_experiment(
        args.manifest,
        args.output,
        args.model,
        args.preclip,
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
