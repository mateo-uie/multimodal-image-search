"""Run the Pre-CLIP experiment with separate image and text models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


DEFAULT_TEXT_MODEL = "sentence-transformers/distiluse-base-multilingual-cased"


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


def extract_image_embeddings(items: list[dict], batch_size: int = 8) -> np.ndarray:
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    transform = weights.transforms()
    embeddings: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = []
            for item in items[start : start + batch_size]:
                with Image.open(item["image_path"]) as image:
                    batch.append(transform(image.convert("RGB")))
            features = feature_extractor(torch.stack(batch)).flatten(start_dim=1)
            embeddings.append(features.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def extract_text_embeddings(
    items: list[dict], model_name: str = DEFAULT_TEXT_MODEL
) -> np.ndarray:
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(
        [item["caption"] for item in items],
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    return normalized @ normalized.T


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = rankdata(first)
    second_ranks = rankdata(second)
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator == 0:
        return 0.0
    return float(first_centered @ second_centered / denominator)


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    indices = np.triu_indices_from(matrix, k=1)
    return matrix[indices]


def nearest_neighbor_overlap(
    image_similarities: np.ndarray,
    text_similarities: np.ndarray,
    k: int = 3,
) -> float:
    overlaps = []
    for index in range(len(image_similarities)):
        image_order = np.argsort(-image_similarities[index]).tolist()
        text_order = np.argsort(-text_similarities[index]).tolist()
        image_neighbors = [neighbor for neighbor in image_order if neighbor != index][:k]
        text_neighbors = [neighbor for neighbor in text_order if neighbor != index][:k]
        overlaps.append(len(set(image_neighbors) & set(text_neighbors)) / k)
    return float(np.mean(overlaps))


def summarize_by_category(items: list[dict], similarities: np.ndarray) -> dict[str, float]:
    summary: dict[str, float] = {}
    categories = sorted({item["category"] for item in items})
    for category in categories:
        indices = [index for index, item in enumerate(items) if item["category"] == category]
        if len(indices) < 2:
            summary[category] = 0.0
            continue
        block = similarities[np.ix_(indices, indices)]
        summary[category] = float(upper_triangle(block).mean())
    return summary


def run_experiment(manifest_path: Path, output_path: Path, batch_size: int) -> None:
    items = load_items(manifest_path)
    image_embeddings = extract_image_embeddings(items, batch_size=batch_size)
    text_embeddings = extract_text_embeddings(items)
    image_similarities = cosine_matrix(image_embeddings)
    text_similarities = cosine_matrix(text_embeddings)
    image_pairs = upper_triangle(image_similarities)
    text_pairs = upper_triangle(text_similarities)

    results = {
        "experiment": "pre-clip",
        "image_model": "torchvision.resnet18(weights=ResNet18_Weights.DEFAULT), classifier removed",
        "text_model": DEFAULT_TEXT_MODEL,
        "embedding_dimensions": {
            "images": int(image_embeddings.shape[1]),
            "captions": int(text_embeddings.shape[1]),
        },
        "analysis": {
            "within_modality_similarity_spearman": spearman_correlation(image_pairs, text_pairs),
            "nearest_neighbor_overlap_at_3": nearest_neighbor_overlap(
                image_similarities, text_similarities, k=3
            ),
            "mean_image_similarity_by_category": summarize_by_category(items, image_similarities),
            "mean_caption_similarity_by_category": summarize_by_category(items, text_similarities),
        },
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
        "image_similarity_matrix": image_similarities.tolist(),
        "text_similarity_matrix": text_similarities.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved Pre-CLIP results to {output_path}")
    print(f"Image embedding dimension: {image_embeddings.shape[1]}")
    print(f"Text embedding dimension: {text_embeddings.shape[1]}")
    print(
        "Within-modality similarity Spearman correlation: "
        f"{results['analysis']['within_modality_similarity_spearman']:.4f}"
    )
    print(
        "Nearest-neighbor overlap at k=3: "
        f"{results['analysis']['nearest_neighbor_overlap_at_3']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("assets/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("results/preclip_results.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_experiment(args.manifest, args.output, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
