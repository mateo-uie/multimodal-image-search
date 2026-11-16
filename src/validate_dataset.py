"""Validate the image-caption dataset manifest and its files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_CATEGORIES = {"Animals", "Food", "Vehicles", "Landscapes"}
EXPECTED_ITEMS_PER_CATEGORY = 5


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as file:
        return json.load(file)


def validate_dataset(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = load_manifest(manifest_path)
    categories = manifest.get("categories")

    if not isinstance(categories, list):
        return ["The manifest must contain a 'categories' list."]

    category_names = [category.get("name") for category in categories]
    if set(category_names) != EXPECTED_CATEGORIES or len(category_names) != len(EXPECTED_CATEGORIES):
        errors.append(
            "Expected exactly the categories: "
            + ", ".join(sorted(EXPECTED_CATEGORIES))
        )

    ids: set[str] = set()
    image_paths: set[str] = set()
    total_items = 0
    dataset_root = manifest_path.parent

    for category in categories:
        category_name = category.get("name", "<unnamed>")
        items = category.get("items")
        if not isinstance(items, list):
            errors.append(f"Category {category_name!r} must contain an 'items' list.")
            continue

        if len(items) != EXPECTED_ITEMS_PER_CATEGORY:
            errors.append(
                f"Category {category_name!r} contains {len(items)} items; "
                f"expected {EXPECTED_ITEMS_PER_CATEGORY}."
            )

        total_items += len(items)
        for item in items:
            item_id = item.get("id")
            image = item.get("image")
            caption = item.get("caption")

            if not isinstance(item_id, str) or not item_id.strip():
                errors.append(f"Category {category_name!r} has an item without a valid id.")
            elif item_id in ids:
                errors.append(f"Duplicate item id: {item_id}.")
            else:
                ids.add(item_id)

            if not isinstance(image, str) or not image.strip():
                errors.append(f"Item {item_id!r} has no valid image path.")
            else:
                image_path = dataset_root / image
                if not image_path.is_file():
                    errors.append(f"Image does not exist: {image_path}.")
                if image in image_paths:
                    errors.append(f"Duplicate image path: {image}.")
                image_paths.add(image)

            if not isinstance(caption, str) or not caption.strip():
                errors.append(f"Item {item_id!r} has an empty caption.")

    if total_items != 20:
        errors.append(f"The dataset contains {total_items} items; expected 20.")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/dataset.json"),
        help="Path to the dataset manifest (default: assets/dataset.json).",
    )
    args = parser.parse_args()

    errors = validate_dataset(args.manifest)
    if errors:
        print("Dataset validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Dataset validation passed: {args.manifest}")
    print("- 4 categories")
    print("- 20 images")
    print("- 20 captions")
    print("- 5 items per category")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
