from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd


DATASET_REPO = "Sahibnoor1/gi-endoscopy-megabank-stage2"
EXPECTED_SPLIT_COUNTS = {"train": 7433, "val": 1593, "test": 1593}
EXPECTED_LABEL_COUNT = 23
PROMPTS = (
    "<image>\nClassify this GI endoscopy image using the HyperKvasir "
    "23-category taxonomy. Return only the category name.",
    "<image>\nWhat is the benchmark category for this HyperKvasir image? "
    "Return only the category name.",
    "<image>\nIdentify the primary HyperKvasir class shown in this image. "
    "Return only the category name.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_prompt(image_id: str) -> str:
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return PROMPTS[int.from_bytes(digest[:4], "big") % len(PROMPTS)]


def display_value(value: str) -> str:
    return value.replace("-", " ")


def load_and_validate_split(split_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(split_csv, low_memory=False)
    required = {
        "image_id",
        "storage_type",
        "shard_file",
        "path_in_shard",
        "file_type",
        "label",
        "organ_region",
        "category",
        "split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Split CSV is missing required columns: {missing}")

    if frame["image_id"].isna().any() or not frame["image_id"].is_unique:
        raise ValueError("image_id must be present and globally unique")
    if frame[list(required)].isna().any().any():
        null_columns = frame[list(required)].columns[
            frame[list(required)].isna().any()
        ].tolist()
        raise ValueError(f"Required columns contain missing values: {null_columns}")

    actual_counts = frame["split"].value_counts().to_dict()
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Official classification split counts changed: "
            f"expected {EXPECTED_SPLIT_COUNTS}, got {actual_counts}"
        )
    if frame["label"].nunique() != EXPECTED_LABEL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LABEL_COUNT} labels, got {frame['label'].nunique()}"
        )
    if set(frame["storage_type"]) != {"tar_shard"}:
        raise ValueError("This builder currently supports tar_shard rows only")
    if set(frame["file_type"]) != {"image"}:
        raise ValueError("Classification split must contain image rows only")

    split_ids = {
        split: set(part["image_id"].astype(str))
        for split, part in frame.groupby("split")
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[left].intersection(split_ids[right])
        if overlap:
            raise ValueError(f"{left}/{right} identity leakage: {len(overlap)} rows")

    return frame


def qwen_record(row: object) -> dict[str, object]:
    image_id = str(row.image_id)
    region = display_value(str(row.organ_region))
    group = display_value(str(row.category))
    return {
        "image": f"images/{image_id}.jpg",
        "conversations": [
            {
                "from": "human",
                "value": deterministic_prompt(image_id),
            },
            {
                "from": "gpt",
                "value": str(row.label),
            },
            {
                "from": "human",
                "value": (
                    "Which broad GI region and benchmark finding group are "
                    "recorded for this image?"
                ),
            },
            {
                "from": "gpt",
                "value": f"GI region: {region}. Finding group: {group}.",
            },
        ],
    }


def write_annotations(frame: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    annotation_dir = output_dir / "annotations"
    index_dir = output_dir / "index"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    index_columns = [
        "image_id",
        "label",
        "organ_region",
        "category",
        "shard_file",
        "path_in_shard",
        "split",
    ]
    for split in ("train", "val", "test"):
        part = frame.loc[frame["split"] == split].sort_values("image_id")
        records = [qwen_record(row) for row in part.itertuples(index=False)]
        annotation_path = annotation_dir / f"hyperkvasir_{split}.json"
        annotation_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        part[index_columns].to_csv(index_dir / f"hyperkvasir_{split}.csv", index=False)
        counts[split] = len(records)

    return counts


def resolve_remote_shards(repo_id: str, shard_basenames: set[str]) -> dict[str, str]:
    from huggingface_hub import HfApi

    files_by_basename: dict[str, list[str]] = defaultdict(list)
    for repo_file in HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset"):
        files_by_basename[PurePosixPath(repo_file).name].append(repo_file)

    resolved: dict[str, str] = {}
    for basename in sorted(shard_basenames):
        matches = files_by_basename.get(basename, [])
        if len(matches) > 1:
            hyperkvasir_matches = [
                path for path in matches if "hyperkvasir" in path.casefold()
            ]
            if len(hyperkvasir_matches) == 1:
                matches = hyperkvasir_matches
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one remote file named {basename!r}, found {matches}"
            )
        resolved[basename] = matches[0]
    return resolved


def materialize_images(
    frame: pd.DataFrame,
    output_dir: Path,
    repo_id: str,
    token: str | None,
) -> int:
    from huggingface_hub import hf_hub_download

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    remote_shards = resolve_remote_shards(repo_id, set(frame["shard_file"]))
    total_written = 0

    for shard_name, part in frame.groupby("shard_file", sort=True):
        remote_name = remote_shards[str(shard_name)]
        local_tar = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=remote_name,
                token=token,
            )
        )

        wanted_exact: dict[str, tuple[str, Path]] = {}
        wanted_basename: dict[str, tuple[str, Path]] = {}
        for row in part.itertuples(index=False):
            member_name = str(row.path_in_shard).lstrip("./")
            destination = image_dir / f"{row.image_id}.jpg"
            item = (str(row.image_id), destination)
            wanted_exact[member_name] = item
            wanted_basename[PurePosixPath(member_name).name] = item

        found: set[str] = set()
        with tarfile.open(local_tar, "r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                normalized = member.name.lstrip("./")
                item = wanted_exact.get(normalized)
                if item is None:
                    item = wanted_basename.get(PurePosixPath(normalized).name)
                if item is None:
                    continue

                image_id, destination = item
                if image_id in found:
                    raise RuntimeError(f"Duplicate tar member for {image_id}")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read {member.name} from {local_tar}")
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
                found.add(image_id)
                total_written += 1

        expected = set(part["image_id"].astype(str))
        missing = sorted(expected.difference(found))
        if missing:
            raise RuntimeError(
                f"Shard {shard_name} is missing {len(missing)} requested images; "
                f"examples: {missing[:5]}"
            )

    return total_written


def validate_materialized(frame: pd.DataFrame, output_dir: Path) -> None:
    missing = [
        image_id
        for image_id in frame["image_id"].astype(str)
        if not (output_dir / "images" / f"{image_id}.jpg").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Materialized dataset is missing {len(missing)} images: {missing[:5]}"
        )


def write_summary(
    frame: pd.DataFrame,
    split_csv: Path,
    output_dir: Path,
    repo_id: str,
    materialized: bool,
) -> None:
    summary = {
        "source_split_csv": str(split_csv.resolve()),
        "source_split_sha256": sha256_file(split_csv),
        "dataset_repo": repo_id,
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "label_count": int(frame["label"].nunique()),
        "label_counts": dict(sorted(Counter(frame["label"]).items())),
        "materialized": materialized,
        "frozen_evaluation_policy": {
            "train": "May be used for fitting Qwen adapters.",
            "val": "Selection and calibration only; never fit on this split.",
            "test": "Final evaluation only; never fit or select on this split.",
        },
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe Qwen3-VL annotations from the official split."
    )
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Download the required tar shards and extract the 10,619 images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_csv = args.split_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_and_validate_split(split_csv)
    counts = write_annotations(frame, output_dir)

    if args.materialize:
        written = materialize_images(
            frame=frame,
            output_dir=output_dir,
            repo_id=args.dataset_repo,
            token=os.getenv("HF_TOKEN"),
        )
        validate_materialized(frame, output_dir)
        print(f"Materialized images: {written}")

    write_summary(
        frame=frame,
        split_csv=split_csv,
        output_dir=output_dir,
        repo_id=args.dataset_repo,
        materialized=args.materialize,
    )
    print(f"Qwen annotations: {counts}")
    print(f"Output directory: {output_dir}")
    print("Leakage checks: OK")


if __name__ == "__main__":
    main()
