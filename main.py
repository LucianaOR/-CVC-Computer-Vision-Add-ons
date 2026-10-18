from pathlib import Path
import csv
from statistics import mean

from src.detectors import METHODS
from src.experiments import run_feature_pair

PROJECT_DIR = Path(__file__).resolve().parent
HPATCHES_DIR = PROJECT_DIR / "hpatches-sequences-release"

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

SEQUENCES_PER_TYPE = 3
TARGET_IMAGES = [2, 3, 4, 5, 6]

RATIO_THRESHOLD = 0.8
REPROJECTION_THRESHOLD = 3.0


def get_sequences(dataset_dir, prefix, limit):
    """Select HPatches sequences by transformation type."""

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"HPatches dataset not found: {dataset_dir}"
        )

    sequences = sorted([
        path for path in dataset_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ])

    return sequences[:limit]


def save_csv(rows, output_path):
    """Save experiment results to CSV."""

    if not rows:
        return

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(results):
    """Calculate macro-averaged metrics for each method and type."""

    summary = []

    for method in ["KAZE", "BRISK", "FAST_BRISK"]:
        for transformation in ["illumination", "viewpoint"]:
            subset = [
                row for row in results
                if row["method"] == method
                and row["transformation"] == transformation
            ]

            if not subset:
                continue

            summary.append({
                "method": method,
                "transformation": transformation,
                "experiments": len(subset),
                "mean_keypoints_image1": mean(
                    row["keypoints_image1"] for row in subset
                ),
                "mean_keypoints_image2": mean(
                    row["keypoints_image2"] for row in subset
                ),
                "mean_putative_matches": mean(
                    row["putative_matches"] for row in subset
                ),
                "mean_correct_matches": mean(
                    row["correct_matches"] for row in subset
                ),
                "mean_precision": mean(
                    row["precision"] for row in subset
                ),
                "mean_pmr": mean(
                    row["pmr"] for row in subset
                ),
                "mean_matching_score": mean(
                    row["matching_score"] for row in subset
                ),
                "mean_detection_description_time1_ms": mean(
                    row["detection_description_time1_ms"]
                    for row in subset
                ),
                "mean_detection_description_time2_ms": mean(
                    row["detection_description_time2_ms"]
                    for row in subset
                ),
                "mean_matching_time_ms": mean(
                    row["matching_time_ms"] for row in subset
                ),
                "mean_descriptor_memory_bytes": mean(
                    row["descriptor_memory_bytes"]
                    for row in subset
                )
            })

    return summary


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    sequences = (
        get_sequences(
            HPATCHES_DIR, "i_", SEQUENCES_PER_TYPE
        )
        +
        get_sequences(
            HPATCHES_DIR, "v_", SEQUENCES_PER_TYPE
        )
    )

    if not sequences:
        raise RuntimeError("No HPatches sequences found.")

    results = []

    print("=== Classical feature comparison ===")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Sequences: {len(sequences)}")

    for method in METHODS:
        print(f"\n========== {method} ==========")

        for sequence_dir in sequences:
            print(f"\nSequence: {sequence_dir.name}")

            for target_index in TARGET_IMAGES:
                print(
                    f"  Pair 1 -> {target_index}...",
                    end=" ",
                    flush=True
                )

                try:
                    # Save figures only for pair 1 -> 2.
                    figures_dir = (
                        FIGURES_DIR
                        if target_index == 2
                        else None
                    )

                    result = run_feature_pair(
                        sequence_dir=sequence_dir,
                        target_index=target_index,
                        method=method,
                        figures_dir=figures_dir,
                        ratio_threshold=RATIO_THRESHOLD,
                        reprojection_threshold=REPROJECTION_THRESHOLD
                    )

                    results.append(result)

                    print(
                        f"matches={result['putative_matches']}, "
                        f"correct={result['correct_matches']}, "
                        f"precision={result['precision']:.3f}"
                    )

                except (FileNotFoundError, ValueError) as error:
                    print(f"SKIPPED: {error}")

    save_csv(
        results,
        TABLES_DIR / "classical_hpatches_results.csv"
    )

    summary = summarize_results(results)

    save_csv(
        summary,
        TABLES_DIR / "classical_hpatches_summary.csv"
    )

    print("\n=== Final comparison ===")

    for row in summary:
        print(
            f"{row['method']:12s} | "
            f"{row['transformation']:12s} | "
            f"precision={row['mean_precision']:.4f} | "
            f"matching_score={row['mean_matching_score']:.4f} | "
            f"matching_time={row['mean_matching_time_ms']:.2f} ms"
        )

    print(f"\nResults saved to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
