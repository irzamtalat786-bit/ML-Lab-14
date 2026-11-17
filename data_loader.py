
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURES, CLASS_CAP, UNKNOWN_REAL


def normalize_column_name(name):
   
    return " ".join(
        str(name)
        .replace("\ufeff", "")
        .strip()
        .lower()
        .split()
    )


def normalize_labels(labels):
    """Normalize common CIC-IDS2017 attack labels."""

    labels = labels.astype(str).str.strip()

    # Case-insensitive normalization of common labels.
    label_map = {
        "benign": "Benign",
        "ftp-patator": "BruteForce",
        "ssh-patator": "BruteForce",
        "bot": "Botnet",
        "infilteration": "Infiltration",
        "infiltration": "Infiltration",
    }

    labels = labels.apply(
        lambda x: label_map.get(x.lower(), x)
    )

    labels = labels.str.replace(
        r"^DDoS$",
        "DoS",
        case=False,
        regex=True
    )

    labels = labels.str.replace(
        r"^DoS.*",
        "DoS",
        case=False,
        regex=True
    )

    labels = labels.str.replace(
        r"^Web Attack.*",
        "WebAttack",
        case=False,
        regex=True
    )

    return labels


def load_data():
   

    paths = glob.glob(str(DATA_DIR / "*.csv"))

    # --------------------------------------------------
    # 1. Check dataset files
    # --------------------------------------------------

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIR}"
        )

    print("\nFound CSV files:", len(paths))

    # --------------------------------------------------
    # 2. Chronological ordering
    # --------------------------------------------------

    weekday_order = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4
    }

    def dataset_file_order(path):

        name = Path(path).name.lower()

        day = next(
            (
                rank
                for weekday, rank in weekday_order.items()
                if weekday in name
            ),
            99
        )

        if "morning" in name:
            session = 0

        elif "afternoon" in name:
            session = 1

        else:
            session = 2

        return day, session, name

    paths = sorted(
        paths,
        key=dataset_file_order
    )

    # --------------------------------------------------
    # 3. Required columns
    # --------------------------------------------------

    required_columns = FEATURES + ["Label"]

    normalized_required = {
        normalize_column_name(column): column
        for column in required_columns
    }

    frames = []

    # --------------------------------------------------
    # 4. Read CSV files
    # --------------------------------------------------

    for path in paths:

        filename = Path(path).name

        try:

            print(f"\nChecking: {filename}")

            # Read headers first.
            # latin1 supports the CIC-IDS2017 CSV encoding
            # and avoids Unicode decoding failures.

            header = pd.read_csv(
                path,
                nrows=0,
                encoding="latin1"
            )

            # Normalize available headers.

            available_columns = {
                normalize_column_name(column): column
                for column in header.columns
            }

            # Identify missing columns.

            missing = [
                column
                for column in normalized_required
                if column not in available_columns
            ]

            if missing:

                print(
                    f"SKIPPED: {filename}"
                )

                print(
                    "Missing columns:",
                    missing
                )

                print(
                    "Available headers:",
                    header.columns.tolist()[:20]
                )

                continue

            # Select columns using a callable.
            # This avoids the previous usecols mismatch.

            part = pd.read_csv(
                path,
                usecols=lambda column: (
                    normalize_column_name(column)
                    in normalized_required
                ),
                encoding="latin1",
                low_memory=False
            )

            # Map actual CSV headers to expected names.

            rename_map = {}

            for column in part.columns:

                normalized = normalize_column_name(column)

                if normalized in normalized_required:

                    rename_map[column] = (
                        normalized_required[normalized]
                    )

            part = part.rename(
                columns=rename_map
            )

            # Verify that all selected columns exist.

            missing_after_read = [
                column
                for column in required_columns
                if column not in part.columns
            ]

            if missing_after_read:

                print(
                    f"SKIPPED: {filename}"
                )

                print(
                    "Missing after reading:",
                    missing_after_read
                )

                continue

            # Keep consistent column order.

            part = part[required_columns]

            frames.append(part)

            print(
                f"SUCCESS: {filename} "
                f"({len(part):,} rows)"
            )

        except Exception as exc:

            print(
                f"ERROR reading {filename}: {exc}"
            )

    # --------------------------------------------------
    # 5. Verify successful loading
    # --------------------------------------------------

    if not frames:

        raise ValueError(
            "\nNo CSV files could be loaded.\n"
            "Check the dataset headers and CSV format."
        )

    print(
        f"\nSuccessfully loaded {len(frames)} "
        f"out of {len(paths)} CSV files."
    )

    # --------------------------------------------------
    # 6. Combine datasets
    # --------------------------------------------------

    df = pd.concat(
        frames,
        ignore_index=True
    )

    print(
        "\nCombined raw records:",
        f"{len(df):,}"
    )

    # --------------------------------------------------
    # 7. Convert features to numeric
    # --------------------------------------------------

    for feature in FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

    # Remove infinite values.

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Keep partially missing feature rows. Model pipelines impute missing
    # values using medians learned from their training split. Drop only rows
    # without a usable label or with every selected feature missing.
    df = df.dropna(subset=["Label"]).copy()
    df = df.dropna(subset=FEATURES, how="all").copy()

    # --------------------------------------------------
    # 8. Normalize attack labels
    # --------------------------------------------------

    df["Label"] = normalize_labels(df["Label"])
    df = df[df["Label"].str.lower().ne("nan")].copy()

    df = df[
        FEATURES + ["Label"]
    ]

    if df.empty:

        raise ValueError(
            "No valid records remain after preprocessing."
        )

    print(
        "\nRecords after cleaning:",
        f"{len(df):,}"
    )

    # --------------------------------------------------
    # 9. Class filtering
    # --------------------------------------------------

    kept_indices = []

    for label, group in df.groupby(
        "Label",
        sort=False
    ):

        # Keep classes with sufficient records.
        # Always preserve the configured unknown class.

        if len(group) >= 200 or label == UNKNOWN_REAL:

            kept_indices.extend(
                group.index[:CLASS_CAP].tolist()
            )

    if not kept_indices:

        raise ValueError(
            "No usable records remain after class filtering."
        )

    # Preserve chronological record order.

    df = df.loc[
        sorted(kept_indices)
    ].reset_index(drop=True)

    # --------------------------------------------------
    # 10. Verify unknown attack class
    # --------------------------------------------------

    available_labels = set(
        df["Label"].unique()
    )

    if UNKNOWN_REAL not in available_labels:

        raise ValueError(
            f"Unknown class '{UNKNOWN_REAL}' was not found.\n"
            f"Available labels: {sorted(available_labels)}\n"
            "Check UNKNOWN_REAL in src/config.py."
        )

    # --------------------------------------------------
    # 11. Final dataset information
    # --------------------------------------------------

    print("\n" + "=" * 50)

    print("FINAL DATASET SUMMARY")

    print("=" * 50)

    print(
        "\nTotal records:",
        f"{len(df):,}"
    )

    print(
        "Total features:",
        len(FEATURES)
    )

    print(
        "Total classes:",
        df["Label"].nunique()
    )

    print(
        "Unknown attack class:",
        UNKNOWN_REAL
    )

    print("\nCLASS DISTRIBUTION")

    print(
        df["Label"].value_counts().to_string()
    )

    print("\n" + "=" * 50)

    # Preserve original train.py interface.

    return df, None, False
