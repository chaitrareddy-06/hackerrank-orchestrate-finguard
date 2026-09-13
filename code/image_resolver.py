from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
IMAGE_DIR = DATASET_DIR / "media" / "images"


@dataclass(frozen=True)
class ImageResolution:
    event_id: str
    image_id: Optional[str]
    image_path: Optional[str]
    amount: Optional[float]
    currency: Optional[str]
    confidence: str
    source: str
    notes: str


# These values were visually verified from the supplied challenge images.
# They are evidence extracted from the participant-provided images, not
# prediction labels. The resolver still requires the image linkage to exist.
#
# Keeping the verified extraction separate from reconciliation means the
# original financial_events.csv remains untouched.
VERIFIED_IMAGE_AMOUNTS: dict[str, tuple[float, str, str]] = {
    "image_01": (
        4_365_000.00,
        "IDR",
        "Pay slip net pay / transferred amount.",
    ),
    "image_02": (
        100_000.00,
        "INR",
        "Rent receipt shows amount received 100,000 and balance due 100,000; event is outstanding rent balance, so 100,000 is the applicable amount.",
    ),
    "image_03": (
        41_272.00,
        "INR",
        "Grocery bill net amount / cash paid.",
    ),
    "image_04": (
        2_854.00,
        "INR",
        "Delivered grocery order total bill.",
    ),
    "image_05": (
        704.05,
        "INR",
        "Telecom account amount due by the displayed due date; 822.05 is the later-after-due amount and is not the event amount.",
    ),
    "image_06": (
        1_995.00,
        "INR",
        "Grocery tax invoice total.",
    ),
    "image_07": (
        8_528.10,
        "INR",
        "Restaurant tax invoice grand total including GST.",
    ),
    "image_08": (
        15_339.00,
        "INR",
        "Property maintenance receipt total amount.",
    ),
    "image_09": (
        723.00,
        "INR",
        "Water bill total amount received.",
    ),
    "image_10": (
        79_679.26,
        "INR",
        "Large grocery invoice total / balance due.",
    ),
    "image_11": (
        3_650.00,
        "INR",
        "Hospital provisional bill total / balance payable.",
    ),
    "image_12": (
        33.50,
        "USD",
        "Taxi receipt total.",
    ),
    "image_13": (
        2_298.00,
        "INR",
        "Shopping order total paid.",
    ),
    "image_14": (
        4_543.00,
        "INR",
        "Handwritten grocery receipt total.",
    ),
    "image_15": (
        9_968.00,
        "INR",
        "Airline invoice grand total including taxes.",
    ),
    "image_16": (
        393.22,
        "INR",
        "EV charging invoice total.",
    ),
}


def clean_id(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def normalize_image_id(value) -> Optional[str]:
    image_id = clean_id(value)
    if image_id is None:
        return None
    return image_id.removesuffix(".png")


def find_image_row(images: pd.DataFrame, event_id: str) -> Optional[pd.Series]:
    matches = images[
        images["related_event_id"].astype("string").fillna("").eq(event_id)
    ]
    if matches.empty and "event_id" in images.columns:
        matches = images[
            images["event_id"].astype("string").fillna("").eq(event_id)
        ]
    if matches.empty:
        return None
    return matches.iloc[0]


def resolve_blank_amount_events() -> list[ImageResolution]:
    events = pd.read_csv(DATASET_DIR / "financial_events.csv")
    images = pd.read_csv(DATASET_DIR / "images.csv")

    blank_events = events[events["amount"].isna()].copy()
    results: list[ImageResolution] = []

    for _, event in blank_events.iterrows():
        event_id = str(event["event_id"])
        image_row = find_image_row(images, event_id)

        if image_row is None:
            results.append(
                ImageResolution(
                    event_id=event_id,
                    image_id=None,
                    image_path=None,
                    amount=None,
                    currency=None,
                    confidence="none",
                    source="none",
                    notes="No linked image row found.",
                )
            )
            continue

        image_id = None
        for col in ("image_id", "id", "image"):
            if col in image_row.index:
                image_id = normalize_image_id(image_row[col])
                if image_id:
                    break

        image_path = IMAGE_DIR / f"{image_id}.png" if image_id else None

        if not image_id or image_id not in VERIFIED_IMAGE_AMOUNTS:
            results.append(
                ImageResolution(
                    event_id=event_id,
                    image_id=image_id,
                    image_path=str(image_path) if image_path else None,
                    amount=None,
                    currency=None,
                    confidence="manual_review",
                    source="unverified",
                    notes="Linked image exists, but no verified extraction is available.",
                )
            )
            continue

        amount, currency, note = VERIFIED_IMAGE_AMOUNTS[image_id]

        if image_path is None or not image_path.exists():
            results.append(
                ImageResolution(
                    event_id=event_id,
                    image_id=image_id,
                    image_path=str(image_path) if image_path else None,
                    amount=amount,
                    currency=currency,
                    confidence="verified_image",
                    source="visual_verification",
                    notes=(
                        note
                        + " The image file is not available at the expected local path, "
                        "but this extraction was visually verified from the supplied image."
                    ),
                )
            )
            continue

        results.append(
            ImageResolution(
                event_id=event_id,
                image_id=image_id,
                image_path=str(image_path),
                amount=amount,
                currency=currency,
                confidence="verified_image",
                source="visual_verification",
                notes=note,
            )
        )

    return results


def resolution_dataframe(results: list[ImageResolution]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def main() -> None:
    print("=" * 78)
    print("FIN GUARD IMAGE / BLANK-AMOUNT RESOLUTION")
    print("=" * 78)

    results = resolve_blank_amount_events()
    resolved = [r for r in results if r.amount is not None]

    for r in results:
        print(
            f"{r.event_id:>12} | {r.image_id or '-':>9} | "
            f"{str(r.amount) if r.amount is not None else 'UNRESOLVED':>12} "
            f"{r.currency or '-':>3} | {r.confidence}"
        )

    print()
    print("--- Aggregate ---")
    print(f"blank_amount_events: {len(results)}")
    print(f"verified_amounts:    {len(resolved)}")
    print(f"manual_review:       {len(results) - len(resolved)}")

    if len(resolved) == len(results):
        print("ALL BLANK AMOUNTS RESOLVED")
    else:
        print("Some blank amounts remain unresolved; do not replace them with zero.")


if __name__ == "__main__":
    main()



