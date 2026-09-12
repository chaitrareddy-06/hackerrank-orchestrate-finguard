from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from dataset/ and fail clearly if it is missing."""
    path = DATASET_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")

    return pd.read_csv(path)


def _parse_date_column(
    df: pd.DataFrame,
    column: str,
    *,
    utc: bool = False,
) -> None:
    """Parse a date/datetime column in-place."""
    if column not in df.columns:
        return

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce",
        utc=utc,
    )


def _parse_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Convert a numeric column to numeric while preserving missing values."""
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def _parse_bool_column(df: pd.DataFrame, column: str) -> None:
    """Normalize a boolean column represented as strings."""
    if column not in df.columns:
        return

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        True: True,
        False: False,
    }

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def _split_pipe_list(value: object) -> List[str]:
    """Convert pipe-delimited profile fields into a clean list."""
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [item.strip() for item in text.split("|") if item.strip()]


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    profiles: pd.DataFrame
    events: pd.DataFrame
    requests: pd.DataFrame
    payment_options: pd.DataFrame
    messages: pd.DataFrame
    images: pd.DataFrame
    exchange_rates: pd.DataFrame
    sample_requests: pd.DataFrame
    output_template: pd.DataFrame

    profile_by_user: Dict[str, dict]
    events_by_user: Dict[str, pd.DataFrame]
    event_by_id: Dict[str, dict]
    request_by_id: Dict[str, dict]
    options_by_request: Dict[str, pd.DataFrame]
    messages_by_request: Dict[str, pd.DataFrame]
    messages_by_event: Dict[str, pd.DataFrame]
    images_by_request: Dict[str, pd.DataFrame]
    images_by_event: Dict[str, pd.DataFrame]
    rates_by_key: Dict[Tuple[str, str, str], float]


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_data() -> DataBundle:
    """Load, normalize, index, and validate the complete participant dataset."""

    # -----------------------------------------------------------------------
    # Load raw CSVs
    # -----------------------------------------------------------------------

    profiles = _read_csv("financial_profiles.csv")
    events = _read_csv("financial_events.csv")
    requests = _read_csv("requests.csv")
    payment_options = _read_csv("request_payment_options.csv")
    messages = _read_csv("messages.csv")
    images = _read_csv("images.csv")
    exchange_rates = _read_csv("exchange_rates.csv")
    sample_requests = _read_csv("sample_requests.csv")
    output_template = _read_csv("output.csv")

    # -----------------------------------------------------------------------
    # Parse dates / datetimes
    # -----------------------------------------------------------------------

    _parse_date_column(events, "event_date")
    _parse_date_column(events, "settlement_date")

    _parse_date_column(requests, "request_date")
    _parse_date_column(requests, "desired_completion_date")

    _parse_date_column(payment_options, "first_payment_date")

    _parse_date_column(
        messages,
        "sent_at",
        utc=True,
    )

    _parse_date_column(exchange_rates, "rate_date")

    _parse_date_column(sample_requests, "request_date")
    _parse_date_column(sample_requests, "desired_completion_date")
    _parse_date_column(sample_requests, "earliest_date_for_full_payment")

    # -----------------------------------------------------------------------
    # Parse numeric fields
    # -----------------------------------------------------------------------

    for column in [
        "current_available_balance",
        "minimum_balance_to_keep",
        "max_installment_months",
    ]:
        _parse_numeric_column(profiles, column)

    for column in [
        "amount",
        "minimum_allowed_amount",
    ]:
        _parse_numeric_column(events, column)

    for column in [
        "requested_amount",
    ]:
        _parse_numeric_column(requests, column)

    for column in [
        "payment_amount",
        "number_of_payments",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    ]:
        _parse_numeric_column(payment_options, column)

    _parse_numeric_column(exchange_rates, "rate")

    for column in [
        "requested_amount",
        "amount_safe_to_pay",
    ]:
        _parse_numeric_column(sample_requests, column)

    # -----------------------------------------------------------------------
    # Parse booleans
    # -----------------------------------------------------------------------

    _parse_bool_column(requests, "allows_partial_payment")

    # -----------------------------------------------------------------------
    # Normalize identifier columns
    # -----------------------------------------------------------------------

    identifier_columns = {
        "profiles": ["user_id"],
        "events": ["event_id", "user_id", "linked_event_id"],
        "requests": ["request_id", "user_id"],
        "payment_options": ["payment_option_id", "request_id"],
        "messages": ["message_id", "user_id", "request_id", "related_event_id"],
        "images": ["image_id", "user_id", "request_id", "related_event_id"],
        "exchange_rates": ["from_currency", "to_currency"],
    }

    frames = {
        "profiles": profiles,
        "events": events,
        "requests": requests,
        "payment_options": payment_options,
        "messages": messages,
        "images": images,
        "exchange_rates": exchange_rates,
    }

    for frame_name, columns in identifier_columns.items():
        frame = frames[frame_name]

        for column in columns:
            if column in frame.columns:
                frame[column] = frame[column].astype("string").str.strip()

    # -----------------------------------------------------------------------
    # Profile convenience fields
    # -----------------------------------------------------------------------

    if "financial_priorities" in profiles.columns:
        profiles["financial_priorities_list"] = profiles[
            "financial_priorities"
        ].apply(_split_pipe_list)

    if "expense_categories_to_protect" in profiles.columns:
        profiles["protected_categories"] = profiles[
            "expense_categories_to_protect"
        ].apply(_split_pipe_list)

    if "expense_categories_user_is_willing_to_reduce" in profiles.columns:
        profiles["reducible_categories"] = profiles[
            "expense_categories_user_is_willing_to_reduce"
        ].apply(_split_pipe_list)

    if "expense_categories_user_is_willing_to_stop" in profiles.columns:
        profiles["stoppable_categories"] = profiles[
            "expense_categories_user_is_willing_to_stop"
        ].apply(_split_pipe_list)

    if "payment_methods_user_will_consider" in profiles.columns:
        profiles["payment_methods"] = profiles[
            "payment_methods_user_will_consider"
        ].apply(_split_pipe_list)

    # -----------------------------------------------------------------------
    # Basic integrity checks
    # -----------------------------------------------------------------------

    required_profile_columns = {
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    }

    required_event_columns = {
        "event_id",
        "user_id",
        "event_type",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    }

    required_request_columns = {
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    }

    required_option_columns = {
        "payment_option_id",
        "request_id",
        "payment_method",
        "payment_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    }

    def check_columns(
        df: pd.DataFrame,
        required: set[str],
        name: str,
    ) -> None:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    check_columns(profiles, required_profile_columns, "financial_profiles.csv")
    check_columns(events, required_event_columns, "financial_events.csv")
    check_columns(requests, required_request_columns, "requests.csv")
    check_columns(
        payment_options,
        required_option_columns,
        "request_payment_options.csv",
    )

    # Unique primary identifiers
    if profiles["user_id"].duplicated().any():
        raise ValueError("financial_profiles.csv contains duplicate user_id values")

    if events["event_id"].duplicated().any():
        raise ValueError("financial_events.csv contains duplicate event_id values")

    if requests["request_id"].duplicated().any():
        raise ValueError("requests.csv contains duplicate request_id values")

    if payment_options["payment_option_id"].duplicated().any():
        raise ValueError(
            "request_payment_options.csv contains duplicate payment_option_id"
        )

    # Every request user should have a profile.
    request_users = set(requests["user_id"].dropna())
    profile_users = set(profiles["user_id"].dropna())

    missing_request_profiles = request_users - profile_users

    if missing_request_profiles:
        raise ValueError(
            "Requests reference users without profiles: "
            f"{sorted(missing_request_profiles)}"
        )

    # -----------------------------------------------------------------------
    # Build indexes
    # -----------------------------------------------------------------------

    profile_by_user = profiles.set_index("user_id").to_dict(orient="index")

    events_by_user = {
        user_id: group.copy()
        for user_id, group in events.groupby("user_id", sort=False)
    }

    event_by_id = events.set_index("event_id").to_dict(orient="index")

    request_by_id = requests.set_index("request_id").to_dict(orient="index")

    options_by_request = {
        request_id: group.copy()
        for request_id, group in payment_options.groupby(
            "request_id",
            sort=False,
        )
    }

    messages_by_request = {
        request_id: group.copy()
        for request_id, group in messages.groupby(
            "request_id",
            dropna=True,
            sort=False,
        )
    }

    messages_by_event = {
        event_id: group.copy()
        for event_id, group in messages.groupby(
            "related_event_id",
            dropna=True,
            sort=False,
        )
    }

    images_by_request = {
        request_id: group.copy()
        for request_id, group in images.groupby(
            "request_id",
            dropna=True,
            sort=False,
        )
    }

    images_by_event = {
        event_id: group.copy()
        for event_id, group in images.groupby(
            "related_event_id",
            dropna=True,
            sort=False,
        )
    }

    # Fixed FX lookup:
    # (YYYY-MM-DD, FROM, TO) -> rate
    rates_by_key: Dict[Tuple[str, str, str], float] = {}

    for row in exchange_rates.itertuples(index=False):
        if pd.isna(row.rate_date) or pd.isna(row.rate):
            continue

        rate_date = row.rate_date.strftime("%Y-%m-%d")
        key = (
            rate_date,
            str(row.from_currency),
            str(row.to_currency),
        )

        rates_by_key[key] = float(row.rate)

    return DataBundle(
        profiles=profiles,
        events=events,
        requests=requests,
        payment_options=payment_options,
        messages=messages,
        images=images,
        exchange_rates=exchange_rates,
        sample_requests=sample_requests,
        output_template=output_template,
        profile_by_user=profile_by_user,
        events_by_user=events_by_user,
        event_by_id=event_by_id,
        request_by_id=request_by_id,
        options_by_request=options_by_request,
        messages_by_request=messages_by_request,
        messages_by_event=messages_by_event,
        images_by_request=images_by_request,
        images_by_event=images_by_event,
        rates_by_key=rates_by_key,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_dataset_diagnostics(data: DataBundle) -> None:
    """Print a compact integrity summary for the loaded dataset."""

    print("=" * 70)
    print("FIN GUARD DATASET DIAGNOSTICS")
    print("=" * 70)

    print(f"Profiles:          {len(data.profiles):,}")
    print(f"Financial events:  {len(data.events):,}")
    print(f"Requests:          {len(data.requests):,}")
    print(f"Payment options:   {len(data.payment_options):,}")
    print(f"Messages:          {len(data.messages):,}")
    print(f"Images:            {len(data.images):,}")
    print(f"Exchange rates:    {len(data.exchange_rates):,}")
    print(f"Public samples:    {len(data.sample_requests):,}")

    print("\n--- Data quality ---")

    print(
        "Events with blank amount:",
        int(data.events["amount"].isna().sum()),
    )

    print(
        "Events with linked_event_id:",
        int(data.events["linked_event_id"].notna().sum()),
    )

    print(
        "Messages linked to events:",
        int(data.messages["related_event_id"].notna().sum()),
    )

    print(
        "Images linked to events:",
        int(data.images["related_event_id"].notna().sum()),
    )

    print("\n--- Request coverage ---")

    requests_with_options = set(data.payment_options["request_id"].dropna())
    requests_without_options = sorted(
        set(data.requests["request_id"]) - requests_with_options
    )

    print(
        "Requests with payment options:",
        len(requests_with_options),
    )

    print(
        "Requests without payment options:",
        len(requests_without_options),
    )

    print("\n--- Currencies ---")
    print(
        ", ".join(sorted(data.profiles["home_currency"].dropna().unique()))
    )

    print("\n--- Event statuses ---")
    print(data.events["status"].value_counts().to_string())

    print("\n--- Event types ---")
    print(data.events["event_type"].value_counts().to_string())

    print("\n--- Partial payment ---")
    print(
        data.requests["allows_partial_payment"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    data = load_data()
    print_dataset_diagnostics(data)