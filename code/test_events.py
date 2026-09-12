from __future__ import annotations

from pathlib import Path
import sys


# Allow imports from the code/ directory when this script is run as:
# python code\test_events.py
CODE_DIR = Path(__file__).resolve().parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from data_loader import load_data
from events import prepare_user_events, summarize_event_treatment


def print_user_test(
    data,
    user_id: str,
    request_date: str,
) -> None:
    prepared = prepare_user_events(
        data.events,
        user_id=user_id,
        request_date=request_date,
    )

    print(f"\n===== {user_id} @ {request_date} =====")

    summary = summarize_event_treatment(prepared)

    print(summary)

    # Show non-settled or otherwise special states.
    interesting = prepared[
        prepared["status"].isin(
            [
                "pending",
                "scheduled",
                "cancelled",
                "failed",
                "unrealized",
            ]
        )
    ]

    if not interesting.empty:
        print(
            interesting[
                [
                    "event_id",
                    "event_type",
                    "direction",
                    "amount",
                    "status",
                    "settlement_date",
                    "cash_effect",
                    "include_in_forecast",
                    "reserve_now",
                    "classification_reason",
                ]
            ].to_string(index=False)
        )


def main() -> None:
    data = load_data()

    # Representative users selected specifically to exercise
    # different financial states.
    tests = [
        ("user_02", "2025-08-05"),
        ("user_03", "2019-09-03"),
        ("user_04", "2024-06-04"),
        ("user_21", "2026-04-03"),
    ]

    for user_id, request_date in tests:
        print_user_test(
            data,
            user_id=user_id,
            request_date=request_date,
        )


if __name__ == "__main__":
    main()