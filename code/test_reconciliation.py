from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allows this test to run alongside code/reconciliation.py after copying both
# files into the repository's code/ directory.
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from reconciliation import reconcile_all, state_summary  # noqa: E402


class LocalBundle:
    pass


def build_bundle(dataset_dir: Path) -> LocalBundle:
    data = LocalBundle()
    data.profiles = pd.read_csv(dataset_dir / "financial_profiles.csv")
    data.events = pd.read_csv(dataset_dir / "financial_events.csv")
    data.requests = pd.read_csv(dataset_dir / "requests.csv")
    data.request_payment_options = pd.read_csv(dataset_dir / "request_payment_options.csv")
    data.messages = pd.read_csv(dataset_dir / "messages.csv")
    data.exchange_rates = pd.read_csv(dataset_dir / "exchange_rates.csv")
    data.profile_by_user = {
        str(row.user_id): row for row in data.profiles.itertuples(index=False)
    }
    data.request_by_id = {
        str(row.request_id): row._asdict()
        for row in data.requests.itertuples(index=False)
    }
    return data


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data = build_bundle(repo_root / "dataset")
    states = reconcile_all(data)

    assert len(states) == len(data.requests) == 250
    assert set(states) == set(data.requests["request_id"].astype(str))

    # Request 26 is a useful known case from the challenge dataset.
    state = states["request_26"]
    assert state.home_currency == "IDR"
    assert state.requested_amount == 15656000.0
    assert state.pending_debit_reserve >= 0
    assert state.minimum_balance_to_keep < state.current_available_balance
    assert len(state.payment_options) == 4
    assert len(state.recurring_candidates) >= 1

    missing_events = sum(len(s.missing_amount_event_ids) for s in states.values())
    unconverted_events = sum(len(s.unconverted_event_ids) for s in states.values())
    pending_reserves = sum(s.pending_debit_reserve for s in states.values())
    future_events = sum(len(s.future_cash_events) for s in states.values())
    recurring = sum(len(s.recurring_candidates) for s in states.values())

    print("RECONCILIATION TESTS PASSED")
    print({
        "requests_reconciled": len(states),
        "missing_amount_events": missing_events,
        "unconverted_events": unconverted_events,
        "future_cash_events": future_events,
        "recurring_candidates": recurring,
        "pending_reserve_sum": round(pending_reserves, 2),
    })
    print("request_26:", state_summary(state))


if __name__ == "__main__":
    main()
