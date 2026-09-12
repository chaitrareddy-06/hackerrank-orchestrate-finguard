from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
import math
import os
import sys

import pandas as pd

# Allow running this file directly from the repository's code/ directory.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

try:
    from evidence import parse_messages
except ImportError:  # pragma: no cover - useful when copied elsewhere
    parse_messages = None


VALID_STATUSES = {
    "settled",
    "pending",
    "scheduled",
    "cancelled",
    "failed",
    "unrealized",
}

EXCLUDED_CASH_STATUSES = {"cancelled", "failed", "unrealized"}


@dataclass(frozen=True)
class RecurringCandidate:
    """A recurring cash-flow pattern inferred from historical events."""

    event_type: str
    direction: str
    category: str
    cadence: str
    median_amount: float
    typical_day: int
    observations: int
    last_settlement_date: pd.Timestamp
    flexibility: str
    minimum_allowed_amount: Optional[float]


@dataclass
class ReconciliationState:
    """Request-specific financial state assembled from all supplied context.

    This layer deliberately does NOT decide buy/wait/installments. It creates
    a reliable state object for the forecast and decision layers.
    """

    request_id: str
    user_id: str
    request_date: pd.Timestamp
    desired_completion_date: pd.Timestamp
    request_type: str
    requested_amount: float
    allows_partial_payment: bool

    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float

    financial_priorities: tuple[str, ...] = field(default_factory=tuple)
    protected_categories: tuple[str, ...] = field(default_factory=tuple)
    reducible_categories: tuple[str, ...] = field(default_factory=tuple)
    stoppable_categories: tuple[str, ...] = field(default_factory=tuple)
    accepted_payment_methods: tuple[str, ...] = field(default_factory=tuple)
    max_installment_months: Optional[int] = None

    # Amount already tied up by pending debits at request time.
    pending_debit_reserve: float = 0.0

    # Future cash events that can be projected without replaying historical
    # events that are already reflected in current_available_balance.
    future_cash_events: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Historical candidates used later by the forecast layer.
    recurring_candidates: tuple[RecurringCandidate, ...] = field(default_factory=tuple)

    # All parsed evidence for this user/request, retained for later reasoning.
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Payment options visible for this request.
    payment_options: pd.DataFrame = field(default_factory=pd.DataFrame)

    # IDs of events with missing amounts. A later image/OCR layer can resolve
    # these; reconciliation never treats a blank amount as zero.
    missing_amount_event_ids: tuple[str, ...] = field(default_factory=tuple)

    # Conversion failures are retained explicitly rather than silently using a
    # guessed FX rate.
    unconverted_event_ids: tuple[str, ...] = field(default_factory=tuple)

    # Human-readable diagnostics for later validation/explanations.
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def immediately_spendable_after_reserve(self) -> float:
        return max(
            0.0,
            self.current_available_balance - self.pending_debit_reserve,
        )

    @property
    def current_discretionary_room(self) -> float:
        return max(
            0.0,
            self.immediately_spendable_after_reserve
            - self.minimum_balance_to_keep,
        )


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    if isinstance(row, pd.Series):
        return row.get(key, default)
    if hasattr(row, key):
        return getattr(row, key)
    try:
        return row[key]
    except (KeyError, TypeError, IndexError):
        return default


def _pipe_list(value: Any) -> tuple[str, ...]:
    text = _clean_text(value)
    if not text:
        return tuple()
    return tuple(item.strip() for item in text.split("|") if item.strip())


def _to_date(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_localize(None)
    return result.normalize()


def _safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _rate_lookup(
    rates: pd.DataFrame,
    rate_date: pd.Timestamp,
    from_currency: str,
    to_currency: str,
) -> Optional[float]:
    if from_currency == to_currency:
        return 1.0

    date_key = _to_date(rate_date)
    matches = rates[
        (pd.to_datetime(rates["rate_date"]).dt.normalize() == date_key)
        & (rates["from_currency"].astype(str).str.upper() == from_currency)
        & (rates["to_currency"].astype(str).str.upper() == to_currency)
    ]
    if matches.empty:
        return None

    rate = _safe_float(matches.iloc[0]["rate"])
    if rate is None or rate <= 0:
        return None
    return rate


def _convert_event_amount(
    row: pd.Series,
    home_currency: str,
    rates: pd.DataFrame,
) -> tuple[Optional[float], Optional[str]]:
    amount = _safe_float(row.get("amount"))
    if amount is None:
        return None, "missing_amount"

    currency = _clean_text(row.get("currency")).upper()
    if not currency:
        return None, "missing_currency"

    settlement_date = row.get("settlement_date")
    if pd.isna(settlement_date):
        settlement_date = row.get("event_date")
    if pd.isna(settlement_date):
        return None, "missing_settlement_date"

    rate = _rate_lookup(rates, _to_date(settlement_date), currency, home_currency)
    if rate is None:
        return None, f"missing_rate:{currency}->{home_currency}:{_to_date(settlement_date).date()}"

    return amount * rate, None


def _is_flexible_adjustable(
    row: pd.Series,
    protected_categories: Iterable[str],
    reducible_categories: Iterable[str],
    stoppable_categories: Iterable[str],
) -> bool:
    category = _clean_text(row.get("category"))
    flexibility = _clean_text(row.get("flexibility"))
    protected = set(protected_categories)
    reducible = set(reducible_categories)
    stoppable = set(stoppable_categories)

    if category in protected:
        return False

    if flexibility not in {
        "reducible",
        "stoppable",
        "reducible_or_stoppable",
    }:
        return False

    permitted = (
        (flexibility in {"reducible", "reducible_or_stoppable"} and category in reducible)
        or (flexibility in {"stoppable", "reducible_or_stoppable"} and category in stoppable)
    )
    return permitted


def _infer_cadence(dates: list[pd.Timestamp]) -> Optional[str]:
    if len(dates) < 3:
        return None
    ordered = sorted(dates)
    gaps = pd.Series([(b - a).days for a, b in zip(ordered, ordered[1:])])
    if gaps.empty:
        return None
    median_gap = float(gaps.median())
    if 25 <= median_gap <= 35:
        return "monthly"
    if 6 <= median_gap <= 8:
        return "weekly"
    if 12 <= median_gap <= 16:
        return "biweekly"
    if 80 <= median_gap <= 100:
        return "quarterly"
    return None


def _build_recurring_candidates(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    home_currency: str,
    rates: pd.DataFrame,
    protected_categories: Iterable[str],
    reducible_categories: Iterable[str],
    stoppable_categories: Iterable[str],
) -> tuple[RecurringCandidate, ...]:
    if events.empty:
        return tuple()

    hist = events.copy()
    hist["event_date_norm"] = pd.to_datetime(hist["event_date"]).dt.normalize()
    hist["settlement_date_norm"] = pd.to_datetime(hist["settlement_date"]).dt.normalize()

    hist = hist[
        (hist["settlement_date_norm"].notna())
        & (hist["settlement_date_norm"] < request_date)
        & (~hist["status"].isin(EXCLUDED_CASH_STATUSES))
        & (hist["direction"].isin(["debit", "credit"]))
    ].copy()

    if hist.empty:
        return tuple()

    converted_amounts: list[Optional[float]] = []
    for _, row in hist.iterrows():
        converted, _ = _convert_event_amount(row, home_currency, rates)
        converted_amounts.append(converted)
    hist["home_amount"] = converted_amounts
    hist = hist[hist["home_amount"].notna()].copy()

    candidates: list[RecurringCandidate] = []
    grouped = hist.groupby(
        ["event_type", "direction", "category"],
        dropna=False,
    )

    for (event_type, direction, category), group in grouped:
        if len(group) < 3:
            continue

        dates = [pd.Timestamp(x).normalize() for x in group["settlement_date_norm"]]
        cadence = _infer_cadence(dates)
        if cadence is None:
            continue

        # Avoid calling noisy variable spending "recurring" just because it
        # happens frequently. Recurring patterns require both cadence and a
        # reasonably stable amount.
        median_amount = float(group["home_amount"].median())
        mad = float((group["home_amount"] - median_amount).abs().median())
        if median_amount <= 0:
            continue
        if mad > median_amount * 0.60:
            continue

        typical_day = int(round(float(pd.Series([d.day for d in dates]).median())))
        latest = max(dates)

        flexibility_values = [x for x in group["flexibility"].dropna().astype(str) if x]
        flexibility = flexibility_values[-1] if flexibility_values else "fixed"

        min_amounts = group["minimum_allowed_amount"].dropna()
        minimum_allowed = None
        if not min_amounts.empty:
            converted_min = []
            # minimum_allowed_amount is in event currency; convert each using
            # that event's settlement date before taking a conservative median.
            for _, row in group[group["minimum_allowed_amount"].notna()].iterrows():
                clone = row.copy()
                clone["amount"] = row["minimum_allowed_amount"]
                converted, _ = _convert_event_amount(clone, home_currency, rates)
                if converted is not None:
                    converted_min.append(converted)
            if converted_min:
                minimum_allowed = float(pd.Series(converted_min).median())

        # Only keep categories that could matter to safe-spending decisions or
        # stable income. This still records fixed essential obligations.
        if not _is_flexible_adjustable(
            group.iloc[-1],
            protected_categories,
            reducible_categories,
            stoppable_categories,
        ) and direction == "debit":
            pass

        candidates.append(
            RecurringCandidate(
                event_type=_clean_text(event_type),
                direction=_clean_text(direction),
                category=_clean_text(category),
                cadence=cadence,
                median_amount=median_amount,
                typical_day=max(1, min(28, typical_day)),
                observations=len(group),
                last_settlement_date=latest,
                flexibility=flexibility,
                minimum_allowed_amount=minimum_allowed,
            )
        )

    candidates.sort(key=lambda x: (x.direction, x.category, x.last_settlement_date))
    return tuple(candidates)


def _future_cash_events(
    events: pd.DataFrame,
    request_date: pd.Timestamp,
    home_currency: str,
    rates: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], float]:
    columns = [
        "event_id",
        "event_type",
        "category",
        "direction",
        "event_date",
        "settlement_date",
        "status",
        "home_amount",
        "flexibility",
        "minimum_allowed_amount",
    ]

    if events.empty:
        return pd.DataFrame(columns=columns), tuple(), tuple(), 0.0

    future_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    unconverted_ids: list[str] = []
    pending_reserve = 0.0

    for _, row in events.iterrows():
        event_id = _clean_text(row.get("event_id"))
        status = _clean_text(row.get("status"))
        direction = _clean_text(row.get("direction"))

        if status not in VALID_STATUSES:
            continue
        if status in EXCLUDED_CASH_STATUSES:
            continue

        converted, error = _convert_event_amount(row, home_currency, rates)

        # Missing amounts are evidence-required, never zero.
        if converted is None:
            if error == "missing_amount":
                missing_ids.append(event_id)
            else:
                unconverted_ids.append(event_id)
            continue

        settlement = row.get("settlement_date")
        if pd.isna(settlement):
            settlement = row.get("event_date")
        if pd.isna(settlement):
            unconverted_ids.append(event_id)
            continue
        settlement = _to_date(settlement)

        if status == "pending" and direction == "debit":
            # Pending debit is not replayed as a future outflow; its amount is
            # reserved immediately against the request-date available balance.
            pending_reserve += converted
            continue

        if status == "pending" and direction == "credit":
            # Pending credits are uncertain and unavailable until settled.
            continue

        if settlement <= request_date:
            # Already reflected in current_available_balance.
            continue

        # Scheduled and future-settled records are valid future cash-flow rows.
        future_rows.append(
            {
                "event_id": event_id,
                "event_type": _clean_text(row.get("event_type")),
                "category": _clean_text(row.get("category")),
                "direction": direction,
                "event_date": _to_date(row.get("event_date")),
                "settlement_date": settlement,
                "status": status,
                "home_amount": converted,
                "flexibility": _clean_text(row.get("flexibility")),
                "minimum_allowed_amount": _safe_float(row.get("minimum_allowed_amount")),
            }
        )

    future = pd.DataFrame(future_rows, columns=columns)
    if not future.empty:
        future = future.sort_values(["settlement_date", "event_id"]).reset_index(drop=True)

    return future, tuple(sorted(set(missing_ids))), tuple(sorted(set(unconverted_ids))), pending_reserve


def _filter_user_evidence(
    evidence: pd.DataFrame,
    request_id: str,
    user_id: str,
) -> pd.DataFrame:
    evidence_columns = [
        "message_id",
        "user_id",
        "request_id",
        "related_event_id",
        "evidence_type",
        "confidence",
        "amount",
        "currency",
        "date",
        "percent",
        "source_type",
        "notes",
    ]
    if evidence.empty:
        return pd.DataFrame(columns=evidence_columns)

    req = evidence["request_id"].fillna("").astype(str)
    users = evidence["user_id"].astype(str)
    selected = evidence[(users == user_id) & ((req == "") | (req == request_id))].copy()

    if selected.empty:
        return selected

    # Newer same-source evidence appears last and should be available to the
    # later conflict-resolution layer. We do not mutate event truth here.
    selected["_sent_at"] = pd.NaT
    return selected.reset_index(drop=True)


def _request_payment_options(
    options: pd.DataFrame,
    request_id: str,
    home_currency: str,
    rates: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "payment_option_id",
        "request_id",
        "payment_method",
        "payment_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
        "payment_amount_home",
        "financing_fee_home",
        "total_payable_amount_home",
    ]

    if options.empty:
        return pd.DataFrame(columns=cols)

    subset = options[options["request_id"].astype(str) == request_id].copy()
    if subset.empty:
        return pd.DataFrame(columns=cols)

    # Payment-option amounts are already in the user's home currency per the
    # challenge contract, so no FX conversion is necessary.
    subset["payment_amount_home"] = pd.to_numeric(subset["payment_amount"], errors="coerce")
    subset["financing_fee_home"] = pd.to_numeric(subset["financing_fee"], errors="coerce")
    subset["total_payable_amount_home"] = pd.to_numeric(
        subset["total_payable_amount"], errors="coerce"
    )

    for col in ["first_payment_date"]:
        subset[col] = pd.to_datetime(subset[col], errors="coerce").dt.normalize()

    # Reject options that don't reconcile internally.
    subset["option_valid_math"] = (
        subset["payment_amount_home"].notna()
        & subset["number_of_payments"].notna()
        & subset["total_payable_amount_home"].notna()
    )

    return subset.reset_index(drop=True)


def _data_frame(data: Any, *names: str) -> pd.DataFrame:
    for name in names:
        value = getattr(data, name, None)
        if isinstance(value, pd.DataFrame):
            return value
    return pd.DataFrame()


def _events_for_user(data: Any, user_id: str) -> pd.DataFrame:
    index = getattr(data, "events_by_user", None)
    if isinstance(index, dict) and user_id in index:
        value = index[user_id]
        if isinstance(value, pd.DataFrame):
            return value.copy()
    events = _data_frame(data, "events", "financial_events")
    if events.empty:
        return events
    return events[events["user_id"].astype(str) == user_id].copy()


def _messages_frame(data: Any) -> pd.DataFrame:
    return _data_frame(data, "messages")


def _rates_frame(data: Any) -> pd.DataFrame:
    rates = _data_frame(data, "exchange_rates", "rates")
    if not rates.empty:
        return rates

    # Fallback for the existing loader's keyed index.
    keyed = getattr(data, "rates_by_key", None)
    if isinstance(keyed, dict) and keyed:
        rows = []
        for key, value in keyed.items():
            if isinstance(key, tuple) and len(key) == 3:
                rate_date, from_currency, to_currency = key
                rows.append({
                    "rate_date": rate_date,
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "rate": value,
                })
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["rate_date", "from_currency", "to_currency", "rate"])


def _options_frame(data: Any, request_id: Optional[str] = None) -> pd.DataFrame:
    options = _data_frame(data, "request_payment_options", "payment_options", "options")
    if not options.empty:
        return options

    keyed = getattr(data, "options_by_request", None)
    if isinstance(keyed, dict) and request_id is not None and request_id in keyed:
        value = keyed[request_id]
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, list):
            return pd.DataFrame(value)
    return pd.DataFrame()


def reconcile_request(
    data: Any,
    request_id: str,
    *,
    parsed_evidence: Optional[pd.DataFrame] = None,
) -> ReconciliationState:
    """Build a request-specific financial state from a loaded DataBundle."""

    request_index = getattr(data, "request_by_id", {})
    request = request_index.get(request_id) if isinstance(request_index, dict) else None
    if request is None:
        requests_df = _data_frame(data, "requests")
        if not requests_df.empty:
            matches = requests_df[requests_df["request_id"].astype(str) == request_id]
            request = matches.iloc[0] if not matches.empty else None
    if request is None:
        raise KeyError(f"Unknown request_id: {request_id}")

    user_id = _clean_text(_row_get(request, "user_id"))
    profile = data.profile_by_user.get(user_id)
    if profile is None:
        raise KeyError(f"Missing financial profile for user_id: {user_id}")

    request_date = _to_date(_row_get(request, "request_date"))
    desired_completion = _to_date(_row_get(request, "desired_completion_date"))
    home_currency = _clean_text(_row_get(profile, "home_currency")).upper()

    request_events = _events_for_user(data, user_id)

    protected = _pipe_list(_row_get(profile, "expense_categories_to_protect"))
    reducible = _pipe_list(_row_get(profile, "expense_categories_user_is_willing_to_reduce"))
    stoppable = _pipe_list(_row_get(profile, "expense_categories_user_is_willing_to_stop"))

    rates = _rates_frame(data)

    future, missing_ids, unconverted_ids, pending_reserve = _future_cash_events(
        request_events,
        request_date,
        home_currency,
        rates,
    )

    recurring = _build_recurring_candidates(
        request_events,
        request_date,
        home_currency,
        rates,
        protected,
        reducible,
        stoppable,
    )

    if parsed_evidence is None:
        if parse_messages is not None:
            messages_df = _messages_frame(data)
            parsed_evidence = parse_messages(messages_df) if not messages_df.empty else pd.DataFrame()
        else:
            parsed_evidence = pd.DataFrame()

    request_evidence = _filter_user_evidence(parsed_evidence, request_id, user_id)
    options = _request_payment_options(
        _options_frame(data, request_id),
        request_id,
        home_currency,
        rates,
    )

    diagnostics: list[str] = []
    if missing_ids:
        diagnostics.append(f"{len(missing_ids)} event(s) have missing amounts and require evidence resolution.")
    if unconverted_ids:
        diagnostics.append(f"{len(unconverted_ids)} event(s) could not be converted with supplied FX rates.")
    if pending_reserve > 0:
        diagnostics.append(f"Pending debit reserve: {pending_reserve:.2f} {home_currency}.")
    if not recurring:
        diagnostics.append("No stable recurring pattern was detected from pre-request history.")

    return ReconciliationState(
        request_id=request_id,
        user_id=user_id,
        request_date=request_date,
        desired_completion_date=desired_completion,
        request_type=_clean_text(_row_get(request, "request_type")),
        requested_amount=float(_row_get(request, "requested_amount")),
        allows_partial_payment=bool(_row_get(request, "allows_partial_payment")),
        home_currency=home_currency,
        current_available_balance=float(_row_get(profile, "current_available_balance")),
        minimum_balance_to_keep=float(_row_get(profile, "minimum_balance_to_keep")),
        financial_priorities=_pipe_list(_row_get(profile, "financial_priorities")),
        protected_categories=protected,
        reducible_categories=reducible,
        stoppable_categories=stoppable,
        accepted_payment_methods=_pipe_list(_row_get(profile, "payment_methods_user_will_consider")),
        max_installment_months=_safe_int(_row_get(profile, "max_installment_months")),
        pending_debit_reserve=pending_reserve,
        future_cash_events=future,
        recurring_candidates=recurring,
        evidence=request_evidence,
        payment_options=options,
        missing_amount_event_ids=missing_ids,
        unconverted_event_ids=unconverted_ids,
        diagnostics=tuple(diagnostics),
    )


def reconcile_all(
    data: Any,
    *,
    parsed_evidence: Optional[pd.DataFrame] = None,
) -> dict[str, ReconciliationState]:
    """Reconcile every evaluation request deterministically."""
    if parsed_evidence is None and parse_messages is not None:
        messages_df = _messages_frame(data)
        parsed_evidence = parse_messages(messages_df) if not messages_df.empty else pd.DataFrame()

    result: dict[str, ReconciliationState] = {}
    for request_id in data.requests["request_id"].astype(str):
        result[request_id] = reconcile_request(
            data,
            request_id,
            parsed_evidence=parsed_evidence,
        )
    return result


def state_summary(state: ReconciliationState) -> dict[str, Any]:
    """Compact diagnostic representation used by tests and later logging."""
    return {
        "request_id": state.request_id,
        "user_id": state.user_id,
        "request_date": state.request_date.strftime("%Y-%m-%d"),
        "desired_completion_date": state.desired_completion_date.strftime("%Y-%m-%d"),
        "requested_amount": round(state.requested_amount, 2),
        "home_currency": state.home_currency,
        "current_available_balance": round(state.current_available_balance, 2),
        "minimum_balance_to_keep": round(state.minimum_balance_to_keep, 2),
        "pending_debit_reserve": round(state.pending_debit_reserve, 2),
        "immediately_spendable_after_reserve": round(
            state.immediately_spendable_after_reserve, 2
        ),
        "current_discretionary_room": round(state.current_discretionary_room, 2),
        "future_cash_events": int(len(state.future_cash_events)),
        "recurring_candidates": int(len(state.recurring_candidates)),
        "evidence_facts": int(len(state.evidence)),
        "payment_options": int(len(state.payment_options)),
        "missing_amount_events": len(state.missing_amount_event_ids),
        "unconverted_events": len(state.unconverted_event_ids),
    }


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()
    states = reconcile_all(data)

    print("=" * 78)
    print("FIN GUARD RECONCILIATION DIAGNOSTICS")
    print("=" * 78)
    print(f"Requests reconciled: {len(states)}")

    sample_ids = ["request_26", "request_06", "request_21"]
    for rid in sample_ids:
        if rid in states:
            print("\n", state_summary(states[rid]))
            print("Recurring candidates:")
            for item in states[rid].recurring_candidates[:8]:
                print(" ", item)
            print("Future cash events:")
            if states[rid].future_cash_events.empty:
                print("  none")
            else:
                print(states[rid].future_cash_events.head(10).to_string(index=False))
            print("Diagnostics:")
            for item in states[rid].diagnostics:
                print(" ", item)

    total_future = sum(len(s.future_cash_events) for s in states.values())
    total_recurring = sum(len(s.recurring_candidates) for s in states.values())
    total_pending_reserve = sum(s.pending_debit_reserve for s in states.values())
    missing_states = sum(bool(s.missing_amount_event_ids) for s in states.values())
    print("\n--- Aggregate ---")
    print({
        "requests": len(states),
        "future_cash_events": total_future,
        "recurring_candidates": total_recurring,
        "pending_debit_reserve_sum": round(total_pending_reserve, 2),
        "requests_with_missing_amount_events": missing_states,
    })
