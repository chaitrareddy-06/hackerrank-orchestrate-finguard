from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional
import calendar

import pandas as pd


OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

EPS = 1e-6


@dataclass(frozen=True)
class SpendAction:
    category: str
    event_id: str
    mode: str
    new_amount: float

    @property
    def label(self) -> str:
        if self.mode == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{_fmt_amount(self.new_amount)}"


@dataclass(frozen=True)
class ForecastResult:
    dates: tuple[pd.Timestamp, ...]
    balances: tuple[float, ...]
    event_labels: tuple[str, ...] = ()

    def headroom(self, minimum_balance: float) -> float:
        if not self.balances:
            return 0.0
        return min(self.balances) - minimum_balance


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _fmt_amount(value: float) -> str:
    if abs(value - round(value)) < 1e-8:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _date(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value!r}")
    return ts.normalize()


def _calendar_add_months(date: pd.Timestamp, months: int) -> pd.Timestamp:
    total = date.year * 12 + (date.month - 1) + months
    year = total // 12
    month = total % 12 + 1
    day = min(date.day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day)


def _next_occurrence(last_date: pd.Timestamp, cadence: str, step: int = 1) -> pd.Timestamp:
    c = cadence.lower()
    if c == "daily":
        return last_date + pd.Timedelta(days=step)
    if c == "weekly":
        return last_date + pd.Timedelta(days=7 * step)
    if c == "biweekly":
        return last_date + pd.Timedelta(days=14 * step)
    if c == "monthly":
        return _calendar_add_months(last_date, step)
    if c == "quarterly":
        return _calendar_add_months(last_date, 3 * step)
    if c == "yearly" or c == "annual":
        return _calendar_add_months(last_date, 12 * step)
    return pd.NaT


def _occurrences(candidate: Any, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    last = _date(candidate.last_settlement_date)
    cadence = _clean(candidate.cadence).lower()
    out: list[pd.Timestamp] = []

    if cadence not in {"daily", "weekly", "biweekly", "monthly", "quarterly", "yearly", "annual"}:
        return out

    current = last
    guard = 0
    while guard < 5000:
        nxt = _next_occurrence(current, cadence)
        if pd.isna(nxt):
            break
        current = nxt
        guard += 1
        if current > end:
            break
        if current >= start:
            out.append(current)

    return out


def _state_event_frame(state: Any, data: Any) -> pd.DataFrame:
    """Return this user's raw event history, preferring reconciliation's helper."""
    try:
        from reconciliation import _events_for_user
        frame = _events_for_user(data, state.user_id)
        return frame.copy()
    except Exception:
        events = getattr(data, "events", None)
        if isinstance(events, pd.DataFrame):
            return events[events["user_id"].astype(str) == str(state.user_id)].copy()
        return pd.DataFrame()


def _representative_flexible_events(state: Any, data: Any) -> list[SpendAction]:
    """
    Build at most one actionable representation per flexible recurring category.
    The event_id comes from the most recent pre-request event in that category.
    """
    events = _state_event_frame(state, data)
    if events.empty:
        return []

    request_date = _date(state.request_date)
    rows = events.copy()

    if "settlement_date" in rows.columns:
        rows["_d"] = pd.to_datetime(rows["settlement_date"], errors="coerce").dt.normalize()
    else:
        rows["_d"] = pd.to_datetime(rows.get("event_date"), errors="coerce").dt.normalize()

    rows = rows[
        (rows["_d"] <= request_date)
        & (rows["direction"].astype(str).str.lower() == "debit")
        & rows["flexibility"].astype(str).str.lower().isin(
            {"reducible", "stoppable", "reducible_or_stoppable"}
        )
    ].copy()

    actions: list[SpendAction] = []
    seen: set[str] = set()

    candidates = list(getattr(state, "recurring_candidates", ()) or ())

    # Build a category -> candidate map only for debit recurring expenses.
    candidate_by_cat = {
        _clean(c.category).lower(): c
        for c in candidates
        if _clean(c.direction).lower() == "debit"
    }

    for category_key, candidate in candidate_by_cat.items():
        category_rows = rows[rows["category"].astype(str).str.lower() == category_key]
        if category_rows.empty:
            continue

        # Protected categories can never be touched.
        if category_key in {x.lower() for x in getattr(state, "protected_categories", ())}:
            continue

        flexibility = _clean(candidate.flexibility).lower()
        stoppable_allowed = category_key in {
            x.lower() for x in getattr(state, "stoppable_categories", ())
        }
        reducible_allowed = category_key in {
            x.lower() for x in getattr(state, "reducible_categories", ())
        }

        mode: Optional[str] = None
        new_amount: Optional[float] = None

        if stoppable_allowed and "stoppable" in flexibility:
            mode = "stop"
            new_amount = 0.0
        elif reducible_allowed and flexibility in {"reducible", "reducible_or_stoppable"}:
            minimum = candidate.minimum_allowed_amount
            if minimum is not None and minimum < candidate.median_amount - EPS:
                mode = "reduce"
                new_amount = float(minimum)

        if mode is None:
            continue

        # Most recent event for this recurring category.
        category_rows = category_rows.sort_values("_d", ascending=False)
        event_id = _clean(category_rows.iloc[0].get("event_id"))
        if not event_id or event_id in seen:
            continue

        seen.add(event_id)
        actions.append(
            SpendAction(
                category=_clean(candidate.category),
                event_id=event_id,
                mode=mode,
                new_amount=float(new_amount),
            )
        )

    return actions


def _build_cash_stream(
    state: Any,
    *,
    changes: Iterable[SpendAction] = (),
    horizon_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a daily conservative cash forecast.

    Baseline rules:
    - Start with current available balance after pending-debit reserve.
    - Apply known future cash events on their settlement date.
    - Forecast recurrence only where reconciliation found a stable cadence.
    - Flexible recurring categories can be stopped/reduced by an explicit action.
    """
    request_date = _date(state.request_date)
    desired = _date(state.desired_completion_date)

    if horizon_days is None:
        horizon_end = max(desired, request_date + pd.Timedelta(days=365))
    else:
        horizon_end = max(desired, request_date + pd.Timedelta(days=horizon_days))

    action_map = {a.category.lower(): a for a in changes}

    days = pd.date_range(request_date, horizon_end, freq="D")
    cash_by_date = {d: 0.0 for d in days}

    # Known future cash flows.
    future = getattr(state, "future_cash_events", None)
    if isinstance(future, pd.DataFrame) and not future.empty:
        for _, row in future.iterrows():
            d = pd.to_datetime(row.get("settlement_date"), errors="coerce")
            if pd.isna(d):
                continue
            d = d.normalize()
            if d < request_date or d > horizon_end:
                continue

            amount = float(row.get("home_amount", 0.0) or 0.0)
            direction = _clean(row.get("direction")).lower()
            if direction == "credit":
                cash_by_date[d] += amount
            elif direction == "debit":
                cash_by_date[d] -= amount

    # Forecast recurring cash flows.
    # Skip a generated occurrence if an explicit future cash event already
    # represents the same category/direction on that date.
    known_keys: set[tuple[str, str, pd.Timestamp]] = set()
    if isinstance(future, pd.DataFrame) and not future.empty:
        for _, row in future.iterrows():
            d = pd.to_datetime(row.get("settlement_date"), errors="coerce")
            if pd.isna(d):
                continue
            known_keys.add(
                (_clean(row.get("category")).lower(),
                 _clean(row.get("direction")).lower(),
                 d.normalize())
            )

    for candidate in getattr(state, "recurring_candidates", ()) or ():
        category = _clean(candidate.category)
        category_key = category.lower()
        direction = _clean(candidate.direction).lower()
        amount = float(candidate.median_amount)

        action = action_map.get(category_key)
        if direction == "debit" and action is not None:
            amount = float(action.new_amount)

        for d in _occurrences(candidate, request_date, horizon_end):
            key = (category_key, direction, d.normalize())
            if key in known_keys:
                continue

            if direction == "credit":
                cash_by_date[d] += amount
            elif direction == "debit":
                cash_by_date[d] -= amount

    # Daily balances after all flows on the day.
    opening = (
        float(state.current_available_balance)
        - float(state.pending_debit_reserve)
    )

    balance = opening
    rows: list[dict[str, Any]] = []

    for d in days:
        balance += cash_by_date[d]
        rows.append({"date": d, "balance": balance, "net_cash": cash_by_date[d]})

    return pd.DataFrame(rows)


def _full_payment_safe_on_or_after(
    forecast: pd.DataFrame,
    requested_amount: float,
    minimum_balance: float,
    start_date: pd.Timestamp,
) -> Optional[pd.Timestamp]:
    if forecast.empty:
        return None

    f = forecast[forecast["date"] >= start_date].copy()
    if f.empty:
        return None

    f["headroom"] = f["balance"] - minimum_balance
    # For a payment on date d to be safe, every later projected balance
    # must retain at least requested_amount of headroom.
    future_min = f["headroom"][::-1].cummin()[::-1]
    for idx, value in future_min.items():
        if value + EPS >= requested_amount:
            return _date(f.loc[idx, "date"])
    return None


def _safe_amount_now(
    baseline: pd.DataFrame,
    requested_amount: float,
    minimum_balance: float,
) -> float:
    if baseline.empty:
        return 0.0

    min_balance = float(baseline["balance"].min())
    safe = min_balance - minimum_balance
    return max(0.0, min(float(requested_amount), safe))


def _apply_payments_is_safe(
    forecast: pd.DataFrame,
    payments: list[tuple[pd.Timestamp, float]],
    minimum_balance: float,
) -> bool:
    if forecast.empty:
        return False

    payment_map: dict[pd.Timestamp, float] = {}
    for date, amount in payments:
        payment_map[_date(date)] = payment_map.get(_date(date), 0.0) + float(amount)

    running = 0.0
    # Once a payment happens, it permanently reduces all subsequent balances.
    for _, row in forecast.iterrows():
        d = _date(row["date"])
        running += payment_map.get(d, 0.0)
        balance_after = float(row["balance"]) - running
        if balance_after + EPS < minimum_balance:
            return False
    return True


def _accepted(state: Any, method: str) -> bool:
    return method in {str(x).strip().lower() for x in getattr(state, "accepted_payment_methods", ())}


def _eligible_installment_options(state: Any) -> pd.DataFrame:
    options = getattr(state, "payment_options", None)
    if not isinstance(options, pd.DataFrame) or options.empty:
        return pd.DataFrame()

    out = options.copy()
    out["payment_method"] = out["payment_method"].astype(str).str.lower()
    out = out[out["payment_method"] == "installments"].copy()

    if not _accepted(state, "installments"):
        return out.iloc[0:0].copy()

    max_months = getattr(state, "max_installment_months", None)
    if max_months is not None and "number_of_payments" in out.columns:
        out = out[
            pd.to_numeric(out["number_of_payments"], errors="coerce")
            <= float(max_months)
        ].copy()

    return out


def _option_payments(option: pd.Series) -> list[tuple[pd.Timestamp, float]]:
    first = pd.to_datetime(option.get("first_payment_date"), errors="coerce")
    amount = float(option.get("payment_amount_home"))
    n = int(option.get("number_of_payments"))
    freq = int(option.get("payment_frequency_days"))

    if pd.isna(first) or n <= 0 or freq <= 0 or amount < 0:
        return []

    return [
        (_date(first) + pd.Timedelta(days=i * freq), amount)
        for i in range(n)
    ]


def _installment_candidates(
    state: Any,
    baseline: pd.DataFrame,
) -> list[dict[str, Any]]:
    options = _eligible_installment_options(state)
    if options.empty:
        return []

    desired = _date(state.desired_completion_date)
    candidates: list[dict[str, Any]] = []

    for _, option in options.iterrows():
        payments = _option_payments(option)
        if not payments:
            continue

        if payments[-1][0] > desired:
            continue

        expected_total = float(option.get("total_payable_amount_home"))
        calculated_total = sum(amount for _, amount in payments)
        if abs(expected_total - calculated_total) > max(0.02, 1e-6 * max(1.0, expected_total)):
            continue

        if _apply_payments_is_safe(baseline, payments, float(state.minimum_balance_to_keep)):
            candidates.append(
                {
                    "option_id": _clean(option.get("payment_option_id")),
                    "payments": payments,
                    "total_cost": expected_total,
                    "count": len(payments),
                    "first_date": payments[0][0],
                }
            )

    candidates.sort(
        key=lambda x: (
            x["total_cost"],
            x["first_date"],
            x["count"],
            x["option_id"],
        )
    )
    return candidates


def _spending_change_sets(
    actions: list[SpendAction],
    max_actions: int = 3,
) -> list[tuple[SpendAction, ...]]:
    if not actions:
        return [tuple()]

    sets: list[tuple[SpendAction, ...]] = [tuple()]
    limit = min(max_actions, len(actions))

    for size in range(1, limit + 1):
        for combo in combinations(actions, size):
            sets.append(combo)

    # Prefer the smallest number of changes first. For equal sizes,
    # larger savings first is handled later by the forecast evaluation.
    return sets


def _safe_with_changes(
    state: Any,
    data: Any,
    requested_amount: float,
    candidate_sets: list[tuple[SpendAction, ...]],
) -> Optional[dict[str, Any]]:
    desired = _date(state.desired_completion_date)
    best: Optional[dict[str, Any]] = None

    for changes in candidate_sets:
        # This helper is specifically for plans that require permitted
        # spending changes. The empty set is handled by the normal baseline
        # logic and must not be classified as a spending-change plan.
        if not changes:
            continue

        forecast = _build_cash_stream(state, changes=changes)
        full_now = _apply_payments_is_safe(
            forecast,
            [(_date(state.request_date), requested_amount)],
            float(state.minimum_balance_to_keep),
        )

        earliest = _full_payment_safe_on_or_after(
            forecast,
            requested_amount,
            float(state.minimum_balance_to_keep),
            _date(state.request_date),
        )

        changed_now = full_now
        completes_by_deadline = full_now or (
            earliest is not None and earliest <= desired
        )
        if not completes_by_deadline:
            continue

        score = (
            len(changes),
            sum(
                0.0 if a.mode == "stop" else max(0.0, a.new_amount)
                for a in changes
            ),
            0 if full_now else 1,
            earliest or desired + pd.Timedelta(days=99999),
        )

        candidate = {
            "changes": changes,
            "forecast": forecast,
            "full_now": changed_now,
            "earliest": earliest,
            "score": score,
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate

    return best


def _baseline_safe_amount_and_date(
    state: Any,
    requested_amount: float,
    baseline: pd.DataFrame,
) -> tuple[float, Optional[pd.Timestamp]]:
    safe = _safe_amount_now(
        baseline,
        requested_amount,
        float(state.minimum_balance_to_keep),
    )
    earliest = _full_payment_safe_on_or_after(
        baseline,
        requested_amount,
        float(state.minimum_balance_to_keep),
        _date(state.request_date),
    )
    return safe, earliest


def _format_plan(payments: list[tuple[pd.Timestamp, float]]) -> str:
    return "|".join(f"{d:%Y-%m-%d}:{_fmt_amount(float(a))}" for d, a in payments)


def _change_string(changes: Iterable[SpendAction]) -> str:
    labels = [a.label for a in changes]
    return "|".join(labels) if labels else "none"


def _explanation(
    *,
    state: Any,
    requested: float,
    safe_now: float,
    method: str,
    status: str,
    changes: Iterable[SpendAction],
    earliest: Optional[pd.Timestamp],
    option: Optional[dict[str, Any]] = None,
) -> str:
    currency = _clean(state.home_currency)
    minimum = float(state.minimum_balance_to_keep)
    balance = float(state.current_available_balance) - float(state.pending_debit_reserve)
    parts = [
        f"Request {_money(requested)} {currency}; spendable balance after pending-debit reserve is {_money(balance)} {currency}.",
        f"Conservative safe amount on request date is {_money(safe_now)} {currency} while keeping the required minimum of {_money(minimum)} {currency}.",
    ]

    if method == "full_payment" and status == "affordable_now":
        parts.append("Full payment is safe now and the forecast stays above the minimum balance.")
    elif method == "full_payment" and changes:
        parts.append("Full payment is made safe by permitted changes to flexible recurring spending.")
    elif method == "installments" and option:
        parts.append(
            f"An allowed installment option completes by {option['payments'][-1][0]:%Y-%m-%d}; "
            f"total payable is {_money(option['total_cost'])} {currency}."
        )
    elif method == "partial_payment":
        parts.append("A two-payment schedule is feasible within the requested deadline.")
    elif method == "wait":
        if earliest is not None:
            parts.append(f"One full payment becomes safe on {earliest:%Y-%m-%d}.")
        else:
            parts.append("No conservative full-payment date is available within the forecast horizon.")
    else:
        parts.append("No safe completion plan using the permitted payment methods is available.")

    if changes:
        parts.append(f"Spending changes: {_change_string(changes)}.")

    priorities = getattr(state, "financial_priorities", ())
    if priorities:
        parts.append(f"Priorities retained: {', '.join(priorities)}.")

    return " ".join(parts)


def decide_request(state: Any, data: Any) -> dict[str, Any]:
    requested = float(state.requested_amount)
    request_date = _date(state.request_date)
    desired = _date(state.desired_completion_date)
    minimum = float(state.minimum_balance_to_keep)

    if requested <= 0:
        return {
            "request_id": state.request_id,
            "amount_safe_to_pay": 0.0,
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "payment_plan": f"{request_date:%Y-%m-%d}:0",
            "earliest_date_for_full_payment": f"{request_date:%Y-%m-%d}",
            "spending_changes_needed": "none",
            "decision_explanation": "Requested amount is zero, so no cash commitment is required.",
        }

    baseline = _build_cash_stream(state)
    safe_now, earliest = _baseline_safe_amount_and_date(
        state, requested, baseline
    )

    # 1. Full payment now.
    if _apply_payments_is_safe(baseline, [(request_date, requested)], minimum):
        return {
            "request_id": state.request_id,
            "amount_safe_to_pay": safe_now,
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "payment_plan": _format_plan([(request_date, requested)]),
            "earliest_date_for_full_payment": f"{request_date:%Y-%m-%d}",
            "spending_changes_needed": "none",
            "decision_explanation": _explanation(
                state=state,
                requested=requested,
                safe_now=safe_now,
                method="full_payment",
                status="affordable_now",
                changes=(),
                earliest=request_date,
            ),
        }

    # 2. Consider installment options that are exactly supplied and allowed.
    installment_candidates = _installment_candidates(state, baseline)
    installment = installment_candidates[0] if installment_candidates else None

    # 3. Consider permitted spending changes that can make full payment safe.
    action_candidates = _representative_flexible_events(state, data)
    change_sets = _spending_change_sets(action_candidates, max_actions=3)
    changed = _safe_with_changes(state, data, requested, change_sets)

    # 4. Partial payment is allowed only under the exact challenge conditions.
    partial_candidate: Optional[dict[str, Any]] = None
    if (
        state.allows_partial_payment
        and _accepted(state, "partial_payment")
        and 0.0 < safe_now < requested
        and earliest is not None
        and earliest <= desired
    ):
        second = requested - safe_now
        payments = [(request_date, safe_now), (earliest, second)]
        # Explicit deterministic simulation.
        if _apply_payments_is_safe(baseline, payments, minimum):
            partial_candidate = {
                "payments": payments,
                "total_cost": requested,
                "count": 2,
                "first_date": request_date,
            }

    # 5. Compare safe full-payment via spending changes against installments and partial.
    candidates: list[dict[str, Any]] = []

    if changed is not None:
        changes = changed["changes"]
        if changed["full_now"]:
            earliest_changed = request_date
        else:
            earliest_changed = changed["earliest"]

        if earliest_changed is not None and earliest_changed <= desired:
            candidates.append(
                {
                    "method": "full_payment",
                    "status": "affordable_with_plan",
                    "payments": [(request_date if changed["full_now"] else earliest_changed, requested)],
                    "total_cost": requested,
                    "change_count": len(changes),
                    "changes": changes,
                    "start": request_date if changed["full_now"] else earliest_changed,
                    "rank_type": 0,
                }
            )

    if installment is not None:
        candidates.append(
            {
                "method": "installments",
                "status": "affordable_with_plan",
                "payments": installment["payments"],
                "total_cost": installment["total_cost"],
                "change_count": 0,
                "changes": tuple(),
                "start": installment["first_date"],
                "rank_type": 1,
                "option": installment,
            }
        )

    if partial_candidate is not None:
        candidates.append(
            {
                "method": "partial_payment",
                "status": "affordable_with_plan",
                "payments": partial_candidate["payments"],
                "total_cost": partial_candidate["total_cost"],
                "change_count": 0,
                "changes": tuple(),
                "start": request_date,
                "rank_type": 1,
            }
        )

    if candidates:
        # The challenge's qualitative preference is:
        # deadline completion -> avoid changes -> minimize total cost ->
        # start earlier -> fewer payments.
        # All candidates here already complete by the deadline.
        candidates.sort(
            key=lambda x: (
                0 if x["change_count"] == 0 else 1,
                x["total_cost"],
                x["start"],
                len(x["payments"]),
                x["rank_type"],
            )
        )
        chosen = candidates[0]
        return {
            "request_id": state.request_id,
            "amount_safe_to_pay": safe_now,
            "affordability_status": chosen["status"],
            "recommended_payment_method": chosen["method"],
            "payment_plan": _format_plan(chosen["payments"]),
            "earliest_date_for_full_payment": (
                f"{chosen['start']:%Y-%m-%d}"
                if chosen["method"] == "full_payment" and chosen["changes"]
                else (f"{earliest:%Y-%m-%d}" if earliest is not None else "")
            ),
            "spending_changes_needed": _change_string(chosen["changes"]),
            "decision_explanation": _explanation(
                state=state,
                requested=requested,
                safe_now=safe_now,
                method=chosen["method"],
                status=chosen["status"],
                changes=chosen["changes"],
                earliest=earliest,
                option=chosen.get("option"),
            ),
        }

    # 6. No immediate completion plan. Wait if a conservative full-pay date
    # exists, even if it misses the requested deadline.
    if earliest is not None:
        return {
            "request_id": state.request_id,
            "amount_safe_to_pay": safe_now,
            "affordability_status": "affordable_later",
            "recommended_payment_method": "wait",
            "payment_plan": _format_plan([(earliest, requested)]),
            "earliest_date_for_full_payment": f"{earliest:%Y-%m-%d}",
            "spending_changes_needed": "none",
            "decision_explanation": _explanation(
                state=state,
                requested=requested,
                safe_now=safe_now,
                method="wait",
                status="affordable_later",
                changes=(),
                earliest=earliest,
            ),
        }

    # 7. Otherwise not affordable within the forecast horizon.
    return {
        "request_id": state.request_id,
        "amount_safe_to_pay": max(0.0, min(safe_now, requested)),
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": _explanation(
            state=state,
            requested=requested,
            safe_now=safe_now,
            method="not_recommended",
            status="not_affordable",
            changes=(),
            earliest=None,
        ),
    }


def validate_decision(row: dict[str, Any], state: Any) -> None:
    """Deterministic contract validation for a single output row."""
    required_methods = {
        "full_payment",
        "partial_payment",
        "installments",
        "wait",
        "not_recommended",
    }
    required_statuses = {
        "affordable_now",
        "affordable_with_plan",
        "affordable_later",
        "not_affordable",
    }

    safe = float(row["amount_safe_to_pay"])
    if safe < -EPS or safe > float(state.requested_amount) + EPS:
        raise ValueError(f"{state.request_id}: amount_safe_to_pay out of bounds")

    if row["affordability_status"] not in required_statuses:
        raise ValueError(f"{state.request_id}: invalid affordability_status")

    if row["recommended_payment_method"] not in required_methods:
        raise ValueError(f"{state.request_id}: invalid payment method")

    changes = str(row["spending_changes_needed"])
    if changes != "none" and len(changes.split("|")) > 3:
        raise ValueError(f"{state.request_id}: too many spending changes")

    if row["recommended_payment_method"] == "partial_payment":
        parts = str(row["payment_plan"]).split("|")
        if len(parts) != 2:
            raise ValueError(f"{state.request_id}: partial payment must contain exactly two payments")

        first_date, first_amount = parts[0].split(":", 1)
        second_date, second_amount = parts[1].split(":", 1)
        _ = _date(first_date)
        second_dt = _date(second_date)
        a1 = float(first_amount)
        a2 = float(second_amount)

        if not state.allows_partial_payment or not _accepted(state, "partial_payment"):
            raise ValueError(f"{state.request_id}: partial payment not allowed")
        if not (0 < a1 < float(state.requested_amount)):
            raise ValueError(f"{state.request_id}: invalid first partial payment")
        if abs(a1 - safe) > 0.02:
            raise ValueError(f"{state.request_id}: first partial payment must equal amount_safe_to_pay")
        if abs((a1 + a2) - float(state.requested_amount)) > 0.02:
            raise ValueError(f"{state.request_id}: partial payments do not sum to request")
        if second_dt > _date(state.desired_completion_date):
            raise ValueError(f"{state.request_id}: partial plan misses deadline")

    if row["recommended_payment_method"] == "installments":
        # Verify every output payment date/amount exists in the chosen supplied option.
        option_match = None
        options = getattr(state, "payment_options", pd.DataFrame())
        if isinstance(options, pd.DataFrame) and not options.empty:
            plan = str(row["payment_plan"])
            for _, option in _eligible_installment_options(state).iterrows():
                expected = _format_plan(_option_payments(option))
                if expected == plan:
                    option_match = option
                    break
        if option_match is None:
            raise ValueError(f"{state.request_id}: installment plan does not match supplied option")

    method = row["recommended_payment_method"]
    status = row["affordability_status"]

    valid_pairs = {
        ("affordable_now", "full_payment"),
        ("affordable_with_plan", "full_payment"),
        ("affordable_with_plan", "partial_payment"),
        ("affordable_with_plan", "installments"),
        ("affordable_later", "wait"),
        ("not_affordable", "not_recommended"),
    }
    if (status, method) not in valid_pairs:
        raise ValueError(
            f"{state.request_id}: invalid status/method combination: {status}/{method}"
        )

    if method == "full_payment" and status == "affordable_now":
        request_date = _date(state.request_date)
        if not str(row["payment_plan"]).startswith(f"{request_date:%Y-%m-%d}:"):
            raise ValueError(f"{state.request_id}: affordable_now full payment must start on request date")

    if method == "full_payment" and status == "affordable_with_plan":
        if str(row["spending_changes_needed"]) == "none":
            raise ValueError(
                f"{state.request_id}: affordable_with_plan full payment requires spending changes"
            )


def decide_all(states: dict[str, Any], data: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for request_id, state in states.items():
        row = decide_request(state, data)
        validate_decision(row, state)
        rows.append(row)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Preserve request order exactly as reconciled.
    request_order = [str(x) for x in data.requests["request_id"]]
    result["_order"] = pd.Categorical(
        result["request_id"], categories=request_order, ordered=True
    )
    result = result.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    return result

