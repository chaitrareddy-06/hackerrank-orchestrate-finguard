from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# Cash-bearing statuses that may affect the financial timeline.
ACTIVE_CASH_STATUSES = {"settled", "pending", "scheduled"}

# Explicitly inactive transactions.
INACTIVE_STATUSES = {"cancelled", "failed"}

# Investment valuation changes are non-cash.
NON_CASH_EVENT_TYPES = {"investment_valuation"}


@dataclass(frozen=True)
class EventDecision:
    """
    Classification of a financial event relative to a request date.

    include_in_forecast:
        Whether the event should be applied as a future cash-flow event.

    reserve_now:
        Whether a pending debit must be reserved against the current
        available balance.

    cash_effect:
        +amount for a cash credit
        -amount for a cash debit
         0 for ignored/non-cash events.

    cash_date:
        Date on which the event affects the future cash timeline, if any.

    reason:
        Human-readable audit explanation.
    """

    include_in_forecast: bool
    reserve_now: bool
    cash_effect: float
    cash_date: Optional[pd.Timestamp]
    reason: str


def _cash_sign(direction: str) -> int:
    """Return the cash-flow sign for an event direction."""

    if direction == "credit":
        return 1

    if direction == "debit":
        return -1

    return 0


def classify_event(
    event: pd.Series,
    request_date: pd.Timestamp,
) -> EventDecision:
    """
    Classify one financial event relative to a request date.

    Current rules implemented here:

    - cancelled and failed events do not affect cash flow
    - unrealized investment valuation is non-cash
    - pending debit is reserved immediately
    - pending credit is NOT treated as available income
    - scheduled cash is applied on settlement date
    - settled cash is applied on settlement date
    - historical events do not alter the profile's current balance anchor
    """

    status = str(event["status"]).strip().lower()
    direction = str(event["direction"]).strip().lower()
    event_type = str(event["event_type"]).strip().lower()

    amount = event["amount"]

    # ------------------------------------------------------------------
    # Missing amount
    # ------------------------------------------------------------------
    # These events must be resolved through evidence before they can
    # participate in cash calculations.
    if pd.isna(amount):
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason="blank amount; requires evidence resolution",
        )

    amount = float(amount)

    # ------------------------------------------------------------------
    # Non-cash investment valuation
    # ------------------------------------------------------------------
    if event_type in NON_CASH_EVENT_TYPES or direction == "non_cash":
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason="non-cash investment valuation",
        )

    # ------------------------------------------------------------------
    # Cancelled / failed
    # ------------------------------------------------------------------
    if status in INACTIVE_STATUSES:
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason=f"{status} event excluded from cash flow",
        )

    # ------------------------------------------------------------------
    # Unknown state
    # ------------------------------------------------------------------
    if status not in ACTIVE_CASH_STATUSES:
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason=f"unsupported status: {status}",
        )

    settlement_date = event.get("settlement_date")

    if pd.isna(settlement_date):
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason="active event has no settlement date",
        )

    settlement_date = pd.Timestamp(settlement_date).normalize()
    request_date = pd.Timestamp(request_date).normalize()

    sign = _cash_sign(direction)

    if sign == 0:
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason=f"non-cash direction: {direction}",
        )

    cash_effect = sign * amount

    # ------------------------------------------------------------------
    # Pending
    # ------------------------------------------------------------------
    if status == "pending":

        # Pending debit:
        #
        # current_available_balance is our starting cash anchor.
        # Therefore reserve this debit immediately and do NOT apply the
        # same debit a second time on its later settlement date.
        if direction == "debit":
            return EventDecision(
                include_in_forecast=False,
                reserve_now=True,
                cash_effect=0.0,
                cash_date=settlement_date,
                reason=(
                    "pending debit; reserve against current safe cash "
                    "and do not apply again in future forecast"
                ),
            )

        # Pending credit:
        # It is not available money yet.
        return EventDecision(
            include_in_forecast=False,
            reserve_now=False,
            cash_effect=0.0,
            cash_date=None,
            reason="pending credit is not yet available income",
        )

    # ------------------------------------------------------------------
    # Scheduled
    # ------------------------------------------------------------------
    if status == "scheduled":
        return EventDecision(
            include_in_forecast=settlement_date >= request_date,
            reserve_now=False,
            cash_effect=cash_effect,
            cash_date=settlement_date,
            reason="scheduled cash event counted on settlement date",
        )

    # ------------------------------------------------------------------
    # Settled
    # ------------------------------------------------------------------
    if status == "settled":
        return EventDecision(
            include_in_forecast=settlement_date >= request_date,
            reserve_now=False,
            cash_effect=cash_effect,
            cash_date=settlement_date,
            reason="settled cash event counted on settlement date",
        )

    # Defensive fallback.
    return EventDecision(
        include_in_forecast=False,
        reserve_now=False,
        cash_effect=0.0,
        cash_date=None,
        reason="unreachable fallback",
    )


def prepare_user_events(
    events: pd.DataFrame,
    user_id: str,
    request_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Prepare one user's raw financial events for later forecasting.

    This layer intentionally does NOT yet:
    - infer recurring transactions
    - interpret messages
    - interpret images
    - resolve cross-source conflicts
    - deduplicate lifecycle records

    Those responsibilities will be implemented separately.
    """

    request_date = pd.Timestamp(request_date).normalize()

    user_events = events[events["user_id"] == user_id].copy()

    if user_events.empty:
        return user_events.assign(
            include_in_forecast=pd.Series(dtype=bool),
            reserve_now=pd.Series(dtype=bool),
            cash_effect=pd.Series(dtype=float),
            cash_date=pd.Series(dtype="datetime64[ns]"),
            classification_reason=pd.Series(dtype=str),
        )

    decisions = [
        classify_event(row, request_date)
        for _, row in user_events.iterrows()
    ]

    user_events["include_in_forecast"] = [
        decision.include_in_forecast
        for decision in decisions
    ]

    user_events["reserve_now"] = [
        decision.reserve_now
        for decision in decisions
    ]

    user_events["cash_effect"] = [
        decision.cash_effect
        for decision in decisions
    ]

    user_events["cash_date"] = [
        decision.cash_date
        for decision in decisions
    ]

    user_events["classification_reason"] = [
        decision.reason
        for decision in decisions
    ]

    return user_events


def summarize_event_treatment(
    prepared_events: pd.DataFrame,
) -> dict:
    """Return useful diagnostics for one user's event treatment."""

    if prepared_events.empty:
        return {
            "total_events": 0,
            "forecast_events": 0,
            "reserved_pending_debits": 0,
            "reserved_pending_amount": 0.0,
            "excluded_events": 0,
            "forecast_cash_effect": 0.0,
        }

    reserved_amounts = prepared_events.loc[
        prepared_events["reserve_now"],
        "amount",
    ]

    forecast_effect = prepared_events.loc[
        prepared_events["include_in_forecast"],
        "cash_effect",
    ]

    return {
        "total_events": int(len(prepared_events)),
        "forecast_events": int(
            prepared_events["include_in_forecast"].sum()
        ),
        "reserved_pending_debits": int(
            prepared_events["reserve_now"].sum()
        ),
        "reserved_pending_amount": float(
            reserved_amounts.sum()
        ),
        "excluded_events": int(
            (~prepared_events["include_in_forecast"]).sum()
        ),
        "forecast_cash_effect": float(
            forecast_effect.sum()
        ),
    }