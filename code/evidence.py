from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional
import re

import pandas as pd


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

CURRENCY_AMOUNT_RE = re.compile(
    r"\b(?P<currency>[A-Z]{3})\s+(?P<amount>\d+(?:\.\d+)?)\b"
)

DATE_RE = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")

PERCENT_RE = re.compile(
    r"\b(?P<percent>\d+(?:\.\d+)?)%\b"
)


# ---------------------------------------------------------------------------
# Structured evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceFact:
    """
    A structured fact extracted from a message.

    The parser does not directly modify financial events.
    It produces evidence that later layers can reconcile against events.
    """

    message_id: str
    user_id: str
    request_id: Optional[str]
    related_event_id: Optional[str]

    evidence_type: str
    confidence: str

    amount: Optional[float] = None
    currency: Optional[str] = None
    date: Optional[pd.Timestamp] = None
    percent: Optional[float] = None

    source_type: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_optional_string(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    return text


def extract_amount(text: str) -> tuple[Optional[float], Optional[str]]:
    """
    Extract the first explicit ISO-currency amount.

    Example:
        "salary is EUR 1661"
        -> (1661.0, "EUR")
    """
    match = CURRENCY_AMOUNT_RE.search(text)

    if not match:
        return None, None

    return float(match.group("amount")), match.group("currency")


def extract_date(text: str) -> Optional[pd.Timestamp]:
    match = DATE_RE.search(text)

    if not match:
        return None

    return pd.Timestamp(match.group("date")).normalize()


def extract_percent(text: str) -> Optional[float]:
    match = PERCENT_RE.search(text)

    if not match:
        return None

    return float(match.group("percent"))


def normalized_text(text: str) -> str:
    return " ".join(str(text).lower().split())


# ---------------------------------------------------------------------------
# Message classification
# ---------------------------------------------------------------------------

def classify_message(row: pd.Series) -> list[EvidenceFact]:
    """
    Convert one message into one or more structured financial facts.

    The parser intentionally produces evidence rather than directly changing
    event rows. Conflict resolution will happen in a later reconciliation
    layer using the official challenge precedence rules.
    """

    text = str(row["message_text"])
    lower = normalized_text(text)

    message_id = str(row["message_id"])
    user_id = str(row["user_id"])

    request_id = clean_optional_string(row.get("request_id"))
    related_event_id = clean_optional_string(row.get("related_event_id"))
    source_type = clean_optional_string(row.get("source_type"))

    amount, currency = extract_amount(text)
    date = extract_date(text)
    percent = extract_percent(text)

    facts: list[EvidenceFact] = []

    def add(
        evidence_type: str,
        *,
        confidence: str = "high",
        fact_amount: Optional[float] = amount,
        fact_currency: Optional[str] = currency,
        fact_date: Optional[pd.Timestamp] = date,
        fact_percent: Optional[float] = percent,
        notes: str = "",
    ) -> None:
        facts.append(
            EvidenceFact(
                message_id=message_id,
                user_id=user_id,
                request_id=request_id,
                related_event_id=related_event_id,
                evidence_type=evidence_type,
                confidence=confidence,
                amount=fact_amount,
                currency=fact_currency,
                date=fact_date,
                percent=fact_percent,
                source_type=source_type,
                notes=notes,
            )
        )

    # ------------------------------------------------------------------
    # Salary / employment evidence
    # ------------------------------------------------------------------

    if (
        ("salary" in lower or "gaji" in lower)
        and ("increased" in lower or "naik" in lower)
    ):
        add(
            "salary_increase",
            notes="Confirmed salary increase stated by employer/payroll.",
        )

    elif (
        ("salary" in lower or "gaji" in lower)
        and ("reduced" in lower or "lower" in lower or "lebih rendah" in lower)
    ):
        add(
            "salary_reduced",
            notes="Temporary/reduced salary for affected payroll cycle.",
        )

    elif (
        ("salary" in lower or "gaji" in lower)
        and (
            "first salary" in lower
            or "first salary" in lower
            or "gaji pertama" in lower
        )
    ):
        add(
            "first_salary_confirmed",
            notes="First salary amount and/or first salary date stated.",
        )

    elif (
        ("salary" in lower or "gaji" in lower)
        and (
            "regular salary" in lower
            or "salary" in lower
            or "gaji rutin" in lower
            or "gaji sebesar" in lower
        )
        and (
            "confirmed" in lower
            or "dikonfirmasi" in lower
        )
    ):
        add(
            "salary_confirmed",
            notes="Base/regular salary explicitly confirmed.",
        )

    # Salary resumes with an explicit date.
    if (
        ("salary" in lower or "gaji" in lower)
        and ("resumes" in lower or "resume" in lower)
    ):
        add(
            "salary_resumes",
            notes="Regular salary resumes from the stated date.",
        )

    # Salary date replacement.
    if (
        ("salary" in lower or "gaji" in lower)
        and (
            "replaces the payroll date" in lower
            or "revised date" in lower
            or "tanggal ini menggantikan" in lower
            or "tanggal terbaru" in lower
        )
    ):
        add(
            "salary_date_amendment",
            notes="New payroll date explicitly replaces an earlier date.",
        )

    # Employment ending means future salary should stop.
    if (
        ("employment has ended" in lower)
        or ("contract has ended" in lower)
        or ("hubungan kerja anda telah berakhir" in lower)
        or ("kontrak musiman saat ini telah berakhir" in lower)
        or ("one household employment record has ended" in lower)
        or ("salah satu sumber pendapatan kerja rumah tangga telah berakhir" in lower)
    ):
        add(
            "employment_ended",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Future income from ended employment/contract should not be projected.",
        )

    # Confirmed base salary while commissions remain unapproved.
    if (
        ("commission" in lower or "komisi" in lower)
        and (
            "pending approval" in lower
            or "belum disetujui" in lower
        )
    ):
        add(
            "commission_unconfirmed",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Unapproved commission must not be counted as confirmed income.",
        )

    # Quarterly/seasonal bonus not approved.
    if (
        ("bonus" in lower)
        and (
            "not been approved" in lower
            or "have not been approved" in lower
            or "still subject to" in lower
            or "belum disetujui" in lower
            or "belum" in lower
        )
    ):
        add(
            "bonus_unconfirmed",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Bonus amount/date not confirmed; do not count as income.",
        )

    # One-time arrears adjustment.
    if (
        ("arrears adjustment" in lower)
        or ("penyesuaian tunggakan satu kali" in lower)
    ):
        add(
            "one_time_arrears_adjustment",
            notes="One-time payroll adjustment, separate from regular salary.",
        )

    # ------------------------------------------------------------------
    # Invoice / freelance income evidence
    # ------------------------------------------------------------------

    if (
        ("client approved an invoice payment" in lower)
        or ("klien menyetujui pembayaran faktur" in lower)
        or ("client approved an invoice" in lower)
    ):
        add(
            "invoice_payment_confirmed",
            notes="Client-approved invoice with expected settlement date.",
        )

    # ------------------------------------------------------------------
    # Payout pending
    # ------------------------------------------------------------------

    if (
        ("payout is still pending" in lower)
        or ("pembayaran berikutnya" in lower and "masih tertunda" in lower)
    ):
        add(
            "payout_pending",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Payout remains uncertain/unavailable until completed.",
        )

    # ------------------------------------------------------------------
    # Refund evidence
    # ------------------------------------------------------------------

    if (
        ("refund has been initiated" in lower)
        or ("refund is still processing" in lower)
        or ("pengembalian dana sudah diproses" in lower)
        or ("pengembalian dana" in lower and "belum masuk" in lower)
    ):
        add(
            "refund_pending",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Refund is not yet credited; do not count as available cash.",
        )

    # ------------------------------------------------------------------
    # Foreign-currency transaction/refund awaiting final settlement amount
    # ------------------------------------------------------------------

    if (
        ("foreign-currency" in lower)
        or ("mata uang asing" in lower)
    ) and (
        ("final home-currency amount" in lower)
        or ("final home-currency credit" in lower)
        or ("jumlah akhir dalam mata uang utama" in lower)
    ):
        add(
            "foreign_currency_pending_final_amount",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Final home-currency amount depends on settlement-date FX rate.",
        )

    # ------------------------------------------------------------------
    # Investment valuation is explicitly non-cash
    # ------------------------------------------------------------------

    if (
        ("portfolio" in lower or "investment" in lower or "investasi" in lower)
        and (
            "market value" in lower
            or "displayed value" in lower
            or "nilai investasi" in lower
            or "nilai yang ditampilkan" in lower
        )
        and (
            "no units have been sold" in lower
            or "has not been sold" in lower
            or "belum dijual" in lower
            or "no cash transaction" in lower
            or "tidak ada transaksi tunai" in lower
        )
    ):
        add(
            "investment_valuation_non_cash",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Displayed investment value is not available cash.",
        )

    # ------------------------------------------------------------------
    # Investment sale settled into cash
    # ------------------------------------------------------------------

    if (
        ("investment sale" in lower)
        and (
            "settled in the cash account" in lower
            or "reached the cash account" in lower
        )
    ) or (
        ("penjualan investasi" in lower)
        and ("sudah masuk ke rekening tunai" in lower)
    ):
        add(
            "investment_sale_settled",
            notes="Investment-sale proceeds explicitly reached cash account.",
        )

    # ------------------------------------------------------------------
    # Prize / windfall evidence
    # ------------------------------------------------------------------

    if (
        ("prize claim" in lower)
        or ("prize proceeds" in lower)
        or ("klaim hadiah" in lower)
        or ("hasil penjualan" in lower and "investasi" in lower)
    ):
        if (
            "still in payment processing" in lower
            or "has not been credited" in lower
            or "belum masuk" in lower
            or "masih dalam proses pembayaran" in lower
        ):
            add(
                "prize_payment_pending",
                fact_amount=None,
                fact_currency=None,
                fact_date=None,
                fact_percent=None,
                notes="Prize proceeds are not yet credited.",
            )

        elif (
            "reached your account" in lower
            or "reached the account" in lower
            or "claim is now closed" in lower
            or "sudah masuk ke rekening" in lower
        ):
            add(
                "prize_payment_received",
                notes="Prize proceeds explicitly stated as received.",
            )

    # Suspicious release-charge language:
    # the message asks the user to pay first, but it does NOT establish
    # confirmed income. We preserve it as evidence rather than treating
    # it as income or as a legitimate required expense.
    if (
        ("pay the release charge" in lower)
        or ("pay the processing charge" in lower)
        or ("bayar biaya pencairan" in lower)
        or ("bayar biaya pemrosesan" in lower)
    ):
        add(
            "unverified_prize_fee_request",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Message requests an upfront fee; does not establish confirmed income.",
        )

    # ------------------------------------------------------------------
    # Work reimbursement
    # ------------------------------------------------------------------

    if (
        ("reimbursement" in lower and "work expense" in lower)
        or ("penggantian atas biaya kerja" in lower)
    ):
        add(
            "work_expense_reimbursement",
            notes="Employer credit is reimbursement, not recurring salary.",
        )

    # ------------------------------------------------------------------
    # Rent amendment
    # ------------------------------------------------------------------

    if (
        ("renewed lease" in lower and "increases monthly rent" in lower)
        or ("perpanjangan sewa" in lower and "menaikkan" in lower)
    ):
        add(
            "rent_increase",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=percent,
            notes="Rent increases by stated percentage from next rent payment.",
        )

    # ------------------------------------------------------------------
    # Internal transfer
    # ------------------------------------------------------------------

    if (
        ("matching debit and credit" in lower)
        and (
            "between your two accounts" in lower
            or "same account holder" in lower
        )
    ) or (
        ("debit dan kredit" in lower)
        and ("transfer antara dua rekening" in lower)
    ):
        add(
            "internal_account_transfer",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Matching debit/credit are internal transfers and net to zero.",
        )

    # ------------------------------------------------------------------
    # Card dispute / reversal pending
    # ------------------------------------------------------------------

    if (
        ("extra card charge" in lower)
        and (
            "reversal has not been posted" in lower
            or "dispute is open" in lower
        )
    ) or (
        ("tagihan kartu tambahan" in lower)
        and ("dana pembalikannya belum tercatat" in lower)
    ):
        add(
            "card_charge_dispute_open",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Charge remains unresolved; reversal has not been posted.",
        )

    # ------------------------------------------------------------------
    # Failed debit with retry
    # ------------------------------------------------------------------

    if (
        ("previous debit attempt failed" in lower)
        or ("debit sebelumnya gagal" in lower)
    ) and (
        "another debit will be attempted" in lower
        or "another debit may be attempted" in lower
        or "akan dicoba" in lower
    ):
        add(
            "failed_debit_retry_expected",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Failed debit does not erase the outstanding obligation; another attempt is expected.",
        )

    # ------------------------------------------------------------------
    # Separate card minimum payments
    # ------------------------------------------------------------------

    if (
        ("minimum payments due on two separate card accounts" in lower)
        or ("minimums belong to separate accounts" in lower)
        or ("minimum payments" in lower and "separate accounts" in lower)
    ):
        add(
            "separate_card_minimums",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="One card payment cannot satisfy the other card's minimum.",
        )

    # ------------------------------------------------------------------
    # Receipt confirms final amount exists, even if message doesn't state it
    # ------------------------------------------------------------------

    if (
        ("receipt has the final" in lower)
        or ("receipt contains the final" in lower)
        or ("receipt has the final amount" in lower)
        or ("receipt has the final INR amount" in lower)
    ):
        add(
            "receipt_final_amount_available",
            fact_amount=None,
            fact_currency=None,
            fact_date=date,
            fact_percent=None,
            notes="Receipt indicates a final transaction amount is available in the linked evidence.",
        )

    # ------------------------------------------------------------------
    # Temporary reduced payroll
    # ------------------------------------------------------------------

    if (
        ("temporary monthly pay" in lower or "temporary pay" in lower)
        and (
            "reduced amount continues" in lower
            or "reduced amount" in lower
            or "pay is reduced" in lower
        )
    ):
        add(
            "salary_reduced",
            notes=(
                "Employer states that temporary reduced monthly pay "
                "continues for the affected payroll cycle."
            ),
        )

    # ------------------------------------------------------------------
    # Investment / portfolio valuation change without cash movement
    # ------------------------------------------------------------------

    if (
        (
            "portfolio" in lower
            or "investment" in lower
            or "market value" in lower
            or "displayed value" in lower
        )
        and (
            "market value" in lower
            or "displayed value" in lower
        )
        and (
            "increased" in lower
            or "decreased" in lower
            or "move with market prices" in lower
            or "market prices" in lower
        )
    ):
        add(
            "investment_valuation_non_cash",
            fact_amount=None,
            fact_currency=None,
            fact_date=date,
            fact_percent=None,
            notes=(
                "Portfolio valuation changed; this is non-cash market-value "
                "evidence and must not be treated as available cash."
            ),
        )

    # ------------------------------------------------------------------
    # Foreign-currency transaction awaiting final settlement amount
    # ------------------------------------------------------------------

    if (
        (
            "charged in a foreign currency" in lower
            or "foreign currency" in lower
        )
        and (
            "final home-currency amount" in lower
            or "home-currency amount" in lower
        )
        and (
            "when the transaction settles" in lower
            or "when it settles" in lower
            or "rate applied when it settles" in lower
        )
    ):
        add(
            "foreign_currency_pending_final_amount",
            fact_amount=None,
            fact_currency=None,
            fact_date=date,
            fact_percent=None,
            notes=(
                "Foreign-currency transaction is awaiting settlement; "
                "the final home-currency amount is not yet known."
            ),
        )

    # ------------------------------------------------------------------
    # If nothing matched, retain a generic evidence record.
    # ------------------------------------------------------------------

    if not facts:
        add(
            "unclassified_financial_message",
            confidence="low",
            fact_amount=None,
            fact_currency=None,
            fact_date=None,
            fact_percent=None,
            notes="Message preserved for review; no deterministic pattern matched.",
        )

    return facts


# ---------------------------------------------------------------------------
# Batch parsing
# ---------------------------------------------------------------------------

def parse_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all messages into a normalized evidence table.

    One message may produce multiple facts.
    """

    facts: list[dict[str, Any]] = []

    for _, row in messages.iterrows():
        message_facts = classify_message(row)

        for fact in message_facts:
            data = asdict(fact)

            if data["date"] is not None:
                data["date"] = pd.Timestamp(data["date"]).strftime("%Y-%m-%d")

            facts.append(data)

    return pd.DataFrame(
        facts,
        columns=[
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
        ],
    )


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def summarize_message_evidence(
    parsed: pd.DataFrame,
) -> dict[str, Any]:
    if parsed.empty:
        return {
            "messages_parsed": 0,
            "facts_extracted": 0,
            "unique_evidence_types": 0,
            "unclassified_facts": 0,
        }

    return {
        "messages_parsed": int(parsed["message_id"].nunique()),
        "facts_extracted": int(len(parsed)),
        "unique_evidence_types": int(
            parsed["evidence_type"].nunique()
        ),
        "unclassified_facts": int(
            (parsed["evidence_type"] == "unclassified_financial_message").sum()
        ),
    }


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()
    evidence = parse_messages(data.messages)

    print("=" * 70)
    print("FIN GUARD MESSAGE EVIDENCE DIAGNOSTICS")
    print("=" * 70)

    print(summarize_message_evidence(evidence))

    print("\n--- Evidence types ---")
    print(
        evidence["evidence_type"]
        .value_counts()
        .to_string()
    )

    print("\n--- Facts with amounts/dates ---")
    useful = evidence[
        evidence["amount"].notna()
        | evidence["date"].notna()
        | evidence["percent"].notna()
    ]

    print(
        useful[
            [
                "message_id",
                "user_id",
                "evidence_type",
                "amount",
                "currency",
                "date",
                "percent",
            ]
        ].to_string(index=False)
    )