from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from data_loader import load_data
from evidence import parse_messages
from reconciliation import reconcile_all
from decision_engine import decide_all


def main() -> None:
    data = load_data()

    messages = getattr(data, "messages", None)
    parsed_evidence = parse_messages(messages) if messages is not None and not messages.empty else None

    states = reconcile_all(data, parsed_evidence=parsed_evidence)
    output = decide_all(states, data)

    output_path = ROOT / "output.csv"
    output.to_csv(output_path, index=False)

    print("=" * 78)
    print("FIN GUARD - BUY OR WAIT DECISION ENGINE")
    print("=" * 78)
    print(f"Requests: {len(output)}")
    print(f"Output:   {output_path}")
    print()
    print("Status distribution:")
    print(output["affordability_status"].value_counts().sort_index().to_string())
    print()
    print("Method distribution:")
    print(output["recommended_payment_method"].value_counts().sort_index().to_string())
    print()
    print(output.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
