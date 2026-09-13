# FinGuard Usage Report

## Project

FinGuard - Buy or Wait?

## Runtime

The submitted solution is a deterministic Python financial decision engine built using the supplied challenge datasets.

## Model Provider and Model Usage

- External model provider: None
- External model: None
- Model calls during final full-dataset run: 0
- Input tokens: 0
- Output tokens: 0
- Total tokens: 0
- Average tokens per request: 0

## Estimated Cost

- Total model/API cost: $0
- Average cost per request: $0

## Processing Performed

The system performs:

- Financial dataset loading and normalization
- Financial event classification
- Evidence extraction from supplied messages
- Missing transaction amount recovery from supplied image evidence
- Financial reconciliation
- Conservative cash-flow forecasting
- Payment-plan evaluation
- Spending-change evaluation
- Decision generation
- Deterministic challenge-rule validation

## Final Full-Dataset Run

- Requests processed: 250
- Output rows generated: 250
- Challenge-rule validation errors: 0

## Execution

```powershell
python code\run.py
```

The command generates `output.csv` in the repository root.

Validation:

```powershell
python code\quality_check.py
```

Current result:

```text
ROWS: 250
ERROR COUNT: 0
ALL CHALLENGE-RULE VALIDATION CHECKS PASSED
```

## Reproducibility

The submitted runtime uses no external model or API.

The same supplied dataset can be processed deterministically using the documented run command.
