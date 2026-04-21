# Document Triage Agent — Simulator Plan

Adapted from the pattern used in `payment-agent/simulator/`. The goal is a synthetic
end-to-end runner that submits documents, automatically resolves HITL workitems as the
"human reviewer", and feeds RLHF outcome labels back to AMP — all without a real human
in the loop.

---

## How the Payment Agent Simulator Works (Reference)

The payment-agent simulator has three components:

1. **`invoice_generator.py`** — Generates a deterministic CSV of fake invoices.
   Each row contains input features *and* a ground-truth `expected_human_decision` label.

2. **`run_e2e.py`** — Orchestrator that acts as a synthetic client:
   - Mints auth tokens via mock OIDC
   - POSTs payment requests to the gateway
   - Detects when a request triggers HITL (gets a step-up/pending response)
   - Resolves the HITL workitem using the CSV ground-truth label
   - Submits RLHF outcomes to AMP via `POST /api/rlhf/outcome`
   - Triggers stage ticks to advance the RLHF pipeline

3. **The gateway** — Passive participant. It doesn't know it's talking to a simulator;
   it just processes requests normally.

**Key insight:** The simulator has *foreknowledge* (ground truth in the CSV). It uses
that to play the role of the human reviewer, closing the RLHF feedback loop with no
real human involvement.

---

## Document Triage Agent — Current State

`agent.py` already has the full HITL flow wired:

- `ACCEPT` / `REQUEST_MORE_INFO` → auto-finishes the instance
- `HUMAN_REVIEW` → calls `POST /api/hitl/request`, sets state to `wait`,
  then polls `GET /api/hitl/get-decision` for up to 3 minutes

The simulator needs to slot in as the entity that resolves those HITL decisions.

---

## What to Build

### File Layout

```
document-triage-agent/
├── agent.py                          # existing — no changes needed
├── simulator/
│   ├── doc_generator.py              # Step 1: generate synthetic docs CSV
│   ├── run_e2e.py                    # Step 2: submit docs + resolve HITL
│   └── domain_profile.json          # Step 3: config (categories, thresholds, templates)
└── SIMULATOR_PLAN.md                 # this file
```

---

## Step 1 — `simulator/doc_generator.py`

Generates a CSV where each row is a synthetic document with a ground-truth label.

### Output Schema (CSV columns)

| Column | Example | Purpose |
|---|---|---|
| `doc_id` | `doc-001` | Unique ID passed through to AMP instance |
| `doc_text` | `"INVOICE #1042 from..."` | The document body sent to `/submit` |
| `doc_type` | `invoice` | Ground-truth document category |
| `expected_decision` | `ACCEPT` | What the agent *should* return |
| `expected_human_resolution` | `approve` | What the simulator says as the human reviewer (only relevant when `expected_decision == HUMAN_REVIEW`) |
| `scenario` | `complete_invoice` | Human-readable test scenario name |

### Document Categories to Cover

```python
SCENARIOS = {
    # Auto-finish cases (no HITL)
    "complete_invoice":        expected_decision="ACCEPT"
    "complete_contract":       expected_decision="ACCEPT"
    "incomplete_form":         expected_decision="REQUEST_MORE_INFO"
    "missing_signature":       expected_decision="REQUEST_MORE_INFO"

    # HITL cases
    "gibberish":               expected_decision="HUMAN_REVIEW", resolution="reject"
    "ambiguous_legal":         expected_decision="HUMAN_REVIEW", resolution="approve"
    "suspicious_amount":       expected_decision="HUMAN_REVIEW", resolution="reject"
    "foreign_language":        expected_decision="HUMAN_REVIEW", resolution="approve"
}
```

### Implementation Notes

- Seed the RNG with a fixed value so the same CSV is reproduced on every run
- Vary phrasing within each category (different names, dates, amounts) to avoid
  the LLM pattern-matching on template artifacts
- Include edge cases: empty documents, very long documents, mixed-language text

---

## Step 2 — `simulator/run_e2e.py`

The orchestrator. Mirrors the structure of `payment-agent/simulator/run_e2e.py`.

### Flow Per Row

```
load CSV row
    │
    ▼
POST /submit  ──────────────────────────────────────────────────┐
    │                                                            │
    ├─ response has instance_id                                  │
    │                                                            │
    ▼                                                            │
if expected_decision == HUMAN_REVIEW:                           │
    poll GET /api/workitems?agent_name=document-triage          │
        until workitem appears for this instance_id             │
    PUT /api/workitems/{id}/status                              │
        body: { "status": "complete",                           │
                "resolution": row["expected_human_resolution"] }│
    POST /api/rlhf/outcome                                      │
        body: { "instance_id": ...,                             │
                "workitem_id": ...,                             │
                "human_label": row["expected_decision"],        │
                "agent_prediction": response["decision"] }      │
else:                                                           │
    assert response["decision"] == row["expected_decision"]     │
    record pass/fail                                            │
    └────────────────────────────────────────────────────────────┘
```

### Key Functions to Implement

```python
def load_rows(csv_path: str) -> list[dict]:
    """Load and validate the generated CSV."""

def submit_document(agent_url: str, api_key: str, row: dict) -> dict:
    """POST to /submit, return response JSON."""

def poll_for_workitem(amp_url: str, api_key: str,
                      instance_id: str, timeout: int = 60) -> str | None:
    """Poll GET /api/workitems until a workitem appears for instance_id.
    Returns workitem_id or None on timeout."""

def complete_workitem(amp_url: str, api_key: str,
                      workitem_id: str, resolution: str) -> None:
    """PUT /api/workitems/{id}/status with the ground-truth resolution."""

def submit_rlhf_outcome(amp_url: str, api_key: str,
                        instance_id: str, workitem_id: str,
                        human_label: str, agent_prediction: str) -> None:
    """POST /api/rlhf/outcome to feed label back to AMP."""

def run(csv_path: str, agent_url: str, amp_url: str, api_key: str) -> None:
    """Main loop over all rows. Print pass/fail per row, summary at end."""
```

### CLI Interface

```bash
# Generate docs first
python simulator/doc_generator.py --output simulator/docs.csv --count 50

# Run the simulator
python simulator/run_e2e.py \
  --csv      simulator/docs.csv \
  --agent    http://localhost:6000 \
  --amp      https://your-amp-backend.com \
  --api-key  $AMP_API_KEY
```

### Output Format (JSON-lines, one per processed row)

```json
{"doc_id": "doc-001", "scenario": "complete_invoice", "status": "pass", "agent_decision": "ACCEPT"}
{"doc_id": "doc-007", "scenario": "gibberish", "status": "pass", "hitl_resolved": true, "resolution": "reject"}
{"doc_id": "doc-012", "scenario": "ambiguous_legal", "status": "fail", "expected": "HUMAN_REVIEW", "got": "ACCEPT"}
```

---

## Step 3 — `simulator/domain_profile.json`

Config file that drives both the generator and the runner. Mirrors
`payment-agent/simulator/domain_profile.example.json`.

```json
{
  "agent_name": "document-triage",
  "hitl_resolution_delay_ms": 0,
  "scenarios": [
    {
      "name": "complete_invoice",
      "weight": 30,
      "expected_decision": "ACCEPT",
      "template": "invoice_complete"
    },
    {
      "name": "gibberish",
      "weight": 10,
      "expected_decision": "HUMAN_REVIEW",
      "expected_human_resolution": "reject",
      "template": "gibberish"
    }
  ],
  "templates": {
    "invoice_complete": "INVOICE #{invoice_id}\nFrom: {vendor}\nTo: {client}\nDate: {date}\nTotal: ${amount}\nPayment: Bank transfer",
    "gibberish": "asdf qwer zxcv {random_words}"
  }
}
```

---

## AMP Endpoints the Simulator Will Call

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/submit` | Submit document to the triage agent |
| `GET` | `/api/workitems` | Poll for HITL workitems by agent/instance |
| `PUT` | `/api/workitems/{id}/status` | Resolve workitem as human reviewer |
| `POST` | `/api/rlhf/outcome` | Feed human label back to AMP |

> **Note:** The agent already calls `POST /api/hitl/request` and polls
> `GET /api/hitl/get-decision` internally. The simulator does *not* call these —
> it calls the workitem endpoints directly, the same way the payment-agent simulator
> does. Verify the exact workitem endpoint paths against your AMP backend before
> implementing.

---

## Open Questions Before Starting

1. **Workitem endpoint shape** — The payment agent polls `GET /api/workitems` and
   resolves via `PUT /api/workitems/{id}/status`. Confirm these exact paths exist in
   your AMP instance (the triage agent uses `POST /api/hitl/request` and
   `GET /api/hitl/get-decision` which may be a different surface).

2. **RLHF outcome endpoint** — Does `POST /api/rlhf/outcome` exist for your AMP setup,
   or does it need to be added?

3. **Agent concurrency** — The `/submit` endpoint blocks while polling HITL (up to 3
   min). The simulator should submit documents sequentially, or the agent needs to be
   made async before running parallel loads.
