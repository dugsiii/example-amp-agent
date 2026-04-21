# Integrating an Agent with AMP's RLHF Pipeline

This guide explains how to integrate any agent with AMP's human-in-the-loop (HITL) and RLHF training pipeline. It uses the document triage agent as a concrete example throughout, but the pattern applies to any agent that makes classification or approval decisions.

---

## Table of Contents

1. [The Core Pattern](#1-the-core-pattern)
2. [What You Need to Build](#2-what-you-need-to-build)
3. [Step 1: Instrument Your Agent for HITL](#step-1-instrument-your-agent-for-hitl)
4. [Step 2: Define Your RLHF Policy](#step-2-define-your-rlhf-policy)
5. [Step 3: Build a Domain Profile](#step-3-build-a-domain-profile)
6. [Step 4: Write the Document Generator](#step-4-write-the-document-generator)
7. [Step 5: Write the Simulator Loop](#step-5-write-the-simulator-loop)
8. [Step 6: Run the Pipeline](#step-6-run-the-pipeline)
9. [Understanding Progressive Autonomy](#understanding-progressive-autonomy)
10. [Reference: Environment & Configuration](#reference-environment--configuration)
11. [Troubleshooting](#troubleshooting)

---

## 1. The Core Pattern

AMP's RLHF pipeline has a specific learning loop:

```
Agent makes a decision
  → sends HITL request to AMP (action fields, signal fields, criteria results)
  → human reviews and approves/rejects
  → outcome is confirmed
  → EXTRACT job pulls confirmed outcomes into a training dataset
  → TRAIN job trains a model on that dataset
  → model is evaluated, then promoted to assist or replace human review
```

The simulator accelerates this loop by replacing the human reviewer with a synthetic one. Instead of waiting weeks for real human feedback, you can generate hundreds of labeled examples in minutes.

**The key insight:** the simulator doesn't fake the data — it runs the real agent, the real HITL workflow, and the real RLHF pipeline. The only thing that's synthetic is the input documents and the human label. Everything in AMP (instances, workitems, outcomes, training) happens for real.

---

## 2. What You Need to Build

There are four components:

| Component | Purpose | Specific to your domain |
|---|---|---|
| **Agent** | Makes decisions, sends HITL requests to AMP | Yes — your business logic |
| **RLHF Policy** | Tells AMP what features to collect and what criteria to evaluate | Yes — your approval rules |
| **Domain Profile** | Describes the distribution of real-world inputs | Yes — your document/request types |
| **Simulator** | Generates synthetic inputs and auto-resolves HITL workitems | Mostly reusable — thin wrapper |

The simulator code in this repo (`doc_generator.py` and `run_e2e.py`) is a good template. The parts that change between domains are the domain profile and the feature/criteria definitions in the policy.

---

## Step 1: Instrument Your Agent for HITL

Your agent needs to do three things for the pipeline to work:

### 1a. Create an AMP instance

When a request arrives, tell AMP you're starting work:

```python
init = requests.post(f"{AMP_URL}/api/agent/init", headers=headers,
                     json={"agent_name": AGENT, "prompt": input_text, "auto_start": True})
iid = init.json()["instance_id"]
```

### 1b. Send a HITL request with structured features

This is the most important step. The features you send here become the training data for your model. Include every signal that should influence the approve/reject decision:

```python
rlhf_payload = {
    "policy_values_source": "external",
    "action_fields": {
        # Structured facts about this specific request
        # These become the model's input features — choose carefully
        "field_one": value_one,
        "field_two": value_two,
    },
    "signal_fields": {
        # Boolean flags derived from the action
        "some_flag": bool_value,
        "another_flag": bool_value,
    },
    "criteria": [
        # Pre-evaluated rule results (hard constraints the model should respect)
        {"criterion_id": "c_some_rule", "result": bool_result},
    ],
}

hitl_body = {
    "caller_id": iid,
    "instance_id": iid,
    "org_id": org_id,
    "agent_name": AGENT,
    "hitl": {"enable": True, "when": "always", "who": "initiator",
             "what": "approval", "where": "amp"},
    "rlhf": rlhf_payload,
}
requests.post(f"{AMP_URL}/api/hitl/request", json=hitl_body, headers=headers)
```

**Choosing your features:**
- Include anything a human reviewer would look at when deciding to approve or reject
- `action_fields` should be factual/measurable (counts, types, categories, amounts)
- `signal_fields` should be boolean flags derived from those facts
- These must match the `action_fields_allowlist` and `signal_keys_allowlist` in your policy (Step 2)

**Document triage example:**

```python
"action_fields": {
    "doc_type":             "invoice",    # what kind of document
    "missing_fields_count": 0,            # how many required fields are missing
    "missing_fields":       "",           # which fields
    "submitted_by":         owner,        # who submitted it
},
"signal_fields": {
    "doc_type_unrecognized":   False,  # agent couldn't classify it
    "has_missing_fields":      False,  # has gaps
    "agent_flagged_ambiguous": True,   # agent wasn't confident
},
```

### 1c. Poll for the human's decision, confirm the outcome, and finish

```python
# Set state to waiting
requests.post(f"{AMP_URL}/api/agent/setState",
              json={"agent_name": AGENT, "instance_id": iid, "state": "wait"})

for _ in range(36):      # poll up to 3 minutes
    time.sleep(5)
    decision = requests.get(f"{AMP_URL}/api/hitl/get-decision",
                            params={"caller_id": iid}, headers=headers).json()
    if decision.get("status") == "complete":
        resolution = decision["resolution"]   # "approve" or "reject"

        # Confirm the RLHF outcome immediately so it's eligible for training
        # without waiting for the default 48-hour settlement window.
        # Non-fatal: if this fails, the settlement sweep will confirm it eventually.
        try:
            requests.post(f"{AMP_URL}/api/rlhf/outcome/feedback",
                          headers={**headers, "Content-Type": "application/json"},
                          json={"org_id": org_id,
                                "instance_id": iid,
                                "outcome": "succeeded",
                                "outcome_source": "agent_callback"},
                          timeout=10)
        except Exception:
            pass  # non-fatal

        final_state = "abort" if resolution == "reject" else "finished"
        requests.post(f"{AMP_URL}/api/agent/setState",
                      json={"agent_name": AGENT, "instance_id": iid,
                            "state": final_state})
        break
```

**Why the agent owns this confirmation, not the simulator:**

RLHF outcomes start as `pending` and only become visible to the EXTRACT training job once they are `confirmed`. Confirmation can happen two ways:

1. **Automatically** — after `outcome_window_hours` expires (default 48h). Designed for production agents that submit real downstream feedback (e.g., "the payment succeeded/failed").
2. **Explicitly** — by calling `POST /api/rlhf/outcome/feedback` with `outcome=succeeded`. This is what the agent does here.

Putting the confirmation call in the agent means it works correctly in both production use (real human reviewer) and simulator use, without the simulator needing to know about it.

### 1d. Force all decisions through HITL during data collection

During the simulator phase you want every request to produce a labeled training example. Override your agent's own routing logic to always request HITL:

```python
# Force all decisions through HITL for simulator training data collection.
# Remove this once you have enough data and are ready for autonomous routing.
result["decision"] = "HUMAN_REVIEW"
```

This ensures 100% of simulator runs produce labeled data. Remove it once you've collected enough examples and trained a model.

---

## Step 2: Define Your RLHF Policy

The policy tells AMP how to evaluate decisions and what to use as training features. Create it in AMP → Policies → New Policy.

### Minimal policy structure

```json
{
  "policy_id": "your-agent-policy-v1",
  "policy_version": "1.0",
  "rlhf_enabled": true,
  "feature_schema_version": "your_domain_v1",

  "action_governance": {
    "action_fields_allowlist": [
      "field_one",
      "field_two"
    ],
    "signal_keys_allowlist": [
      "some_flag",
      "another_flag"
    ],
    "criteria": [
      {
        "criterion_id": "c_your_hard_rule",
        "text": "Human-readable description of the rule",
        "type": "hard",
        "evaluator": {
          "strategy": "compute",
          "inputs": { "flag": "{{action.some_flag}}" },
          "formula": "!flag"
        }
      }
    ]
  },

  "params": {
    "initial_train_min_human_rows": 100,
    "retrain_min_human_rows": 50,
    "outcome_feedback": {
      "outcome_window_hours": 0
    }
  }
}
```

### Criteria types

**Hard criteria** — must pass for an approve recommendation. If any hard criterion fails, the model will recommend reject regardless of score. Use these for non-negotiable rules:

```json
{
  "criterion_id": "c_no_missing_required_fields",
  "text": "All required fields are present",
  "type": "hard",
  "evaluator": {
    "strategy": "compute",
    "inputs": { "has_missing_fields": "{{action.has_missing_fields}}" },
    "formula": "!has_missing_fields"
  }
}
```

**Soft criteria** — influence the recommendation but don't block it. Use these for signals that are informative but not absolute:

```json
{
  "criterion_id": "c_amount_reasonable",
  "text": "Requested amount is within normal range",
  "type": "soft",
  "evaluator": {
    "strategy": "compute",
    "inputs": { "flagged": "{{action.amount_flagged}}" },
    "formula": "!flagged"
  }
}
```

### Key parameters

| Parameter | Recommended for simulator | Production |
|---|---|---|
| `initial_train_min_human_rows` | 100 | 200–500 depending on domain complexity |
| `retrain_min_human_rows` | 50 | 100–200 |
| `outcome_window_hours` | **0** (confirm immediately) | 24–48 (allow time for real-world reversal feedback) |

`outcome_window_hours: 0` is critical for the simulator. See [Troubleshooting](#troubleshooting) for why.

Once the policy is created, set it as **Active** in AMP before running `--schema-source policy` in the generator.

---

## Step 3: Build a Domain Profile

The domain profile is a JSON file that describes the realistic distribution of inputs your agent sees. It drives the document generator.

### Structure

```json
{
  "agent_name": "your-agent",
  "hitl_resolution_delay_ms": 0,
  "scenarios": [
    {
      "name": "a_clear_approve",
      "weight": 30,
      "expected_human_resolution": "approve",
      "template": "template_name_a"
    },
    {
      "name": "a_clear_reject",
      "weight": 25,
      "expected_human_resolution": "reject",
      "template": "template_name_b"
    }
  ],
  "templates": {
    "template_name_a": "Template text with {placeholder} variables...",
    "template_name_b": "Another template..."
  }
}
```

### Designing scenarios

Each scenario represents a class of real-world inputs. Think about:

- **What types of inputs does your agent receive?** (e.g., for expense approval: receipts, travel requests, equipment purchases)
- **Which types should be approved, which rejected?** — assign `expected_human_resolution` based on your business rules
- **How frequently does each type appear in production?** — use `weight` to match that distribution

**Document triage scenarios as an example:**

| Scenario | Weight | Resolution | Why |
|---|---|---|---|
| `complete_invoice` | 25 | approve | All fields present, type recognized |
| `complete_contract` | 15 | approve | Signed, complete |
| `incomplete_form` | 15 | reject | Missing supervisor signature, cost center |
| `missing_signature` | 15 | reject | Contract unsigned |
| `gibberish` | 10 | reject | Unrecognizable content |
| `ambiguous_legal` | 8 | approve | Boilerplate legal notice — valid but unusual |
| `suspicious_invoice` | 7 | reject | Unusual amount, no receipt, 24h wire transfer demand |
| `foreign_language` | 5 | approve | Valid document, non-English |

### Approve/reject ratio

The weight distribution determines your training data class balance. The generator prints the expected distribution before generating:

```
expected distribution: approve=53% reject=47%  (from 8 scenarios)
```

Aim for a ratio that matches your production data. A 60/40 or 50/50 split is generally healthier for model training than a highly skewed ratio.

### Templates

Templates are strings with `{placeholder}` variables that get filled with random values from lists you define in the profile:

```json
"templates": {
  "expense_request": "EXPENSE REQUEST\nEmployee: {name}\nAmount: ${amount}\nPurpose: {purpose}\nReceipt: {has_receipt}\nApproval: {approver}"
}
```

Add any domain-specific lists you need (`names`, `departments`, `amount_ranges`, etc.).

---

## Step 4: Write the Document Generator

The generator reads the domain profile, generates synthetic inputs, and writes a CSV for the simulator. Use `doc_generator.py` as a starting template.

### What to keep (domain-agnostic)

- The `.env` loading block at the top
- The `PolicyRuntimeClient` class — calls `GET /api/rlhf/policy` and `POST /api/rlhf/policy/evaluate` exactly as-is
- The `DocumentRecord` dataclass — same columns, same CSV schema
- The CLI argument parsing structure — all flags work the same way

### What to change (domain-specific)

**1. `_build_action_params`** — translate your scenario/template into the feature dictionary the policy evaluator receives. This must match the `action_fields` and `signal_fields` your agent sends in Step 1:

```python
def _build_action_params(self, scenario: str, doc_type: str, ...) -> Dict[str, Any]:
    return {
        "doc_type": doc_type,
        "some_flag": computed_bool,
        "another_flag": computed_bool,
        # match exactly what your agent sends in the HITL request
    }
```

**2. `_build_static_criteria`** — hardcoded criteria evaluation for `--schema-source static` mode (no AMP connection required):

```python
def _build_static_criteria(self, doc_type: str, ...) -> List[Dict]:
    return [
        {"criterion_id": "c_your_rule", "result": bool_result, "type": "hard"},
    ]
```

**3. Template filling logic** — replace the document-triage-specific template fields (`{invoice_id}`, `{vendor}`, etc.) with the fields relevant to your domain.

### Generator flags (all reusable as-is)

| Flag | Description |
|---|---|
| `--count N` | Number of inputs to generate |
| `--output path` | Output CSV path |
| `--seed N` | Random seed for reproducibility |
| `--sim-run-id ID` | Label for grouping this batch in AMP |
| `--schema-source static\|policy` | Static: uses hardcoded criteria. Policy: fetches live criteria from AMP |
| `--profile path` | Path to your domain profile JSON |

---

## Step 5: Write the Simulator Loop

The simulator submits inputs to the agent, resolves HITL workitems, and confirms outcomes. Use `run_e2e.py` as a starting template.

### What to keep (domain-agnostic)

Almost all of `run_e2e.py` is reusable without changes:

- `.env` loading block
- `_submit_document` — posts to `/submit`, handles the blocking HITL poll
- `_poll_for_workitem` — polls `GET /api/workitems` until the workitem appears
- `_complete_workitem` — `PUT /api/workitems/{id}/status` with the resolution
- `_submit_rlhf_outcome` — posts to `/api/rlhf/outcome`
- `_bootstrap_rlhf` — calls `/api/rlhf/bootstrap` before the sim loop
- `_process_hitl_row` — orchestrates the above steps for one row
- `run` and `main` — the main loop and CLI

### What to change

**1. The `/submit` payload field name** — your agent may expect a different field name. In the document triage agent it's `document`:

```python
payload = {
    'doc_id': row['doc_id'],
    'document': row['doc_text'],  # ← change this to match your agent's input field
}
```

**2. The agent endpoint** — if your agent uses a different route than `/submit`, update `_submit_document`.

**3. The `expected_human_resolution` column name** — if you name your label column differently in the generator, update the reference in `_process_hitl_row`.

That's typically it. The HITL workitem polling, RLHF outcome submission, and outcome feedback confirmation are all AMP API calls that don't change between domains.

### Outcome confirmation: agent's responsibility, not the simulator's

The simulator only needs to submit the outcome record. Confirmation is handled by the agent itself (see Step 1c). This keeps the simulator thin and means confirmation works correctly in production too, without the simulator needing to know about it.

If you write your own agent and do not add the confirmation call to it, outcomes will stay `pending` for 48 hours (the default settlement window) before the EXTRACT job can see them. Set `outcome_window_hours: 0` in your policy as a fallback if you want the settlement sweep to handle it immediately instead.

---

## Step 6: Run the Pipeline

```bash
# 1. Start your agent
python agent.py

# 2. Generate inputs
python simulator/doc_generator.py --count 100 --seed 1 --output simulator/docs.csv

# 3. Run the simulator
python simulator/run_e2e.py --csv simulator/docs.csv

# 4. Repeat with new seeds until AMP triggers training
python simulator/doc_generator.py --count 100 --seed 2 --output simulator/docs.csv
python simulator/run_e2e.py --csv simulator/docs.csv
```

Each round produces ~100 labeled examples. With `initial_train_min_human_rows: 100`, one round is enough to trigger the first training run. Watch AMP → Agents → your agent → RLHF for the STAGE_TICK → EXTRACT → TRAIN → EVAL sequence.

### Expected simulator output (per document)

```json
{"event": "rlhf_bootstrap", "ok": true, "enabled": true, "stage_tick_enqueued": true}
{"doc_id": "abc-001", "scenario": "complete_invoice", "status": "pass",
 "hitl_resolved": true, "resolution": "approve", "rlhf_submitted": true}
...
{"event": "sim_end", "total": 100, "passed": 100, "errors": 0}
```

`rlhf_submitted: true` on every row means the outcome was both submitted and confirmed. These rows are immediately eligible for the next EXTRACT run.

---

## Understanding Progressive Autonomy

AMP's pipeline moves through three stages. You start at Stage 0 and earn progression through demonstrated performance.

### Stage 0 — Full HITL

Every decision goes to a human. The agent's job is to collect and structure the right information for the reviewer, not to decide autonomously. The simulator populates Stage 0 training data.

**What you're building here:** a labeled dataset of human decisions with structured features.

### Stage 1 — Model Recommendation

Once enough Stage 0 data is collected and a model is trained and admin-approved, AMP shows the model's recommendation alongside each workitem. The human still decides. AMP tracks how often the human agrees with the model.

**Promotion gate:** recommendation-human agreement rate must exceed a configured threshold (default 70%) over a minimum number of evaluations.

### Stage 2 — Narrow Automation

High-confidence model predictions skip human review. Only low-confidence or edge-case decisions still go to HITL. Requires explicit admin approval.

**The HITL override in your agent (`result["decision"] = "HUMAN_REVIEW"`):** remove this before Stage 2 activates, or the agent will override AMP's autonomous routing and send everything to HITL anyway.

---

## Reference: Environment & Configuration

### `.env` file

```bash
AMP_API_KEY=your-api-key
BACKEND_URL=https://your-amp-instance.example.com
AGENT_NAME=your-agent-name
AMP_ORG_ID=O-XXXX-...

# Your agent-specific vars (example)
OPENAI_API_KEY=sk-...
PORT=6000
```

All simulator scripts load this file automatically using the same manual parser as `agent.py`. Variables already set in the environment are not overwritten.

### AMP API calls used by the pattern

| Call | Purpose |
|---|---|
| `POST /api/agent/init` | Create an AMP instance for this request |
| `POST /api/agent/setState` | Transition state (wait / finished / abort) |
| `POST /api/hitl/request` | Request human review with RLHF payload |
| `GET /api/hitl/get-decision` | Poll for the human's decision |
| `GET /api/workitems` | List pending workitems (simulator) |
| `PUT /api/workitems/{id}/status` | Resolve a workitem (simulator) |
| `POST /api/rlhf/bootstrap` | Initialize RLHF runtime state (simulator) |
| `POST /api/rlhf/outcome` | Submit the labeled outcome (simulator) |
| `POST /api/rlhf/outcome/feedback` | Confirm the outcome immediately — called by the agent after HITL resolves |
| `GET /api/rlhf/policy` | Fetch the active policy (generator, policy mode) |
| `POST /api/rlhf/policy/evaluate` | Evaluate criteria against action params (generator) |
| `POST /api/alp/agents/.../progress` | Write progress messages to the instance log |
| `POST /api/alp/agents/.../artifacts/{name}` | Upload a result artifact |

---

## Troubleshooting

### TRAIN fails repeatedly with `dataset_jsonl is empty`

**Root cause:** RLHF outcomes are stuck as `pending`. The EXTRACT job only returns `confirmed` rows, so it writes an empty dataset.

**Fix:** Make sure your agent calls `POST /api/rlhf/outcome/feedback` after receiving the HITL decision (see Step 1c). If you cannot change the agent, set `outcome_window_hours: 0` in your policy's `params.outcome_feedback` — the settlement sweep will then confirm outcomes immediately on the next STAGE_TICK.

### `rlhf_bootstrap_failed_404: policy_not_found`

`AMP_ORG_ID` in your `.env` doesn't match what AMP has for this policy. Verify your org ID in AMP → Settings.

### `workitem_not_found_before_timeout`

The agent didn't send a HITL request within the polling window. Check:
- Is the agent running and reachable?
- Does the agent have a valid `AMP_API_KEY` and `BACKEND_URL`?
- Is the HITL-agent service running on the AMP server (`pm2 status`)?

### `Repeated job failures` red alert in AMP

This means 3+ jobs of the same type failed within 30 minutes. At Stage 0 this is a monitoring alert only — HITL still works. Check AMP → Agents → RLHF → Recent Jobs for the `last_error` field on the failing job type.

The most common root cause is TRAIN failing because of empty datasets (see above). Fix the outcome confirmation step and the alert will clear once the old failed jobs age out of the 30-minute window.

### Outcomes showing as `pending` in the database

This is normal until either:
- `/api/rlhf/outcome/feedback` is called (immediate) — the agent does this after each HITL resolution
- The `outcome_window_expires_at` timestamp passes (delayed by `outcome_window_hours`)

If you're checking the database directly, query `rlhf_approval_outcome` and look for `outcome_status = 'confirmed'`. If all rows are `pending`, the feedback step is missing or the policy has a non-zero `outcome_window_hours`.

### Agent resolves HITL but the simulator times out waiting for the workitem

The workitem poll loop looks for a pending approval workitem where `instance_id` starts with the agent name. If your agent name doesn't match `AGENT_NAME` in `.env`, or if the workitem appears under a different action type than `approval`, the poll won't find it. Verify the workitem appears in AMP → Workitems with the expected `instance_id` prefix.