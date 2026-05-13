"""
Document Triage Agent
Classifies and routes incoming documents.
Outputs: ACCEPT | REQUEST_MORE_INFO | HUMAN_REVIEW
"""

import os
import json
from datetime import datetime, timezone
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a document triage agent. Analyze the submitted document and respond with JSON only.

Classify the document and return:
{
  "doc_type": "invoice|contract|form|other",
  "missing_fields": ["list of missing required fields, empty if none"],
  "decision": "ACCEPT | REQUEST_MORE_INFO | HUMAN_REVIEW",
  "reason": "one sentence explanation"
}

Rules:
- ACCEPT: document type is clear and all key fields are present
- REQUEST_MORE_INFO: document is recognizable but missing critical fields
- HUMAN_REVIEW: document is ambiguous, unusual, or cannot be classified
"""

def triage_document(document_text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Document:\n\n{document_text}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return json.loads(response.choices[0].message.content)


def _normalize_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "met", "pass", "passed")
    return bool(value)


def _criterion_result(criterion):
    if not isinstance(criterion, dict):
        return False
    if "result" in criterion:
        return _normalize_bool(criterion.get("result"))
    status = str(criterion.get("status") or "").strip().lower()
    if status:
        return status in ("met", "pass", "passed", "true")
    return False


def _criterion_for_hitl(criterion):
    """Convert simulator/policy criteria into the compact HITL RLHF shape."""
    if not isinstance(criterion, dict):
        return None
    criterion_id = str(criterion.get("criterion_id") or "").strip()
    if not criterion_id:
        return None
    out = {"criterion_id": criterion_id, "result": _criterion_result(criterion)}
    for key in ("criterion_text", "text", "hardness", "evaluator_type", "signals"):
        if key in criterion and criterion[key] not in (None, ""):
            out[key] = criterion[key]
    return out


def _signals_from_criteria(criteria_override):
    signals = {}
    if not isinstance(criteria_override, list):
        return signals

    for criterion in criteria_override:
        if not isinstance(criterion, dict):
            continue

        criterion_signals = criterion.get("signals")
        if isinstance(criterion_signals, dict):
            for key, value in criterion_signals.items():
                if key:
                    signals[key] = value

        criterion_id = str(criterion.get("criterion_id") or "").strip()
        if not criterion_id:
            continue
        passed = _criterion_result(criterion)
        if criterion_id == "c_doc_type_recognized":
            signals.setdefault("doc_type_unrecognized", not passed)
        elif criterion_id == "c_no_missing_fields":
            signals.setdefault("has_missing_fields", not passed)
        elif criterion_id == "c_not_ambiguous":
            signals.setdefault("agent_flagged_ambiguous", not passed)

    return signals


def build_rlhf_payload(result, missing, owner, override_context=None):
    """Build RLHF context without letting the forced-HITL training mode distort features."""
    override_context = override_context if isinstance(override_context, dict) else {}
    action_override = override_context.get("action_fields") if isinstance(override_context.get("action_fields"), dict) else {}
    signal_override = override_context.get("signal_fields") if isinstance(override_context.get("signal_fields"), dict) else {}
    criteria_override = override_context.get("criteria") if isinstance(override_context.get("criteria"), list) else None
    signal_defaults = _signals_from_criteria(criteria_override)
    signal_fields = {**signal_defaults, **signal_override}

    original_decision = str(result.get("decision") or "").strip().upper()
    doc_type = str(action_override.get("doc_type") or result.get("doc_type") or "other").strip() or "other"
    missing_text = action_override.get("missing_fields")
    if missing_text is None:
        missing_text = ", ".join(missing) if missing else ""
    missing_count = action_override.get("missing_fields_count")
    if missing_count is None:
        missing_count = len(missing)

    agent_flagged_ambiguous = signal_fields.get("agent_flagged_ambiguous")
    if agent_flagged_ambiguous is None:
        agent_flagged_ambiguous = original_decision == "HUMAN_REVIEW"
    else:
        agent_flagged_ambiguous = _normalize_bool(agent_flagged_ambiguous)

    criteria = []
    if criteria_override is not None:
        criteria = [c for c in (_criterion_for_hitl(item) for item in criteria_override) if c]
    if not criteria:
        criteria = [
            {"criterion_id": "c_doc_type_recognized", "result": doc_type != "other"},
            {"criterion_id": "c_no_missing_fields", "result": len(missing) == 0},
            {"criterion_id": "c_not_ambiguous", "result": not agent_flagged_ambiguous},
        ]

    return {
        "policy_values_source": override_context.get("policy_values_source") or "external",
        "feature_schema_version": override_context.get("feature_schema_version") or "triage_v1",
        "action_fields": {
            **action_override,
            "doc_type": doc_type,
            "missing_fields_count": missing_count,
            "missing_fields": missing_text,
            "submitted_by": action_override.get("submitted_by") or owner,
        },
        "signal_fields": {
            **signal_fields,
            "doc_type_unrecognized": _normalize_bool(signal_fields.get("doc_type_unrecognized", doc_type == "other")),
            "has_missing_fields": _normalize_bool(signal_fields.get("has_missing_fields", len(missing) > 0)),
            "agent_flagged_ambiguous": agent_flagged_ambiguous,
        },
        "criteria": criteria,
    }


# --- Test inputs ---
TEST_DOCUMENTS = {
    "complete_invoice": """
        INVOICE #1042
        From: Acme Corp | To: Wayne Industries
        Date: 2026-04-07 | Due: 2026-04-21
        Item: Consulting Services - 10 hrs @ $150/hr
        Subtotal: $1,500 | Tax: $120 | Total: $1,620
        Payment: Bank transfer to Acme Corp account #4521
    """,

    "incomplete_contract": """
        SERVICE AGREEMENT
        This agreement is between [PARTY A] and Wayne Industries.
        Services: Software development for 3 months.
        Payment terms: TBD
        Signatures: ________________  ________________
    """,

    "gibberish": """
        asdf qwer zxcv 1234 hello world random text
        not a real document at all ??? !!!
    """
}


if __name__ == "__main__":
    import sys, time, requests
    from flask import Flask, request as freq, jsonify

    # --- run tests without AMP ---
    if "--test" in sys.argv:
        for name, doc in TEST_DOCUMENTS.items():
            print(f"\n{'='*50}\nTEST: {name}\n{'='*50}")
            print(json.dumps(triage_document(doc), indent=2))
        sys.exit(0)

    # --- AMP env ---
    AGENT  = os.environ["AGENT_NAME"]
    OWNER  = os.environ.get("OWNER_EMAIL", "")
    KEY    = os.environ["AMP_API_KEY"]
    URL    = os.environ["BACKEND_URL"].rstrip("/")
    H      = {"X-API-Key": KEY}
    HJ     = {**H, "Content-Type": "application/json"}
    BASE   = lambda iid: f"{URL}/api/alp/agents/{requests.utils.quote(AGENT, safe='')}/instances/{requests.utils.quote(iid, safe='')}"

    def amp_post(url, **kwargs):
        """POST to AMP, raising on connection errors or non-2xx responses."""
        try:
            r = requests.post(url, **kwargs)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"AMP POST {url} failed: {e}") from e

    def amp_get(url, **kwargs):
        """GET from AMP, raising on connection errors or non-2xx responses."""
        try:
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"AMP GET {url} failed: {e}") from e

    def amp_json(response):
        """Parse JSON from an AMP response, raising on invalid JSON."""
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            raise RuntimeError(f"AMP returned non-JSON response: {response.text[:200]}") from e

    def log(iid, message, level="INFO"):
        """Write to MySQL audit log AND meta.json progress (UI display). Fire-and-forget."""
        ts = datetime.now(timezone.utc).isoformat()
        try:
            requests.post(f"{URL}/api/log", headers=HJ, json={
                "instance_id": iid, "service": AGENT, "level": level,
                "username": OWNER, "message": message, "timestamp": ts},
                timeout=5)
            requests.post(f"{BASE(iid)}/progress", headers=HJ,
                          json={"message": message}, timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"[WARN] log delivery failed: {e}")

    app = Flask(__name__)

    @app.post("/submit")
    def submit():
        body = freq.get_json(force=True) or {}
        doc  = body.get("document", "")
        if not doc:
            return jsonify({"error": "document required"}), 400
        rlhf_context = body.get("rlhf_context") if isinstance(body.get("rlhf_context"), dict) else {}

        print(f"[DEBUG] OWNER={repr(OWNER)} URL={repr(URL)}")
        # 1. Create AMP instance
        try:
            init = amp_json(amp_post(f"{URL}/api/agent/init", headers=HJ,
                                     json={"agent_name": AGENT, "prompt": doc, "auto_start": True}))
        except RuntimeError as e:
            return jsonify({"error": "failed to create instance", "detail": str(e)}), 500
        iid = init.get("instance_id")
        if not iid:
            return jsonify({"error": "failed to create instance", "detail": init}), 500
        print(f"[INIT] {iid}")

        try:
            # 2. Mark active + triage
            
            log(iid, "instance created, starting triage")

            result  = triage_document(doc)
            missing = result.get("missing_fields") or []
            original_decision = result.get("decision")
            print(f"[TRIAGE] {result}")

            log(iid, f"  triage complete: decision={original_decision} type={result['doc_type']} missing={missing or 'none'}")
            rlhf_payload = build_rlhf_payload(result, missing, OWNER, rlhf_context)

            # Force all decisions through HITL for simulator training data collection.
            result["decision"] = "HUMAN_REVIEW"

            report = f"# Triage Report\n\n**Decision:** {result['decision']}\n**Type:** {result['doc_type']}\n**Reason:** {result['reason']}\n**Missing:** {', '.join(missing) or 'None'}"

            # 3. Upload artifact
            amp_post(f"{BASE(iid)}/artifacts/triage-result.md", headers=H, data=report.encode())
            log(iid, "  artifact uploaded: triage-result.md")

            # 4. Route
            if result["decision"] == "HUMAN_REVIEW":
                # 4a. Request HITL
                org = amp_json(amp_post(f"{URL}/api/internal/users/resolve", headers=HJ,
                                       json={"username": OWNER})).get("org_id")
                hitl_cfg = {"enable": True, "when": "always", "who": "initiator", "what": "approval", "where": "amp"}
                hitl_body = {"caller_id": iid, "instance_id": iid,
                             "org_id": str(org) if org else None,
                             "agent_name": AGENT, "hitl": hitl_cfg, "rlhf": rlhf_payload}
                print(f"[HITL REQUEST BODY]\n{json.dumps(hitl_body, indent=2)}")
                amp_post(f"{URL}/api/hitl/request", headers=HJ, json=hitl_body)
                log(iid, "HITL request sent, awaiting human review")

                # 4b. setState → wait, then poll for decision
                amp_post(f"{URL}/api/agent/setState", headers=HJ,
                         json={"agent_name": AGENT, "instance_id": iid, "state": "wait"})
                log(iid, "state → wait")

                for i in range(36):  # poll up to 3 min
                    time.sleep(5)
                    log(iid, f"  polling HITL decision: attempt {i+1}/36")
                    decision = amp_json(amp_get(f"{URL}/api/hitl/get-decision",
                                               params={"caller_id": iid}, headers=H))
                    if decision.get("status") == "complete":
                        resolution = decision.get("resolution", "unknown")

                        # Confirm the RLHF outcome immediately so it's eligible for
                        # training without waiting for the 48h settlement window.
                        # Non-fatal: settlement sweep will catch it eventually if this fails.
                        try:
                            requests.post(f"{URL}/api/rlhf/outcome/feedback", headers=HJ,
                                          json={"org_id": str(org) if org else None,
                                                "instance_id": iid,
                                                "outcome": "succeeded",
                                                "outcome_source": "agent_callback"},
                                          timeout=10)
                            log(iid, "  RLHF outcome confirmed")
                        except Exception as e:
                            log(iid, f"  RLHF outcome feedback failed (non-fatal): {e}", level="WARN")

                        if resolution == "modify":
                            # Extract modification instructions from information field
                            information = decision.get("information", "")
                            modify_instructions = ""
                            if "\n" in information:
                                modify_instructions = information.split("\n", 1)[1].strip()
                            log(iid, f"HITL modify requested: instructions={modify_instructions or '(none)'}")

                            # Re-triage with modification instructions as context
                            modified_prompt = doc
                            if modify_instructions:
                                modified_prompt = f"{doc}\n\nReviewer instructions: {modify_instructions}"
                            result = triage_document(modified_prompt)
                            missing = result.get("missing_fields") or []
                            log(iid, f"  re-triage complete: decision={result['decision']} type={result['doc_type']}")

                            report = (
                                f"# Triage Report (Modified)\n\n"
                                f"**Reviewer Instructions:** {modify_instructions or 'N/A'}\n\n"
                                f"**Decision:** {result['decision']}\n"
                                f"**Type:** {result['doc_type']}\n"
                                f"**Reason:** {result['reason']}\n"
                                f"**Missing:** {', '.join(missing) or 'None'}"
                            )
                            amp_post(f"{BASE(iid)}/artifacts/triage-result.md", headers=H, data=report.encode())
                            log(iid, "  updated artifact: triage-result.md")

                            amp_post(f"{URL}/api/agent/setState", headers=HJ,
                                     json={"agent_name": AGENT, "instance_id": iid, "state": "finished"})
                            log(iid, "state → finished (after modify)")
                            return jsonify({"instance_id": iid, "decision": result["decision"], "resolution": "modify",
                                            "modify_instructions": modify_instructions})

                        final_state = "abort" if resolution == "reject" else "finished"
                        amp_post(f"{URL}/api/agent/setState", headers=HJ,
                                 json={"agent_name": AGENT, "instance_id": iid, "state": final_state})
                        log(iid, f"HITL resolved: resolution={resolution} → state={final_state}")
                        return jsonify({"instance_id": iid, "decision": result["decision"], "resolution": resolution})

                log(iid, "  HITL poll timed out after 3 minutes", level="WARN")
                return jsonify({"instance_id": iid, "decision": result["decision"], "status": "waiting_for_human"})

            else:
                # 5. Auto-finish
                amp_post(f"{URL}/api/agent/setState", headers=HJ,
                         json={"agent_name": AGENT, "instance_id": iid, "state": "finished"})
                log(iid, f"  auto-finished: decision={result['decision']} reason={result['reason']}")
                return jsonify({"instance_id": iid, "decision": result["decision"], "reason": result["reason"],
                                "missing_fields": missing})

        except Exception as e:
            print(f"[ERROR] {e}")
            log(iid, f"  exception: {e}", level="ERROR")
            try:
                amp_post(f"{URL}/api/agent/setState", headers=HJ,
                         json={"agent_name": AGENT, "instance_id": iid, "state": "abort"})
            except RuntimeError as abort_err:
                print(f"[WARN] failed to set abort state: {abort_err}")
            return jsonify({"error": str(e)}), 500

    port = int(os.environ.get("PORT", 6000))
    print(f"Document Triage Agent listening on :{port}  (agent-initiated mode)")
    app.run(port=port)
