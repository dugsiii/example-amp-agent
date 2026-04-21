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
            print(f"[TRIAGE] {result}")

            log(iid, f"  triage complete: decision={result['decision']} type={result['doc_type']} missing={missing or 'none'}")

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
                rlhf_payload = {
                    "policy_values_source": "external",
                    "action_fields": {
                        "doc_type":            result["doc_type"],
                        "missing_fields_count": len(missing),
                        "missing_fields":      ", ".join(missing) if missing else "",
                        "submitted_by":        OWNER,
                    },
                    "signal_fields": {
                        "doc_type_unrecognized":  result["doc_type"] == "other",
                        "has_missing_fields":     len(missing) > 0,
                        "agent_flagged_ambiguous": True,  # always true on HUMAN_REVIEW path
                    },
                    "criteria": [
                        {"criterion_id": "c_doc_type_recognized", "result": result["doc_type"] != "other"},
                        {"criterion_id": "c_no_missing_fields",   "result": len(missing) == 0},
                        {"criterion_id": "c_not_ambiguous",       "result": False},  # always false on HUMAN_REVIEW path
                    ],
                }
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