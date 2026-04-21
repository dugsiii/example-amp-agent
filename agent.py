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

    def log(iid, message, level="INFO"):
        """Write to MySQL audit log AND meta.json progress (UI display)."""
        ts = datetime.now(timezone.utc).isoformat()
        requests.post(f"{URL}/api/log", headers=HJ, json={
            "instance_id": iid, "service": AGENT, "level": level,
            "username": OWNER, "message": message, "timestamp": ts})
        requests.post(f"{BASE(iid)}/progress", headers=HJ, json={"message": message})

    app = Flask(__name__)

    @app.post("/submit")
    def submit():
        body = freq.get_json(force=True) or {}
        doc  = body.get("document", "")
        if not doc:
            return jsonify({"error": "document required"}), 400

        print(f"[DEBUG] OWNER={repr(OWNER)} URL={repr(URL)}")
        # 1. Create AMP instance
        init = requests.post(f"{URL}/api/agent/init", headers=HJ,
                             json={"agent_name": AGENT, "prompt": doc, "auto_start": True}).json()
        iid = init.get("instance_id")
        if not iid:
            return jsonify({"error": "failed to create instance", "detail": init}), 500
        print(f"[INIT] {iid}")

        try:
            # 2. Mark active + triage
            
            requests.post(f"{URL}/api/log", headers=HJ, json={
                "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                "message": "[PROGRESS] instance created, starting triage",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "record"})

            result  = triage_document(doc)
            missing = result.get("missing_fields") or []
            print(f"[TRIAGE] {result}")

            requests.post(f"{URL}/api/log", headers=HJ, json={
                "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                "message": f"[PROGRESS] triage complete: decision={result['decision']} type={result['doc_type']} missing={missing or 'none'}",
                "timestamp": datetime.now(timezone.utc).isoformat()})

            report = f"# Triage Report\n\n**Decision:** {result['decision']}\n**Type:** {result['doc_type']}\n**Reason:** {result['reason']}\n**Missing:** {', '.join(missing) or 'None'}"

            # 3. Upload artifact
            requests.post(f"{BASE(iid)}/artifacts/triage-result.md", headers=H, data=report.encode())
            requests.post(f"{URL}/api/log", headers=HJ, json={
                "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                "message": "[PROGRESS] artifact uploaded: triage-result.md",
                "timestamp": datetime.now(timezone.utc).isoformat()})

            # 4. Route
            if result["decision"] == "HUMAN_REVIEW":
                # 4a. Request HITL
                org = requests.post(f"{URL}/api/internal/users/resolve", headers=HJ,
                                    json={"username": OWNER}).json().get("org_id")
                hitl_cfg = {"enable": True, "when": "always", "who": "initiator", "what": "approval", "where": "amp"}
                requests.post(f"{URL}/api/hitl/request", headers=HJ,
                              json={"caller_id": iid, "instance_id": iid,
                                    "org_id": str(org) if org else None,
                                    "agent_name": AGENT, "hitl": hitl_cfg})
                requests.post(f"{URL}/api/log", headers=HJ, json={
                    "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                    "message": "HITL request sent, awaiting human review",
                    "timestamp": datetime.now(timezone.utc).isoformat()})

                # 4b. setState → wait, then poll for decision
                requests.post(f"{URL}/api/agent/setState", headers=HJ,
                              json={"agent_name": AGENT, "instance_id": iid, "state": "wait"})
                requests.post(f"{URL}/api/log", headers=HJ, json={
                    "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                    "message": "state → wait",
                    "timestamp": datetime.now(timezone.utc).isoformat()})

                for i in range(36):  # poll up to 3 min
                    time.sleep(5)
                    requests.post(f"{URL}/api/log", headers=HJ, json={
                        "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                        "message": f"[PROGRESS] polling HITL decision: attempt {i+1}/36",
                        "timestamp": datetime.now(timezone.utc).isoformat()})
                    decision = requests.get(f"{URL}/api/hitl/get-decision",
                                            params={"caller_id": iid}, headers=H).json()
                    if decision.get("status") == "complete":
                        resolution = decision.get("resolution", "unknown")
                        final_state = "abort" if resolution == "reject" else "finished"
                        requests.post(f"{URL}/api/agent/setState", headers=HJ,
                                      json={"agent_name": AGENT, "instance_id": iid, "state": final_state})
                        requests.post(f"{URL}/api/log", headers=HJ, json={
                            "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                            "message": f"HITL resolved: resolution={resolution} → state={final_state}",
                            "timestamp": datetime.now(timezone.utc).isoformat()})
                        return jsonify({"instance_id": iid, "decision": result["decision"], "resolution": resolution})

                requests.post(f"{URL}/api/log", headers=HJ, json={
                    "instance_id": iid, "service": AGENT, "level": "WARN", "username": OWNER,
                    "message": "[PROGRESS] HITL poll timed out after 3 minutes",
                    "timestamp": datetime.now(timezone.utc).isoformat()})
                return jsonify({"instance_id": iid, "decision": result["decision"], "status": "waiting_for_human"})

            else:
                # 5. Auto-finish
                requests.post(f"{URL}/api/agent/setState", headers=HJ,
                              json={"agent_name": AGENT, "instance_id": iid, "state": "finished"})
                requests.post(f"{URL}/api/log", headers=HJ, json={
                    "instance_id": iid, "service": AGENT, "level": "INFO", "username": OWNER,
                    "message": f"[PROGRESS] auto-finished: decision={result['decision']} reason={result['reason']}",
                    "timestamp": datetime.now(timezone.utc).isoformat()})
                return jsonify({"instance_id": iid, "decision": result["decision"], "reason": result["reason"],
                                "missing_fields": missing})

        except Exception as e:
            print(f"[ERROR] {e}")
            requests.post(f"{URL}/api/log", headers=HJ, json={
                "instance_id": iid, "service": AGENT, "level": "ERROR", "username": OWNER,
                "message": f"[PROGRESS] exception: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()})
            requests.post(f"{URL}/api/agent/setState", headers=HJ,
                          json={"agent_name": AGENT, "instance_id": iid, "state": "abort"})
            return jsonify({"error": str(e)}), 500

    port = int(os.environ.get("PORT", 6000))
    print(f"Document Triage Agent listening on :{port}  (agent-initiated mode)")
    app.run(port=port)
