"""
Document Triage Agent — End-to-End Simulator

Submits synthetic documents to the triage agent, automatically resolves HITL
workitems as the synthetic human reviewer, and feeds RLHF outcome labels back
to AMP — all without a real human in the loop.

Reads configuration from the .env file in the project root (parent of simulator/).
All values can be overridden with CLI flags.

Usage:
    # 1. Generate docs first
    python simulator/doc_generator.py --output simulator/docs.csv --count 50

    # 2. Run the simulator (values come from .env automatically)
    python simulator/run_e2e.py

    # Override individual values as needed
    python simulator/run_e2e.py --limit 10 --reviewer sim-reviewer-01
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Load .env from project root (parent of simulator/) into os.environ.
# Only sets variables that aren't already in the environment, matching
# standard dotenv behaviour.
_ENV_FILE = Path(__file__).resolve().parents[1] / '.env'
if _ENV_FILE.exists():
    with open(_ENV_FILE, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith('#') or '=' not in _line:
                continue
            _k, _, _v = _line.partition('=')
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _k and _k not in os.environ:
                os.environ[_k] = _v


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _load_rows(csv_path: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(dict(row))
            if limit and len(out) >= limit:
                break
    return out


def _submit_document(
    agent_url: str,
    api_key: str,
    row: Dict[str, str],
    timeout_sec: float = 240.0,
) -> Dict[str, Any]:
    """POST the document to /submit and return the response JSON.

    The agent's /submit blocks for up to ~3 minutes on HUMAN_REVIEW cases
    while it polls for a HITL decision. Callers should run this in a thread
    for those rows so the simulator can concurrently resolve the workitem.
    """
    url = f"{agent_url.rstrip('/')}/submit"
    headers = {'X-API-Key': api_key}
    payload = {
        'doc_id': row['doc_id'],
        'document': row['doc_text'],
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"submit_timeout_or_error: {exc}") from exc
    if resp.status_code >= 400:
        raise RuntimeError(f"submit_failed_{resp.status_code}: {resp.text}")
    return resp.json() if resp.content else {}


def _poll_for_workitem(
    amp_url: str,
    api_key: str,
    instance_id: str,
    agent_name: str,
    timeout_sec: int = 120,
) -> Optional[Dict[str, Any]]:
    """Poll GET /api/workitems until a pending approval workitem appears for
    the given instance_id. Returns the workitem dict or None on timeout."""
    deadline = time.time() + timeout_sec
    headers = {'X-API-Key': api_key}
    workitems_url = f"{amp_url.rstrip('/')}/api/workitems"

    while time.time() < deadline:
        try:
            resp = requests.get(workitems_url, headers=headers, timeout=60.0)
        except requests.exceptions.RequestException:
            time.sleep(0.5)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(
                f"workitems_fetch_failed_{resp.status_code}: {resp.text}"
            )
        data = resp.json() if resp.content else {}
        rows = data.get('workitems') if isinstance(data, dict) else []
        if not isinstance(rows, list):
            rows = []

        for wi in rows:
            if str(wi.get('instance_id') or '').strip() != instance_id:
                continue
            if str(wi.get('status') or '').strip().lower() != 'pending':
                continue
            if str(wi.get('action') or '').strip().lower() != 'approval':
                continue
            return wi

        time.sleep(0.5)

    return None


def _complete_workitem(
    amp_url: str,
    api_key: str,
    workitem_id: str,
    resolution: str,
) -> None:
    """PUT /api/workitems/{id}/status with the ground-truth resolution."""
    headers = {'X-API-Key': api_key}
    url = f"{amp_url.rstrip('/')}/api/workitems/{workitem_id}/status"
    payload = {'status': 'Complete', 'resolution': resolution}
    try:
        resp = requests.put(url, json=payload, headers=headers, timeout=60.0)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"workitem_complete_timeout_or_error: {exc}") from exc
    if resp.status_code >= 400:
        raise RuntimeError(
            f"workitem_complete_failed_{resp.status_code}: {resp.text}"
        )


def _submit_rlhf_outcome(
    amp_url: str,
    api_key: str,
    org_id: str,
    agent_name: str,
    decision_point_id: str,
    decision: str,
    reviewer_id: str,
    sim_run_id: str,
    sequence_no: int,
) -> None:
    """POST /api/rlhf/outcome to feed the human label back to AMP."""
    headers = {'X-API-Key': api_key}
    url = f"{amp_url.rstrip('/')}/api/rlhf/outcome"
    now_iso = _iso_now()
    payload = {
        'org_id': org_id,
        'agent_name': agent_name,
        'decision_point_id': decision_point_id,
        'human': {
            'decision': decision,
            'user_id': reviewer_id,
            'role': 'sim_human',
            'comment': f'simulated_outcome seq={sequence_no}',
        },
        'run_mode': 'sim',
        'sim_run_id': sim_run_id,
        'sim_time': now_iso,
        'checkpoint_id': f'seq-{sequence_no}',
        'completed_at': now_iso,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60.0)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"rlhf_outcome_submit_timeout_or_error: {exc}"
        ) from exc
    if resp.status_code >= 400:
        raise RuntimeError(
            f"rlhf_outcome_submit_failed_{resp.status_code}: {resp.text}"
        )


def _submit_outcome_feedback(
    amp_url: str,
    api_key: str,
    org_id: str,
    instance_id: str,
) -> None:
    """POST /api/rlhf/outcome/feedback to immediately confirm the pending outcome.

    Without this, outcomes stay 'pending' for 48 hours (default settlement window)
    and fetchDatasetWindow skips them — EXTRACT writes an empty JSONL and TRAIN
    fails with 'dataset_jsonl is empty'. Submitting feedback with outcome=succeeded
    immediately sets outcome_status='confirmed', making rows visible to EXTRACT.

    Both approve and reject decisions use outcome='succeeded': the human decision
    is always considered correct ground truth in the simulator context.
    """
    headers = {'X-API-Key': api_key}
    url = f"{amp_url.rstrip('/')}/api/rlhf/outcome/feedback"
    payload = {
        'org_id': org_id,
        'instance_id': instance_id,
        'outcome': 'succeeded',
        'outcome_source': 'agent_callback',
        'notes': 'simulator_immediate_confirmation',
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60.0)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"outcome_feedback_timeout_or_error: {exc}") from exc
    if resp.status_code >= 400:
        raise RuntimeError(
            f"outcome_feedback_failed_{resp.status_code}: {resp.text}"
        )


def _process_hitl_row(
    row: Dict[str, str],
    agent_url: str,
    amp_url: str,
    api_key: str,
    org_id: str,
    agent_name: str,
    reviewer_id: str,
    sim_run_id: str,
    seq: int,
    hitl_wait_timeout: int,
    result_holder: Dict[str, Any],
) -> None:
    """Orchestrate a single HUMAN_REVIEW row end-to-end.

    Called on the main thread; runs _submit_document in a background thread
    (because it blocks while the agent polls HITL), then resolves the workitem
    and waits for the submit thread to finish.
    """
    doc_id = row['doc_id']
    scenario = row['scenario']
    expected_resolution = row['expected_human_resolution']  # 'approve' or 'reject'

    # Launch blocking /submit in background so we can concurrently resolve WI.
    holder: Dict[str, Any] = {}

    def _bg_submit() -> None:
        try:
            holder['response'] = _submit_document(
                agent_url=agent_url,
                api_key=api_key,
                row=row,
                timeout_sec=float(hitl_wait_timeout + 60),
            )
        except Exception as exc:
            holder['error'] = str(exc)

    t = threading.Thread(target=_bg_submit, daemon=True)
    t.start()

    # Give the agent a moment to create the instance and post the HITL request.
    time.sleep(2.0)

    # We don't know instance_id yet — poll for any new pending workitem for
    # this agent that is not yet in seen_workitems. Use agent_name prefix match.
    deadline = time.time() + hitl_wait_timeout
    wi: Optional[Dict[str, Any]] = None
    headers = {'X-API-Key': api_key}
    workitems_url = f"{amp_url.rstrip('/')}/api/workitems"

    while time.time() < deadline:
        # Check if submit already errored.
        if 'error' in holder:
            raise RuntimeError(f"submit_error: {holder['error']}")

        try:
            resp = requests.get(workitems_url, headers=headers, timeout=60.0)
        except requests.exceptions.RequestException:
            time.sleep(0.5)
            continue

        if resp.status_code >= 400:
            raise RuntimeError(
                f"workitems_fetch_failed_{resp.status_code}: {resp.text}"
            )

        data = resp.json() if resp.content else {}
        rows_wi = data.get('workitems') if isinstance(data, dict) else []
        if not isinstance(rows_wi, list):
            rows_wi = []

        for wi_candidate in rows_wi:
            wi_id = str(wi_candidate.get('workitem_id') or '').strip()
            if not wi_id:
                continue
            instance_id = str(wi_candidate.get('instance_id') or '').strip()
            if not instance_id.startswith(agent_name + '-'):
                continue
            if str(wi_candidate.get('status') or '').strip().lower() != 'pending':
                continue
            if str(wi_candidate.get('action') or '').strip().lower() != 'approval':
                continue
            # Exclude RLHF admin workitems.
            wi_id_lower = wi_id.lower()
            if wi_id_lower.startswith('rlhf-model-approval-') or wi_id_lower.startswith('rlhf-stage2-promotion-'):
                continue
            # Match to this doc: check that the workitem appeared after we
            # submitted (it's new enough — we track seen per run in result_holder).
            seen: set = result_holder.get('_seen_workitems', set())
            if wi_id in seen:
                continue
            wi = wi_candidate
            break

        if wi is not None:
            break
        time.sleep(0.5)

    if wi is None:
        t.join(timeout=10)
        raise RuntimeError(
            f"workitem_not_found_before_timeout: doc_id={doc_id}"
        )

    workitem_id = str(wi.get('workitem_id') or '')
    instance_id = str(wi.get('instance_id') or '').strip()
    result_holder.setdefault('_seen_workitems', set()).add(workitem_id)

    # Resolve workitem with ground-truth label.
    _complete_workitem(
        amp_url=amp_url,
        api_key=api_key,
        workitem_id=workitem_id,
        resolution=expected_resolution,
    )

    # Submit RLHF outcome if a decision point is attached.
    decision_point_id = str(wi.get('rlhf_decision_point_id') or '').strip()
    rlhf_submitted = False
    if decision_point_id:
        _submit_rlhf_outcome(
            amp_url=amp_url,
            api_key=api_key,
            org_id=org_id,
            agent_name=agent_name,
            decision_point_id=decision_point_id,
            decision=expected_resolution,
            reviewer_id=reviewer_id,
            sim_run_id=sim_run_id,
            sequence_no=seq,
        )
        # Immediately confirm the outcome so it's eligible for training.
        # Without this, outcomes stay 'pending' for 48h (default settlement
        # window) and fetchDatasetWindow skips them — TRAIN fails on empty JSONL.
        if instance_id:
            _submit_outcome_feedback(
                amp_url=amp_url,
                api_key=api_key,
                org_id=org_id,
                instance_id=instance_id,
            )
        rlhf_submitted = True

    # Wait for the submit thread to finish (agent gets the decision and returns).
    t.join(timeout=float(hitl_wait_timeout + 30))
    if t.is_alive():
        raise RuntimeError(f"submit_thread_did_not_finish: doc_id={doc_id}")
    if 'error' in holder:
        raise RuntimeError(f"submit_error: {holder['error']}")

    submit_response = holder.get('response', {})
    agent_decision = str(submit_response.get('decision') or '').strip()

    result_holder['result'] = {
        'doc_id': doc_id,
        'scenario': scenario,
        'status': 'pass',
        'expected_decision': 'HUMAN_REVIEW',
        'agent_decision': agent_decision,
        'hitl_resolved': True,
        'resolution': expected_resolution,
        'workitem_id': workitem_id,
        'rlhf_submitted': rlhf_submitted,
    }


def _bootstrap_rlhf(amp_url: str, api_key: str, org_id: str, agent_name: str) -> None:
    """Call /api/rlhf/bootstrap to initialize RLHF runtime state before the sim loop."""
    headers = {'X-API-Key': api_key}
    url = f"{amp_url.rstrip('/')}/api/rlhf/bootstrap"
    payload = {
        'org_id': org_id,
        'agent_name': agent_name,
        'enqueue_stage_tick': True,
        'force_reset_state': False,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30.0)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"rlhf_bootstrap_failed: {exc}") from exc
    if resp.status_code >= 400:
        raise RuntimeError(f"rlhf_bootstrap_failed_{resp.status_code}: {resp.text}")
    body = resp.json() if resp.content else {}
    print(json.dumps({
        'event': 'rlhf_bootstrap',
        'ok': bool(body.get('ok')),
        'enabled': bool(body.get('enabled')),
        'stage_tick_enqueued': bool(body.get('stage_tick_enqueued')),
        'policy_source': body.get('policy_source'),
    }))
    if not body.get('enabled'):
        raise RuntimeError(
            'rlhf_not_enabled: bootstrap succeeded but RLHF is not enabled for this agent — '
            'enable RLHF in the AMP policy before running the simulator'
        )


def run(args: argparse.Namespace) -> None:
    rows = _load_rows(args.csv, args.limit)
    if not rows:
        raise SystemExit('no_rows_loaded')

    sim_run_id = args.sim_run_id or f"sim-{uuid.uuid4().hex[:8]}"

    print(json.dumps({
        'event': 'sim_start',
        'rows': len(rows),
        'org_id': args.org_id,
        'agent_name': args.agent_name,
        'agent_url': args.agent_url,
        'amp_url': args.amp_url,
        'sim_run_id': sim_run_id,
    }))

    _bootstrap_rlhf(
        amp_url=args.amp_url,
        api_key=args.api_key,
        org_id=args.org_id,
        agent_name=args.agent_name,
    )

    passed = 0
    errors = 0
    # Shared state across rows for workitem deduplication.
    state: Dict[str, Any] = {'_seen_workitems': set()}

    for idx, row in enumerate(rows, start=1):
        doc_id = row['doc_id']
        scenario = row['scenario']

        try:
            result_holder: Dict[str, Any] = {'_seen_workitems': state['_seen_workitems']}
            _process_hitl_row(
                row=row,
                agent_url=args.agent_url,
                amp_url=args.amp_url,
                api_key=args.api_key,
                org_id=args.org_id,
                agent_name=args.agent_name,
                reviewer_id=args.reviewer,
                sim_run_id=sim_run_id,
                seq=idx,
                hitl_wait_timeout=args.hitl_wait_timeout,
                result_holder=result_holder,
            )
            state['_seen_workitems'] = result_holder.get('_seen_workitems', set())
            entry = result_holder.get('result', {})
            print(json.dumps(entry))
            passed += 1

        except Exception as exc:
            errors += 1
            print(json.dumps({
                'doc_id': doc_id,
                'scenario': scenario,
                'status': 'error',
                'error': str(exc),
            }))

    print(json.dumps({
        'event': 'sim_end',
        'total': len(rows),
        'passed': passed,
        'errors': errors,
        'sim_run_id': sim_run_id,
    }))


def main() -> None:
    # Derive agent URL from PORT env var if set (matches how agent.py uses PORT).
    _port = 6000
    _default_agent_url = f"http://localhost:{_port}"
    _default_amp_url = os.environ["BACKEND_URL"]
    _default_api_key = os.environ["AMP_API_KEY"]
    _default_org_id = os.environ["AMP_ORG_ID"]
    _default_agent_name = os.environ["AGENT_NAME"]

    parser = argparse.ArgumentParser(
        description='Document triage agent end-to-end simulator'
    )
    parser.add_argument(
        '--csv',
        default='simulator/docs.csv',
        help='Path to the generated documents CSV',
    )
    parser.add_argument(
        '--agent',
        dest='agent_url',
        default=_default_agent_url,
        help=f'Base URL of the document-triage agent (default: {_default_agent_url})',
    )
    parser.add_argument(
        '--amp',
        dest='amp_url',
        default=_default_amp_url,
        help=f'Base URL of the AMP backend (default from BACKEND_URL)',
    )
    parser.add_argument(
        '--api-key',
        dest='api_key',
        default=_default_api_key,
        help='AMP API key — defaults to AMP_API_KEY from .env',
    )
    parser.add_argument(
        '--org-id',
        dest='org_id',
        default=_default_org_id,
        help='AMP org ID for RLHF outcome payloads — defaults to AMP_ORG_ID from .env',
    )
    parser.add_argument(
        '--agent-name',
        dest='agent_name',
        default=_default_agent_name,
        help='Agent name used to filter workitems — defaults to AGENT_NAME from .env',
    )
    parser.add_argument(
        '--reviewer',
        dest='reviewer',
        default='sim-reviewer-01',
        help='Reviewer user_id used in RLHF outcome payloads',
    )
    parser.add_argument(
        '--sim-run-id',
        dest='sim_run_id',
        default='',
        help='Simulation run ID for grouping outcomes (auto-generated if omitted)',
    )
    parser.add_argument(
        '--hitl-wait-timeout',
        dest='hitl_wait_timeout',
        type=int,
        default=120,
        help='Seconds to wait for a HITL workitem to appear (default: 120)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Process only the first N rows (useful for quick tests)',
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit('AMP_API_KEY not set in .env and --api-key not provided')

    run(args)


if __name__ == '__main__':
    main()
