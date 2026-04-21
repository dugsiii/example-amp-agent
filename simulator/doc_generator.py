"""
Document Triage Agent — Synthetic Document Generator

Generates a CSV of synthetic documents with ground-truth labels for the simulator.
Supports static criteria generation and policy-mode criteria fetch from AMP.

Usage:
    # Static mode (default)
    python simulator/doc_generator.py --output simulator/docs.csv --count 50

    # Policy mode — fetches live criteria from AMP
    python simulator/doc_generator.py \
        --schema-source policy \
        --output simulator/docs.csv --count 50
"""

import argparse
import csv
import json
import logging
import os
import random
import string
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Load .env from project root (parent of simulator/) into os.environ.
# Only sets variables not already in the environment — same pattern as run_e2e.py.
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

PROFILE_PATH = Path(__file__).parent / 'domain_profile.json'


# ---------------------------------------------------------------------------
# Policy runtime client (mirrors payment-agent pattern exactly)
# ---------------------------------------------------------------------------

class PolicyRuntimeClient:
    def __init__(
        self,
        amp_backend_url: str,
        amp_api_key: str,
        org_id: str,
        agent_name: str,
        tool_name: str,
        action_name: str,
        timeout_sec: float,
        logger: logging.Logger,
    ):
        self.amp_backend_url = amp_backend_url.rstrip('/')
        self.amp_api_key = amp_api_key
        self.org_id = org_id
        self.agent_name = agent_name
        self.tool_name = tool_name
        self.action_name = action_name
        self.timeout_sec = timeout_sec
        self.logger = logger

    def _headers(self) -> Dict[str, str]:
        return {'X-API-Key': self.amp_api_key}

    def _request_json(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        url = f"{self.amp_backend_url}{path}"
        body = json.dumps(payload or {}).encode('utf-8') if method.upper() != 'GET' else None
        req = urllib.request.Request(url=url, method=method.upper(), data=body)
        for k, v in self._headers().items():
            req.add_header(k, v)
        if body is not None:
            req.add_header('Content-Type', 'application/json')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                parsed = json.loads(raw.decode('utf-8'))
                return parsed if isinstance(parsed, dict) else None
        except urllib.error.HTTPError as exc:
            try:
                text = exc.read().decode('utf-8', errors='replace')
            except Exception:
                text = str(exc)
            self.logger.warning('policy request failed method=%s path=%s status=%s body=%s', method, path, exc.code, text)
            return None
        except Exception as exc:
            self.logger.warning('policy request error method=%s path=%s error=%s', method, path, exc)
            return None

    def fetch_context(self) -> Optional['PolicyContext']:
        if not self.amp_backend_url or not self.amp_api_key or not self.agent_name:
            self.logger.warning(
                'policy context disabled: missing config amp_backend_url=%s api_key_set=%s agent_name=%s',
                bool(self.amp_backend_url), bool(self.amp_api_key), bool(self.agent_name),
            )
            return None

        params: Dict[str, str] = {'agent_name': self.agent_name}
        if self.org_id:
            params['org_id'] = self.org_id
        query = urllib.parse.urlencode(params)
        body = self._request_json('GET', f'/api/rlhf/policy?{query}')
        if not isinstance(body, dict):
            return None

        policy_json = body.get('policy_json') if isinstance(body.get('policy_json'), dict) else {}
        action_governance = policy_json.get('action_governance') if isinstance(policy_json.get('action_governance'), dict) else {}
        criteria_raw = action_governance.get('criteria')
        criteria_meta: Dict[str, Dict[str, Any]] = {}
        if isinstance(criteria_raw, list):
            for item in criteria_raw:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get('criterion_id') or '').strip()
                if not cid:
                    continue
                text = str(item.get('text') or item.get('description') or '').strip()
                evaluator = item.get('evaluator') if isinstance(item.get('evaluator'), dict) else {}
                inputs = evaluator.get('inputs') if isinstance(evaluator.get('inputs'), dict) else {}
                input_keys: List[str] = []
                for raw in inputs.values():
                    if not isinstance(raw, str):
                        continue
                    s = raw.strip()
                    if s.startswith('{{') and s.endswith('}}'):
                        s = s[2:-2].strip()
                    if s.startswith('action.'):
                        s = s[len('action.'):]
                    if s and s not in input_keys:
                        input_keys.append(s)
                criteria_meta[cid] = {'text': text, 'input_keys': input_keys}

        return PolicyContext(
            policy_id=str(policy_json.get('policy_id') or ''),
            policy_version=str(policy_json.get('policy_version') or body.get('policy_version') or ''),
            feature_schema_version=str(
                body.get('feature_schema_version')
                or policy_json.get('feature_schema_version')
                or 'triage_v1'
            ),
            action_fields_allowlist=[
                str(x) for x in (action_governance.get('action_fields_allowlist') or [])
                if isinstance(x, str) and x.strip()
            ],
            signal_keys_allowlist=[
                str(x) for x in (action_governance.get('signal_keys_allowlist') or [])
                if isinstance(x, str) and x.strip()
            ],
            criteria_meta=criteria_meta,
        )

    def evaluate(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            'agent_name': self.agent_name,
            'params': params,
            'tool_name': self.tool_name,
            'action_name': self.action_name,
        }
        if self.org_id:
            payload['org_id'] = self.org_id
        return self._request_json('POST', '/api/rlhf/policy/evaluate', payload)


@dataclass
class PolicyContext:
    policy_id: str
    policy_version: str
    feature_schema_version: str
    action_fields_allowlist: List[str]
    signal_keys_allowlist: List[str]
    criteria_meta: Dict[str, Dict[str, Any]]


@dataclass
class DocumentRecord:
    doc_id: str
    scenario: str
    doc_type: str
    expected_human_resolution: str
    criteria_json: str
    feature_schema_version: str
    sim_run_id: str
    seed: int
    sequence_no: int
    doc_text: str


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

FOREIGN_TEXTS = [
    (
        "FACTURA No. {invoice_id}\nDe: {vendor} | Para: {client}\nFecha: {date}\n"
        "Descripción: {item_desc}\nImporte: ${amount} USD\nCondiciones: Pago en 30 días",
        "Invoice from {vendor} to {client}, amount ${amount}, payment in 30 days"
    ),
    (
        "RECHNUNG Nr. {invoice_id}\nVon: {vendor} | An: {client}\nDatum: {date}\n"
        "Leistung: {item_desc}\nBetrag: ${amount} USD\nZahlungsziel: 30 Tage",
        "Invoice from {vendor} to {client}, amount ${amount}, payment within 30 days"
    ),
    (
        "请款单 #{invoice_id}\n供应商: {vendor} | 客户: {client}\n日期: {date}\n"
        "服务项目: {item_desc}\n金额: ${amount} USD\n付款条件: 30天",
        "Invoice from {vendor} to {client}, amount ${amount}, 30-day payment terms"
    ),
]

LEGAL_BOILERPLATES = [
    "Nothing herein shall be construed as a waiver of any rights or obligations under applicable law.",
    "The parties acknowledge that this notice is subject to all applicable statutes of limitations.",
    "All communications regarding this matter should be directed to the undersigned's legal representative.",
    "This notice is provided in accordance with applicable regulatory requirements and does not create any new legal obligations.",
]

DOC_TYPE_MAP = {
    'complete_invoice':  'invoice',
    'complete_contract': 'contract',
    'incomplete_form':   'form',
    'missing_signature': 'contract',
    'gibberish':         'other',
    'ambiguous_legal':   'other',
    'suspicious_amount': 'invoice',
    'foreign_language':  'invoice',
}


def _rand_date(rng: random.Random) -> str:
    start = date(2025, 1, 1)
    end = date(2026, 12, 31)
    return (start + timedelta(days=rng.randint(0, (end - start).days))).strftime('%Y-%m-%d')


def _due_date(base: str, days: int = 30) -> str:
    return (date.fromisoformat(base) + timedelta(days=days)).strftime('%Y-%m-%d')


def _rand_id(rng: random.Random, prefix: str = '', length: int = 4) -> str:
    digits = ''.join(rng.choices(string.digits, k=length))
    return f"{prefix}{digits}" if prefix else digits


def _rand_matter(rng: random.Random) -> str:
    return f"{rng.randint(2024, 2026)}-{_rand_id(rng, length=5)}-{''.join(rng.choices(string.ascii_uppercase, k=2))}"


def _gibberish(rng: random.Random, words: List[str]) -> str:
    lines = []
    for _ in range(rng.randint(3, 6)):
        lines.append(' '.join(rng.choices(words, k=rng.randint(4, 10))))
    lines.append(' '.join([str(rng.randint(0, 9999)) for _ in range(rng.randint(4, 8))]))
    lines.append('??? !!! @@@ ### $$$')
    return '\n'.join(lines)


def _render_template(template: str, rng: random.Random, profile: Dict[str, Any]) -> str:
    vendors: List[str] = profile.get('vendors', ['Vendor Corp'])
    clients: List[str] = profile.get('clients', ['Client Inc'])
    items: List[str] = profile.get('items', ['Consulting Services'])
    depts: List[str] = profile.get('departments', ['Operations'])
    states: List[str] = profile.get('states', ['California'])
    gibberish_words: List[str] = profile.get('gibberish_words', ['asdf', 'qwer'])

    vendor = rng.choice(vendors)
    client = rng.choice(clients)
    item_desc = rng.choice(items)
    dept = rng.choice(depts)
    state = rng.choice(states)
    date_str = _rand_date(rng)
    due = _due_date(date_str, rng.choice([15, 30, 45, 60]))
    expense_date = _due_date(date_str, -rng.randint(1, 14))
    amount = rng.randint(500, 50000)
    rate = rng.randint(50, 500)
    qty = rng.randint(1, 40)
    subtotal = rate * qty
    tax = round(subtotal * 0.08)
    total = subtotal + tax
    monthly_rate = rng.randint(2000, 15000)
    duration = rng.randint(3, 18)
    contract_total = monthly_rate * duration
    terms = rng.choice([15, 30, 45, 60])
    invoice_id = _rand_id(rng, prefix='INV-', length=4)
    po_number = _rand_id(rng, prefix='PO-', length=6)
    account_no = ''.join(rng.choices(string.digits, k=10))
    routing_no = ''.join(rng.choices(string.digits, k=9))
    matter_id = _rand_matter(rng)
    sig1 = f"/s/ {rng.choice(['J. Smith', 'A. Johnson', 'M. Brown', 'K. Davis', 'R. Wilson'])}"
    sig2 = f"/s/ {rng.choice(['L. Taylor', 'C. Anderson', 'T. Martinez', 'P. Harris', 'N. Clark'])}"
    name = rng.choice(['Alice Chen', 'Bob Kumar', 'Carol White', 'David Park', 'Emma Torres'])
    category = rng.choice(['Travel', 'Software', 'Equipment', 'Training', 'Supplies'])
    purpose = rng.choice([
        'Client meeting travel expenses', 'Annual software license renewal',
        'Office equipment for remote work', 'Professional development conference',
        'Project-related supplies',
    ])
    suspicious_amount = rng.choice([999999, 1234567, 750000, 888888, 500000])
    legal_boilerplate = rng.choice(LEGAL_BOILERPLATES)

    foreign_entry = rng.choice(FOREIGN_TEXTS)
    foreign_template, translation_note = foreign_entry
    foreign_text = foreign_template.format(
        invoice_id=invoice_id, vendor=vendor, client=client,
        date=date_str, item_desc=item_desc, amount=amount,
    )
    translation_note = translation_note.format(vendor=vendor, client=client, amount=amount)

    vars_map = {
        'invoice_id': invoice_id, 'vendor': vendor, 'client': client,
        'item_desc': item_desc, 'dept': dept, 'state': state,
        'date': date_str, 'due_date': due, 'expense_date': expense_date,
        'amount': f"{amount:,}", 'rate': f"{rate:,}", 'qty': qty,
        'subtotal': f"{subtotal:,}", 'tax': f"{tax:,}", 'total': f"{total:,}",
        'monthly_rate': f"{monthly_rate:,}", 'duration': duration,
        'contract_total': f"{contract_total:,}", 'terms': terms,
        'payment_schedule': '50% on start, 50% on completion',
        'po_number': po_number, 'account_no': account_no, 'routing_no': routing_no,
        'matter_id': matter_id, 'sig1': sig1, 'sig2': sig2,
        'name': name, 'category': category, 'purpose': purpose,
        'suspicious_amount': f"{suspicious_amount:,}",
        'legal_boilerplate': legal_boilerplate,
        'gibberish_text': _gibberish(rng, gibberish_words),
        'foreign_text': foreign_text, 'translation_note': translation_note,
        'service_desc': f"{item_desc} services for {rng.randint(3, 24)} months",
    }
    return template.format(**vars_map)


# ---------------------------------------------------------------------------
# Document generator
# ---------------------------------------------------------------------------

class DocumentGenerator:
    def __init__(
        self,
        seed: int,
        sim_run_id: str,
        profile: Dict[str, Any],
        schema_source: str = 'static',
        policy_context: Optional[PolicyContext] = None,
        policy_client: Optional[PolicyRuntimeClient] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.seed = seed
        self.sim_run_id = sim_run_id
        self.rng = random.Random(seed)
        self.profile = profile
        self.schema_source = (schema_source or 'static').strip().lower()
        self.policy_context = policy_context
        self.policy_client = policy_client
        self.logger = logger or logging.getLogger('doc_generator')
        self._policy_fallback_logged = False

    def _weighted_choice(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        weights = [float(item.get('weight', 0)) for item in items]
        return self.rng.choices(items, weights=weights, k=1)[0]

    def _build_action_params(self, scenario: Dict[str, Any], doc_type: str) -> Dict[str, Any]:
        """Build action/signal fields matching what agent.py sends to the HITL/RLHF endpoints."""
        name = scenario['name']
        has_missing = name in ('incomplete_form', 'missing_signature')
        missing_count = 3 if has_missing else 0
        is_ambiguous = name in ('gibberish', 'ambiguous_legal', 'foreign_language', 'suspicious_amount')
        return {
            'doc_type': doc_type,
            'missing_fields_count': missing_count,
            'missing_fields': 'supervisor_signature, cost_center, approval_code' if has_missing else '',
            'doc_type_unrecognized': doc_type == 'other',
            'has_missing_fields': has_missing,
            'agent_flagged_ambiguous': is_ambiguous,
            'tool_name': self.policy_client.tool_name if self.policy_client else 'document',
            'action_type': self.policy_client.action_name if self.policy_client else 'triage',
            'resource_type': 'document',
            'feature_schema_version': (
                self.policy_context.feature_schema_version if self.policy_context else 'triage_v1'
            ),
        }

    def _build_policy_params(self, all_params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.policy_context:
            return all_params
        keys = list(dict.fromkeys(
            self.policy_context.action_fields_allowlist + self.policy_context.signal_keys_allowlist
        ))
        if not keys:
            return all_params
        return {k: all_params[k] for k in keys if k in all_params}

    def _build_static_criteria(self, scenario: Dict[str, Any], doc_type: str) -> List[Dict[str, Any]]:
        name = scenario['name']
        has_missing = name in ('incomplete_form', 'missing_signature')
        is_ambiguous = name in ('gibberish', 'ambiguous_legal', 'foreign_language', 'suspicious_amount')
        is_suspicious = name == 'suspicious_amount'
        return [
            {
                'criterion_id': 'c_doc_type_recognized',
                'criterion_text': 'Document type is recognized',
                'evaluator_type': 'compute',
                'hardness': 'hard',
                'status': 'not_met' if doc_type == 'other' else 'met',
                'signals': {'doc_type': doc_type, 'doc_type_unrecognized': doc_type == 'other'},
            },
            {
                'criterion_id': 'c_no_missing_fields',
                'criterion_text': 'Document has no missing required fields',
                'evaluator_type': 'compute',
                'hardness': 'hard',
                'status': 'not_met' if has_missing else 'met',
                'signals': {'has_missing_fields': has_missing},
            },
            {
                'criterion_id': 'c_not_ambiguous',
                'criterion_text': 'Document content is not ambiguous or unclassifiable',
                'evaluator_type': 'llm',
                'hardness': 'soft',
                'status': 'not_met' if is_ambiguous else 'met',
                'signals': {'agent_flagged_ambiguous': is_ambiguous},
            },
            {
                'criterion_id': 'c_amount_not_suspicious',
                'criterion_text': 'Document amount is not anomalously large or suspicious',
                'evaluator_type': 'compute',
                'hardness': 'hard',
                'status': 'not_met' if is_suspicious else 'met',
                'signals': {'suspicious_amount': is_suspicious},
            },
        ]

    def _policy_criteria_and_decision(
        self, scenario: Dict[str, Any], doc_type: str
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        if self.schema_source != 'policy' or not self.policy_client:
            return None, None

        all_params = self._build_action_params(scenario, doc_type)
        eval_params = self._build_policy_params(all_params)
        eval_result = self.policy_client.evaluate(eval_params)

        if not isinstance(eval_result, dict):
            if not self._policy_fallback_logged:
                self.logger.warning('policy mode enabled but evaluator unavailable; using static criteria fallback')
                self._policy_fallback_logged = True
            return None, None

        criteria_rows = eval_result.get('criteria_results')
        if not isinstance(criteria_rows, list):
            criteria_rows = []

        criteria: List[Dict[str, Any]] = []
        for item in criteria_rows:
            if not isinstance(item, dict):
                continue
            criterion_id = str(item.get('criterion_id') or '')
            meta = self.policy_context.criteria_meta.get(criterion_id) if self.policy_context else None
            text = str(item.get('text') or '') or str((meta or {}).get('text') or '') or criterion_id
            input_keys = (meta or {}).get('input_keys') if isinstance((meta or {}).get('input_keys'), list) else []
            full_params = self._build_action_params(scenario, doc_type)
            signals: Dict[str, Any] = {k: full_params[k] for k in input_keys if k in full_params}
            criteria.append({
                'criterion_id': criterion_id,
                'criterion_text': text,
                'evaluator_type': str(item.get('evaluator_type') or 'compute'),
                'hardness': str(item.get('hardness') or 'soft'),
                'status': str(item.get('status') or 'not_met'),
                'signals': signals,
            })

        decision_raw = str(eval_result.get('decision') or '').strip().lower()
        if decision_raw in ('allow', 'approve'):
            decision_hint = 'approve'
        elif decision_raw in ('reject', 'deny', 'blocked', 'block'):
            decision_hint = 'reject'
        else:
            decision_hint = None
        return criteria, decision_hint

    def generate(self, count: int) -> List[DocumentRecord]:
        scenarios: List[Dict[str, Any]] = self.profile['scenarios']
        templates: Dict[str, str] = self.profile['templates']
        rows: List[DocumentRecord] = []

        for idx in range(1, count + 1):
            scenario = self._weighted_choice(scenarios)
            doc_type = DOC_TYPE_MAP.get(scenario['name'], 'other')
            doc_text = _render_template(templates[scenario['template']], self.rng, self.profile)

            policy_criteria, policy_resolution = self._policy_criteria_and_decision(scenario, doc_type)
            static_resolution = scenario.get('expected_human_resolution') or 'approve'
            expected_resolution = policy_resolution or static_resolution
            criteria = policy_criteria or self._build_static_criteria(scenario, doc_type)

            rows.append(DocumentRecord(
                doc_id=f"doc-{idx:04d}",
                scenario=scenario['name'],
                doc_type=doc_type,
                expected_human_resolution=expected_resolution,
                criteria_json=json.dumps(criteria),
                feature_schema_version=(
                    self.policy_context.feature_schema_version if self.policy_context else 'triage_v1'
                ),
                sim_run_id=self.sim_run_id,
                seed=self.seed,
                sequence_no=idx,
                doc_text=doc_text,
            ))

        return rows


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(records: List[DocumentRecord], output_path: str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError('no records to write')
    fieldnames = list(asdict(records[0]).keys())
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))
    return out


def _load_profile(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('doc_generator')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic document triage CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=50)
    parser.add_argument('--sim-run-id', default='sim-run-001')
    parser.add_argument('--output', default='simulator/docs.csv')
    parser.add_argument('--profile', default=str(PROFILE_PATH))
    parser.add_argument('--log-file', default='simulator/out/generator.log')
    parser.add_argument(
        '--schema-source',
        choices=['static', 'policy'],
        default='static',
        help='static: built-in criteria; policy: fetch live criteria from AMP (default: static)',
    )
    parser.add_argument('--amp-backend-url', default=os.environ.get('BACKEND_URL', ''))
    parser.add_argument('--amp-api-key', default=os.environ.get('AMP_API_KEY', ''))
    parser.add_argument('--org-id', default=os.environ.get('AMP_ORG_ID', ''))
    parser.add_argument('--agent-name', default=os.environ.get('AGENT_NAME', ''))
    parser.add_argument('--tool-name', default='document')
    parser.add_argument('--action-name', default='triage')
    parser.add_argument('--policy-timeout-sec', type=float, default=20.0)
    args = parser.parse_args()

    logger = _setup_logger(args.log_file)
    profile = _load_profile(args.profile)

    manifest = {
        'event': 'generator_start',
        'seed': args.seed,
        'count': args.count,
        'sim_run_id': args.sim_run_id,
        'output': args.output,
        'schema_source': args.schema_source,
        'agent_name': args.agent_name,
        'profile': args.profile,
    }
    print(json.dumps(manifest))
    logger.info('generator_start %s', json.dumps(manifest))

    policy_context: Optional[PolicyContext] = None
    policy_client: Optional[PolicyRuntimeClient] = None

    if args.schema_source == 'policy':
        if not args.amp_backend_url or not args.amp_api_key or not args.agent_name:
            raise SystemExit('--schema-source policy requires --amp-backend-url, --amp-api-key, and --agent-name')
        policy_client = PolicyRuntimeClient(
            amp_backend_url=args.amp_backend_url,
            amp_api_key=args.amp_api_key,
            org_id=args.org_id,
            agent_name=args.agent_name,
            tool_name=args.tool_name,
            action_name=args.action_name,
            timeout_sec=args.policy_timeout_sec,
            logger=logger,
        )
        policy_context = policy_client.fetch_context()
        if policy_context:
            print(json.dumps({
                'event': 'policy_context_loaded',
                'policy_id': policy_context.policy_id,
                'policy_version': policy_context.policy_version,
                'feature_schema_version': policy_context.feature_schema_version,
                'criteria_count': len(policy_context.criteria_meta),
            }))
        else:
            print(json.dumps({'event': 'policy_context_unavailable', 'fallback': 'static'}))

    generator = DocumentGenerator(
        seed=args.seed,
        sim_run_id=args.sim_run_id,
        profile=profile,
        schema_source=args.schema_source,
        policy_context=policy_context,
        policy_client=policy_client,
        logger=logger,
    )

    records = generator.generate(args.count)
    out_path = write_csv(records, args.output)

    from collections import Counter
    dist = Counter(r.scenario for r in records)
    resolution_dist = Counter(r.expected_human_resolution for r in records)

    summary = {
        'event': 'generator_done',
        'count': len(records),
        'output': str(out_path),
        'schema_source': args.schema_source,
        'scenario_distribution': dict(sorted(dist.items(), key=lambda x: -x[1])),
        'resolution_distribution': dict(resolution_dist),
    }
    print(json.dumps(summary))
    logger.info('generator_done %s', json.dumps(summary))


if __name__ == '__main__':
    main()
