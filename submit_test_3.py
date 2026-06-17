import requests

# Case 3: Soft failures only — all hard criteria pass
# - doc_type recognized, no missing fields, has line items, not overdue, amount < $10k
# - due date 2026-05-01 — 2 days out (c_payment_not_urgent FAIL)
# - wire transfer (c_standard_payment FAIL)
# - EUR currency (c_usd_currency FAIL)

payload = {
    "invoice_total": 8500,
    "document": """INVOICE #INV-7892
Vendor: Global Tech GmbH
Bill To: Wayne Industries
Invoice Date: 2026-04-29
Due Date: 2026-05-01
Payment Method: Wire Transfer

Line Items:
1. Software Licensing Q2 - $5,000
2. Support Services - $3,500

Subtotal: EUR 8,500
Tax: $0
Total: EUR 8,500"""
}

r = requests.post("http://localhost:6000/submit", json=payload)
print(r.json())
