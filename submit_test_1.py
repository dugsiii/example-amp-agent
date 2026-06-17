import requests

# Case 1: All clear — passes every hard and soft criterion
# - doc_type recognized (invoice)
# - no missing fields
# - has line items
# - not overdue (due in 16 days)
# - amount $3,200 (< $10k)
# - USD, bank transfer

payload = {
    "invoice_total": 3200,
    "document": """INVOICE #INV-5501
Vendor: Meridian Solutions LLC
Bill To: Wayne Industries
Invoice Date: 2026-04-25
Due Date: 2026-05-15
Payment Method: Bank Transfer

Line Items:
1. UX Audit - 8 hrs @ $200/hr = $1,600
2. Accessibility Report - flat fee = $1,600

Subtotal: $3,200
Tax: $0
Total: $3,200 USD"""
}

r = requests.post("http://localhost:6000/submit", json=payload)
print(r.json())
