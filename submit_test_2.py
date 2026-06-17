import requests

# Case 2: Multiple hard failures
# - missing vendor_name and line items (c_no_missing_fields FAIL, c_has_line_items FAIL)
# - due date 2026-04-01 — 28 days overdue (c_not_overdue FAIL)
# - amount $12,500 > $10k (c_amount_reasonable FAIL)
# - cash payment

payload = {
    "invoice_total": 12500,
    "document": """INVOICE
Bill To: Wayne Industries
Invoice Date: 2026-03-01
Due Date: 2026-04-01
Payment Method: Cash

Total: $12,500"""
}

r = requests.post("http://localhost:6000/submit", json=payload)
print(r.json())
