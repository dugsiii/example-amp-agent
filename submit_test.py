import requests

payload = {
    "invoice_total": 2160,
    "document": """INVOICE #INV-2042
Vendor: Acme Consulting LLC
Bill To: Wayne Industries
Invoice Date: 2026-04-29
Due Date: 2026-05-13
Payment Method: Bank Transfer

Line Items:
1. Strategy Consulting - 10 hrs @ $150/hr = $1,500
2. Research Report - flat fee = $500

Subtotal: $2,000
Tax (8%): $160
Total: $2,160 USD"""
}

r = requests.post("http://localhost:6000/submit", json=payload)
print(r.json())
