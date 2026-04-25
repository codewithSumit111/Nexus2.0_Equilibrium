import sys
import os
sys.path.append(os.path.join(os.getcwd(), "backend"))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_generate_sar():
    print("Testing /api/generate-sar...")
    payload = {
        "case_id": "DEBUG-CASE-999",
        "alert_data": {
            "alert_id": "AL-123",
            "alert_reason": "High volume of cash deposits followed by immediate transfers.",
            "customer_details": {
                "name": "John Doe",
                "account_id": "123456789"
            },
            "transactions": [
                {"date": "2024-01-01", "amount": 9000, "description": "Cash Deposit"},
                {"date": "2024-01-01", "amount": 8900, "description": "Transfer Out"}
            ]
        }
    }
    response = client.post("/api/generate-sar", json=payload)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response: {response.text}")
    else:
        print("Success!")

if __name__ == "__main__":
    os.environ["MOCK_LLM"] = "true"
    test_generate_sar()
