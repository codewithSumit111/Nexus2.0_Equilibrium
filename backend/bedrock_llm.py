"""
NEXUS 2.0 — Amazon Bedrock LLM Client
Thin wrapper around boto3 for Claude 3.5 Sonnet calls via Bedrock.
"""

import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))

# Lazy-initialised client (created on first call)
_client = None


def _get_client():
    """Lazy-init the Bedrock Runtime client."""
    global _client
    if _client is None:
        # Check if we are in mock mode
        if os.getenv("MOCK_LLM", "false").lower() == "true":
            return "MOCK"
            
        _client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
        )
    return _client


def invoke_claude(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int | None = None,
) -> str:
    """
    Send a single-turn message to Claude via Bedrock and return the text response.
    Supports MOCK_LLM=true for testing without credentials.
    """
    client = _get_client()

    if client == "MOCK":
        # Simulate responses based on the system prompt keywords
        if "EXACTLY ONE of the following typologies" in system_prompt:
            return "Elder Financial Exploitation"
        if "senior AML investigator" in system_prompt:
            return "1. **Pattern Summary**: The case involves a possible elder exploitation scenario. \n2. **Key Indicators**: Large cash withdrawals and new POA. \n3. **Transaction Flow**: ACCT-001 to ACCT-002. \n4. **Risk Assessment**: High. \n5. **Regulatory Triggers**: BSA SAR filing required."
        if "expert SAR (Suspicious Activity Report) narrative" in system_prompt:
            return "This SAR narrative describes suspicious activity involving [PERSON_1] and [PERSON_2]. [PERSON_1] is a long-term customer who recently added [PERSON_2] as Power of Attorney. Following this, several large withdrawals and transfers to [PERSON_2]'s account were observed. This behavior is consistent with elder financial exploitation."
        if "Compliance Officer reviewing a SAR" in system_prompt:
            return "SCORE: 9\nFEEDBACK: Excellent narrative that clearly connects the suspicious patterns with the transaction data. Masking is correctly applied."
        return "Mock response: System prompt not recognized."

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens or MAX_TOKENS,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    response = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )

    result = json.loads(response["body"].read())

    # Claude returns content as a list of blocks
    text_blocks = [
        block["text"]
        for block in result.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(text_blocks)
