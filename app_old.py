from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

import requests
from openai import OpenAI
from flask import Flask, jsonify, request, send_from_directory

APP_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(APP_ROOT), static_url_path="")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def load_dotenv_settings(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv_settings(APP_ROOT / ".env")

MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL", "")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "anthropic").strip().lower()

SYSTEM_PROMPT = """You are a computer vision model for a "Long Hair Gender Identification" system.
Carefully examine the uploaded photo and return ONLY a valid JSON object - no markdown, no extra text.

DECISION LOGIC (apply exactly):
- If estimatedAge >= 20 AND estimatedAge <= 30:
    predictedGender = hairLength === 'long' ? 'female' : 'male'
    ruleApplied = 'hair-based'
- Else (age < 20 OR age > 30):
    predictedGender = actualGender
    ruleApplied = 'actual-gender'

JSON schema:
{
  "estimatedAge": <integer>,
  "ageRange": "<string, e.g. '22-26'>",
  "hairLength": "<'long' or 'short'>",
  "hairLengthDetail": "<short descriptive phrase about the hair>",
  "actualGender": "<'male' or 'female'>",
  "actualGenderConfidence": <0-100 integer>,
  "predictedGender": "<computed as above>",
  "ruleApplied": "<'hair-based' or 'actual-gender'>",
  "ageGroupLabel": "<'In 20-30 range' or 'Outside 20-30 range'>",
  "overallConfidence": <0-100 integer>,
  "reasoning": "<2-3 sentence plain English explanation of the decision>"
}"""


def extract_text_content(content: object) -> str:
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if isinstance(content, str):
        return content
    return ""


def parse_model_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def build_openai_vision_messages(mime_type: str, image_base64: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyse this person's photo and return the JSON result as instructed.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}",
                    },
                },
            ],
        }
    ]


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index() -> object:
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health() -> object:
    return jsonify(status="ok")


@app.route("/api/analyse", methods=["POST", "OPTIONS"])
def analyse() -> object:
    if request.method == "OPTIONS":
        return jsonify(ok=True)

    payload = request.get_json(silent=True) or {}
    mime_type = str(payload.get("mimeType", "")).strip()
    image_base64 = str(payload.get("imageBase64", "")).strip()

    api_key = ""
    if MODEL_PROVIDER == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    if not api_key:
        if MODEL_PROVIDER == "openai":
            return jsonify(error="Set OPENAI_API_KEY in the backend environment."), 500
        return jsonify(error="Set ANTHROPIC_API_KEY in the backend environment."), 500
    if MODEL_PROVIDER == "openai" and not os.environ.get("OPENAI_MODEL", "").strip():
        return jsonify(error="Set OPENAI_MODEL in the backend environment when MODEL_PROVIDER=openai."), 500
    if not mime_type or not image_base64:
        return jsonify(error="Image data is missing."), 400

    try:
        base64.b64decode(image_base64, validate=True)
    except Exception:
        return jsonify(error="Image data is not valid base64."), 400

    if MODEL_PROVIDER == "openai":
        model_to_use = OPENAI_MODEL_NAME or os.environ.get("OPENAI_MODEL", "") or MODEL_NAME
        if model_to_use and not model_to_use.startswith("nvidia/"):
            model_to_use = f"nvidia/{model_to_use}"
        
        api_base_url = os.environ.get("OPENAI_API_URL", OPENAI_CHAT_URL).strip() or OPENAI_CHAT_URL
        # Remove /chat/completions suffix if present (OpenAI client adds it)
        if api_base_url.endswith("/chat/completions"):
            api_base_url = api_base_url.rsplit("/chat/completions", 1)[0]
        
        print(f"[OpenAI-compatible] Base URL: {api_base_url}, Model: {model_to_use}")
        provider_name = "OpenAI-compatible"
    else:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        body = {
            "model": MODEL_NAME,
            "max_tokens": 1000,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyse this person's photo and return the JSON result as instructed.",
                        },
                    ],
                }
            ],
        }
        api_url = ANTHROPIC_URL
        provider_name = "Anthropic"

    try:
        if MODEL_PROVIDER == "openai":
            try:
                client = OpenAI(base_url=api_base_url, api_key=api_key)
                response_obj = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *build_openai_vision_messages(mime_type, image_base64),
                    ],
                    temperature=0,
                    max_tokens=1000,
                )
                raw_text = extract_text_content(response_obj.choices[0].message.content)
            except Exception as exc:
                return jsonify(error=f"Failed to reach {provider_name}: {str(exc)}"), 502
        else:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
            }
            body = {
                "model": MODEL_NAME,
                "max_tokens": 1000,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": image_base64,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyse this person's photo and return the JSON result as instructed.",
                            },
                        ],
                    }
                ],
            }
            api_url = ANTHROPIC_URL
            response = requests.post(api_url, headers=headers, json=body, timeout=90)
            
            raw_response_text = ""
            try:
                upstream = response.json()
            except ValueError:
                upstream = None
                raw_response_text = response.text

            if not response.ok:
                error_message = None
                if isinstance(upstream, dict):
                    # Anthropic-style error
                    if "error" in upstream and isinstance(upstream["error"], dict):
                        error_message = upstream["error"].get("message")
                    else:
                        error_message = upstream.get("message")
                fallback = raw_response_text or (response.text if response is not None else "")
                return (
                    jsonify(
                        error=error_message or f"API error: {response.status_code}",
                        upstream_text=(fallback[:1000] + "..." if len(fallback) > 1000 else fallback),
                    ),
                    response.status_code,
                )
            
            raw_text = extract_text_content(upstream.get("content"))
    except requests.RequestException as exc:
        return jsonify(error=f"Failed to reach {provider_name}: {exc}"), 502
    
    if not raw_text:
        return jsonify(error=f"{provider_name} response did not include a text result."), 502

    try:
        result = parse_model_json(raw_text)
    except json.JSONDecodeError:
        return jsonify(error="Could not parse the model JSON response."), 502

    return jsonify(result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
