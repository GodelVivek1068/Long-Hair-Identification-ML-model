from __future__ import annotations

import base64
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from flask import Flask, jsonify, request, send_from_directory, Response
from openai import OpenAI

APP_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(APP_ROOT), static_url_path="")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))


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
    if not isinstance(content, (dict, list, str)):
        content_data = object_to_dict(content)
    else:
        content_data = {}
    if content_data:
        return extract_text_content(content_data)
    if isinstance(content, list):
        return "".join(extract_text_content(block) for block in content)
    if isinstance(content, dict):
        for key in ("text", "output_text", "content"):
            value = content.get(key)
            if isinstance(value, (str, list, dict)):
                text = extract_text_content(value)
                if text:
                    return text
        return ""
    if isinstance(content, str):
        return content
    return ""


def object_to_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {}


def extract_openai_response_text(response_obj: object) -> str:
    response = object_to_dict(response_obj)
    choices = response.get("choices", [])
    if not isinstance(choices, list):
        return ""

    for choice in choices:
        choice_data = object_to_dict(choice)
        message = object_to_dict(choice_data.get("message", {}))

        text = extract_text_content(message.get("content"))
        if text:
            return text

        text = extract_text_content(choice_data.get("text"))
        if text:
            return text

        # Some OpenAI-compatible providers expose non-standard text fields.
        for key in ("reasoning_content", "reasoning", "refusal"):
            text = extract_text_content(message.get(key))
            if text:
                return text

    return ""


def summarize_openai_response(response_obj: object) -> str:
    response = object_to_dict(response_obj)
    choices = response.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return "No choices were returned."

    choice = object_to_dict(choices[0])
    message = object_to_dict(choice.get("message", {}))
    details = {
        "finish_reason": choice.get("finish_reason"),
        "message_keys": sorted(message.keys()),
        "content": message.get("content"),
        "refusal": message.get("refusal"),
    }
    text = json.dumps(details, default=str)
    return text[:1000] + ("..." if len(text) > 1000 else "")


def parse_json_candidate(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


def iter_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None

    return candidates


def parse_model_json(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    parsed = parse_json_candidate(cleaned)
    if parsed is not None:
        return parsed

    for candidate in iter_json_object_candidates(cleaned):
        parsed = parse_json_candidate(candidate)
        if parsed is not None:
            return parsed

    raise json.JSONDecodeError("No valid JSON object found in model response.", cleaned, 0)


def openai_base_url() -> str:
    api_base_url = os.environ.get("OPENAI_API_URL", OPENAI_CHAT_URL).strip() or OPENAI_CHAT_URL
    if api_base_url.endswith("/chat/completions"):
        api_base_url = api_base_url.rsplit("/chat/completions", 1)[0]
    return api_base_url.rstrip("/")


def openai_error_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)

    try:
        body = response.json()
    except Exception:
        body = getattr(response, "text", "")

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if body.get("message"):
            return str(body["message"])

    body_text = str(body)
    return body_text[:1000] + ("..." if len(body_text) > 1000 else "")


def build_openai_vision_messages(mime_type: str, image_base64: str) -> list[dict[str, Any]]:
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
def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.get("/")
def index() -> Response:
    return jsonify(status="ok", service="long-hair-identification-backend", health="/health", analyse="/api/analyse")


@app.get("/health")
def health() -> Response:
    return jsonify(status="ok")


@app.route("/api/analyse", methods=["POST", "OPTIONS"])
def analyse() -> tuple[Response, int] | Response:
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

    raw_text = ""
    response_obj: object | None = None
    provider_name = "Unknown"

    try:
        if MODEL_PROVIDER == "openai":
            model_to_use = OPENAI_MODEL_NAME or os.environ.get("OPENAI_MODEL", "")
            api_base_url = openai_base_url()
            provider_name = "OpenAI-compatible"
            try:
                client = OpenAI(base_url=api_base_url, api_key=api_key)
                response_obj = client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
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
                        },
                    ],
                    temperature=0,
                    max_tokens=OPENAI_MAX_TOKENS,
                )
                raw_text = extract_openai_response_text(response_obj)
            except Exception as exc:
                return jsonify(error=f"Failed to reach {provider_name}: {openai_error_detail(exc)}"), 502
        else:
            provider_name = "Anthropic"
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
            response = requests.post(ANTHROPIC_URL, headers=headers, json=body, timeout=90)
            
            raw_response_text = ""
            try:
                upstream = response.json()
            except ValueError:
                upstream = None
                raw_response_text = response.text

            if not response.ok:
                error_message = None
                if isinstance(upstream, dict):
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
            
            if upstream:
                raw_text = extract_text_content(upstream.get("content"))
    except requests.RequestException as exc:
        return jsonify(error=f"Failed to reach {provider_name}: {exc}"), 502
    
    if not raw_text:
        detail = ""
        if MODEL_PROVIDER == "openai" and response_obj is not None:
            detail = summarize_openai_response(response_obj)
        return jsonify(error=f"{provider_name} response did not include a text result.", upstream_detail=detail), 502

    try:
        result = parse_model_json(raw_text)
    except json.JSONDecodeError:
        snippet = raw_text.strip()
        snippet = snippet[:1000] + ("..." if len(snippet) > 1000 else "")
        return jsonify(error="Could not parse the model JSON response.", model_text=snippet), 502

    return jsonify(result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)

