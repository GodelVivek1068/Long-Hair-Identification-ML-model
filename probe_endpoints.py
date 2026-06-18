import os
import requests

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL')
OPENAI_API_URL = os.environ.get('OPENAI_API_URL', 'https://integrate.api.nvidia.com/v1/chat/completions')

candidates = [
    OPENAI_API_URL,
    'https://integrate.api.nvidia.com/v1/completions',
    f'https://integrate.api.nvidia.com/v1/engines/{OPENAI_MODEL}/completions' if OPENAI_MODEL else None,
    f'https://integrate.api.nvidia.com/v1/engines/{OPENAI_MODEL}/chat/completions' if OPENAI_MODEL else None,
    f'https://integrate.api.nvidia.com/v1/chat/completions',
]

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}' if OPENAI_API_KEY else '',
    'x-api-key': OPENAI_API_KEY if OPENAI_API_KEY else '',
}

payload = {
    'model': OPENAI_MODEL,
    'messages': [{'role': 'system', 'content': 'Say hello.'}],
}

for url in [u for u in candidates if u]:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print('URL:', url)
        print('STATUS:', r.status_code)
        print('BODY:', r.text[:1000])
    except Exception as e:
        print('URL:', url)
        print('ERR', e)
    print('-' * 60)
