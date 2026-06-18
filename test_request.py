import requests
import json

payload = {
    "mimeType": "image/png",
    "imageBase64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
}
try:
    r = requests.post("http://127.0.0.1:5000/api/analyse", json=payload, timeout=60)
    print('STATUS', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print('RAW:', r.text)
except Exception as e:
    print('ERR', e)
