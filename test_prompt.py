import httpx
import sys
import json

payload = {
  "model": "qwen-plus",
  "messages": [
    {
      "role": "user",
      "content": "从金华到杭州自驾怎么走"
    }
  ],
  "stream": True
}
try:
    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", "http://127.0.0.1:8000/api/v1/chat", json=payload, headers={"User-Agent": "Apifox/1.0.0"}) as response:
            for line in response.iter_lines():
                if line:
                    print(line)
                    sys.stdout.flush()
except Exception as e:
    print(f"Error: {e}")
