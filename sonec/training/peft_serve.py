#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for PEFT / Unsloth / Axolotl adapters.

Requires: pip install 'sonec[train-cuda]' (transformers peft torch)
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing PEFT serve deps. pip install 'sonec[train-cuda]'"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    if device == "cpu":
        model.to(device)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *a: object) -> None:  # noqa: A003
            return

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                self._json(
                    200,
                    {"object": "list", "data": [{"id": args.model, "object": "model"}]},
                )
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._json(404, {"error": "not_found"})
                return
            length = int(self.headers.get("Content-Length") or "0")
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            messages = data.get("messages") or []
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            max_new = int(data.get("max_tokens") or 512)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            )
            self._json(
                200,
                {
                    "id": "sonec-peft",
                    "object": "chat.completion",
                    "model": args.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

    print(
        json.dumps(
            {
                "serving": f"http://{args.host}:{args.port}/v1",
                "model": args.model,
                "adapter": args.adapter,
                "device": device,
            },
            indent=2,
        )
    )
    # Mount under /v1 by accepting paths that end with the route.
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
