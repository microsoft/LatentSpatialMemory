from __future__ import annotations

from typing import Any


def response_to_text(response: Any) -> str:
    """Extract plain text content from a LiteLLM response."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def print_token_consumption(response: Any, output_text: str) -> None:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    print(f"[Qwen3VL] message: {output_text}")
    if usage is None:
        print("[Qwen3VL] token usage: unavailable")
        return

    print(
        "[Qwen3VL] token usage: "
        f"prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}"
    )
