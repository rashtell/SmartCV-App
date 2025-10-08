import os
from typing import Tuple

import anthropic
import openai
import requests

from config import CLAUDE_ANTHROPIC, GPT_OPENAI, OLLAMA_LOCAL


def call_claude(system_prompt: str, user_prompt: str, config: dict) -> Tuple[bool, str]:
    api_key = config.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return False, "Anthropic API key not configured"
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Note: anthropic client API varies — adapt if needed
        message = client.messages.create(
            model=config.get("claude_model") or "claude-sonnet-4-20250514",
            max_tokens=config.get("max_tokens", 4000),
            temperature=config.get("temperature", 0.7),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        # If message.content is a list of ContentBlock, join their string representations
        if isinstance(message.content, list):
            content_str = "".join([str(block) for block in message.content])
        else:
            content_str = str(message.content)
        return True, content_str
    except Exception as e:
        return False, f"Error calling Claude: {e}"


def call_openai(system_prompt: str, user_prompt: str, config: dict) -> Tuple[bool, str]:
    api_key = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return False, "OpenAI API key not configured"
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config.get("openai_model") or "gpt-4-turbo-preview",
            max_tokens=config.get("max_tokens", 4000),
            temperature=config.get("temperature", 0.7),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return True, response.choices[0].message.content or ""
    except Exception as e:
        return False, f"Error calling OpenAI: {e}"


def call_ollama(system_prompt: str, user_prompt: str, config: dict) -> Tuple[bool, str]:
    try:
        url = f"{config.get('ollama_url')}/api/generate"
        payload = {
            "model": config.get("ollama_model"),
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "stream": False,
            "options": {"temperature": config.get("temperature", 0.7)},
        }
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return True, data.get("response", "")
    except Exception as e:
        return False, f"Error calling Ollama: {e}"


def generate_with_llm(
    system_prompt: str, user_prompt: str, provider: str, config: dict
) -> str:
    provider = provider or OLLAMA_LOCAL
    if provider == CLAUDE_ANTHROPIC:
        ok, res = call_claude(system_prompt, user_prompt, config)
    elif provider == GPT_OPENAI:
        ok, res = call_openai(system_prompt, user_prompt, config)
    elif provider == OLLAMA_LOCAL:
        ok, res = call_ollama(system_prompt, user_prompt, config)
    else:
        return "Error: Unknown model choice"
    if not ok:
        return f"❌ {res}"
    return res


def get_ollama_models(ollama_url: str = "http://localhost:11434") -> tuple:
    try:
        url = f"{ollama_url}/api/tags"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [m["name"] for m in models], f"✅ Found {len(models)} models"
    except requests.exceptions.ConnectionError:
        return ["Connection Error"], "❌ Cannot connect to Ollama"
    except Exception as e:
        return ["Error"], f"❌ Error fetching models: {e}"
