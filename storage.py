import json
from datetime import datetime
from pathlib import Path

MEMORY_FILE = Path("conversation_history.json")


def load_memory():
    if MEMORY_FILE.exists():
        try:
            return json.loads(MEMORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_memory(history):
    try:
        MEMORY_FILE.write_text(json.dumps(history, indent=2))
        return True
    except Exception:
        return False


def append_memory(item: dict):
    history = load_memory()
    history.append({**item, "timestamp": datetime.now().isoformat()})
    save_memory(history)
