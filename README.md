# CV & Cover Letter Generator (Refactored)

Run:

```bash
pip install -r requirements.txt
python run.py
```

This refactor splits functionality into modules:

- `config.py` manages config file loading/saving
- `storage.py` persists generation history
- `scraping.py` contains scraping utilities (LinkedIn and generic)
- `llm_clients.py` wraps calls to Anthropic (Claude), OpenAI, and Ollama
- `pdf_utils.py` exports generated content to PDF
- `utils.py` misc helpers (job description parsing, autofill)
- `app.py` builds the Gradio UI

Environment:

- Use `.env` or set environment variables `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` if not saved via the UI
