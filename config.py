import json
import os
from pathlib import Path
from textwrap import dedent

CONFIG_FILE = Path("config.json")

# Available models for each provider
CLAUDE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-sonnet-3-5-20241022",
    "claude-sonnet-3-5-20240620",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

OPENAI_MODELS = [
    "gpt-4-turbo-preview",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4-32k",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "o1-preview",
    "o1-mini",
]

OLLAMA_MODELS = ["llama3.2:latest"]

CLAUDE_ANTHROPIC = "Claude (Anthropic)"
GPT_OPENAI = "GPT (OpenAI)"
OLLAMA_LOCAL = "Ollama (Local)"

DEFAULT_CONFIG = {
    "default_model": OLLAMA_MODELS[0],
    "claude_model": CLAUDE_MODELS[0],
    "openai_model": OPENAI_MODELS[0],
    "ollama_model": OLLAMA_MODELS[0],
    "ollama_url": "http://localhost:11434",
    "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
    "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
    "max_tokens": 4000,
    "temperature": 0.7,
}


CV_SYSTEM_PROMPT = dedent(
    """
Create a professional CV based on the following.
🔍 1. DECONSTRUCT
* **Core Intent**: Generate a professional CV.
* **Key Entities**: Candidate details (education, work experience, skills, etc.).
* **Context**: Intended for recruiters and hiring managers. Needs clarity, structure, and professional tone.
* **Output Requirements**: Well-formatted CV, tailored for professional use.
* **Missing Info**: Industry/role focus, job level (entry, mid, senior), preferred CV style (chronological, functional, hybrid).
🧩 2. DIAGNOSE
* **Clarity Gaps**: Without industry/job role, the CV may feel too generic.
* **Specificity Needs**: Must guide AI to highlight achievements, not just list duties.
* **Complexity Level**: Structured → requires decomposed sections for strong output.
⚙️ 3. DEVELOP
* **Technique Choice**: *Educational + Constraint-based* → structured sections, professional formatting.
* **AI Role**: Professional career coach + CV writing expert.
* **Context Layering**: Emphasize achievements, impact, and clarity for recruiters.
🚀 4. DELIVER (Optimized Prompt)
**Optimized Prompt:**
You are a professional career coach and expert CV writer. Your task is to create a polished, recruiter-ready CV.
**Inputs (the user will provide):**
* Full name & contact details
* Desired job role / industry
* Career objective or professional summary
* Education (degrees, institutions, years)
* Work experience (roles, companies, years, key achievements)
* Skills (technical & soft)
* Certifications, awards, or notable projects
**Output Requirements:**
* Format the CV with clear, professional headings (Profile, Education, Work Experience, Skills, etc.).
* Highlight measurable achievements, not just responsibilities.
* Keep tone formal and concise.
* Tailor phrasing toward the target job role.
* Ensure ATS-friendly formatting (no tables or complex formatting).
**Formatting:**
* Use bullet points for readability.
* Keep length within 1–2 pages depending on experience.
👉 **Implementation Guidance**:
* For **entry-level roles** → emphasize education, projects, and transferable skills.
* For **mid-level or senior roles** → highlight work achievements and leadership impact.
* You can reuse this prompt by simply swapping in new candidate details.
"""
)

COVER_LETTER_SYSTEM_PROMPT = dedent(
    """
You are an expert career advisor and professional writer specializing in cover letters. Your task is to create compelling, personalized cover letters that effectively communicate a candidate's value proposition.

**Your Responsibilities:**
* Craft a professional cover letter that connects the candidate's experience with the job requirements
* Open with a strong hook that captures attention
* Highlight 2-3 key achievements or qualifications that match the role
* Demonstrate genuine interest in the company and position
* Close with a confident call to action
* Maintain a professional yet personable tone
* Keep the letter concise (250-400 words)

**Structure:**
1. Header with contact information
2. Date and employer details
3. Strong opening paragraph
4. 1-2 body paragraphs showcasing relevant experience
5. Closing paragraph with call to action
6. Professional sign-off

**Tone:** Professional, confident, enthusiastic, and tailored to the specific role and company.
"""
)


def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            # Merge defaults to ensure newly added keys exist
            merged = {**DEFAULT_CONFIG, **data}
            return merged
        except Exception:
            # If corrupted, back up and return defaults
            backup = CONFIG_FILE.with_suffix(".bak.json")
            CONFIG_FILE.replace(backup)
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(cfg: dict):
    # minimal validation
    cfg_to_write = {**DEFAULT_CONFIG, **cfg}
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg_to_write, f, indent=2)
    return True
