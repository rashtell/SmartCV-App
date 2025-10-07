import json
import os
import re
from datetime import datetime

import anthropic
import gradio as gr
import openai
import requests
from bs4 import BeautifulSoup
from fpdf import FPDF

# Configuration file
CONFIG_FILE = "config.json"
MEMORY_FILE = "conversation_history.json"

# Default configuration
DEFAULT_CONFIG = {
    "default_model": "claude",
    "claude_model": "claude-sonnet-4-20250514",
    "openai_model": "gpt-4-turbo-preview",
    "ollama_model": "llama2",
    "ollama_url": "http://localhost:11434",
    "anthropic_api_key": "",
    "openai_api_key": "",
    "max_tokens": 4000,
    "temperature": 0.7
}

# System prompts
CV_SYSTEM_PROMPT = """
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

COVER_LETTER_SYSTEM_PROMPT = """
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


def load_config():
    """Load configuration from file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Merge with defaults to handle new keys
            return {**DEFAULT_CONFIG, **config}
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def load_memory():
    """Load conversation history from file"""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_memory(history):
    """Save conversation history to file"""
    with open(MEMORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def scrape_linkedin(url):
    """Scrape basic information from LinkedIn profile (public data only)"""
    try:
        # Enhanced headers to mimic real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        info = {
            'name': '',
            'headline': '',
            'summary': '',
            'experience': '',
            'education': '',
            'skills': ''
        }
        
        # Extract from page title
        title = soup.find('title')
        if title and title.text:
            parts = title.text.split('-')[0].strip()
            info['name'] = parts.split('|')[0].strip()
        
        # Extract meta tags
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content = meta_desc.get('content', '')
            info['headline'] = str(content) if content is not None else ""
            
        # Try to extract from Open Graph tags
        og_title = soup.find('meta', property='og:title')
        if og_title:
            og_content = og_title.get('content', '')
            # Handle if og_content is a list (AttributeValueList) or None
            if isinstance(og_content, list):
                og_content = og_content[0] if og_content else ''
            if og_content:
                info['name'] = str(og_content).split('-')[0].strip()
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            og_desc_content = og_desc.get('content', '')
            if isinstance(og_desc_content, list):
                og_desc_content = og_desc_content[0] if og_desc_content else ''
            info['headline'] = str(og_desc_content) if og_desc_content is not None else ''
        
        # Extract any visible text content
        body_text = soup.get_text(separator=' ', strip=True)
        
        # Try to extract education and experience keywords
        if 'university' in body_text.lower() or 'college' in body_text.lower():
            # Extract education context
            lines = body_text.split('.')
            for line in lines:
                if any(word in line.lower() for word in ['university', 'college', 'bachelor', 'master', 'phd']):
                    info['education'] += line.strip() + '\n'
        
        # Look for JSON-LD structured data
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_ld_scripts:
            try:
                if script.string is not None:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        if data.get('@type') == 'Person':
                            info['name'] = data.get('name', info['name'])
                            info['headline'] = data.get('jobTitle', info['headline'])
            except:
                pass
        
        if not info['name'] and not info['headline']:
            return None, "⚠️ LinkedIn scraping blocked or profile not public. LinkedIn limits automated access. Try using 'Generic Profile' mode or manually paste the profile text."
        
        return info, "✅ Successfully scraped LinkedIn profile (limited public data available)"
    
    except requests.exceptions.RequestException as e:
        return None, f"❌ Error: Could not access LinkedIn. LinkedIn blocks automated scraping. Try: 1) Copy/paste profile text manually, or 2) Use a public portfolio/resume URL instead. Details: {str(e)}"
    except Exception as e:
        return None, f"❌ Error scraping LinkedIn: {str(e)}"


def scrape_generic_profile(url):
    """Scrape information from generic profile pages or portfolio sites"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract text content
        text_content = soup.get_text(separator='\n', strip=True)
        
        # Try to find email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text_content)
        
        # Try to find phone
        phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
        phones = re.findall(phone_pattern, text_content)
        
        # Try to extract name from title or h1
        name = ''
        title_tag = soup.find('title')
        if title_tag:
            name = title_tag.text.split('|')[0].split('-')[0].strip()
        
        h1 = soup.find('h1')
        if h1 and len(h1.text.strip()) < 50:
            name = h1.text.strip()
        
        # Extract meta description as summary
        summary = ''
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            summary = meta_desc.get('content', '')
        
        info = {
            'name': name,
            'email': emails[0] if emails else '',
            'phone': phones[0] if phones else '',
            'headline': summary[:200] if summary else '',
            'summary': text_content[:500],
            'content': text_content[:2000]
        }
        
        return info, "✅ Successfully scraped profile information"
    
    except Exception as e:
        return None, f"❌ Error scraping profile: {str(e)}"


def call_claude(system_prompt, user_prompt, config):
    """Call Claude API"""
    api_key = config.get('anthropic_api_key') or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: Anthropic API key not configured"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=config['claude_model'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content
    except Exception as e:
        return f"Error calling Claude: {str(e)}"


def call_openai(system_prompt, user_prompt, config):
    """Call OpenAI API"""
    api_key = config.get('openai_api_key') or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "Error: OpenAI API key not configured"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=config['openai_model'],
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI: {str(e)}"


def call_ollama(system_prompt, user_prompt, config):
    """Call Ollama API"""
    try:
        url = f"{config['ollama_url']}/api/generate"
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        payload = {
            "model": config['ollama_model'],
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": config['temperature']
            }
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"


def get_ollama_models(ollama_url):
    """Fetch available Ollama models"""
    try:
        url = f"{ollama_url}/api/tags"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        models = response.json().get('models', [])
        model_names = [model['name'] for model in models]
        if not model_names:
            return ["No models found"], "⚠️ No Ollama models installed. Run 'ollama pull <model>' to install."
        return model_names, f"✅ Found {len(model_names)} Ollama models"
    except requests.exceptions.ConnectionError:
        return ["Connection Error"], "❌ Cannot connect to Ollama. Make sure Ollama is running (ollama serve)"
    except Exception as e:
        return ["Error"], f"❌ Error fetching models: {str(e)}"


def update_ollama_model_list(ollama_url):
    """Update Ollama model dropdown"""
    models, status = get_ollama_models(ollama_url)
    return gr.Dropdown(choices=models, value=models[0] if models else None), status


def generate_with_llm(system_prompt, user_prompt, model_choice, config):
    """Route to appropriate LLM based on choice"""
    if model_choice == "Claude (Anthropic)":
        return call_claude(system_prompt, user_prompt, config)
    elif model_choice == "GPT (OpenAI)":
        return call_openai(system_prompt, user_prompt, config)
    elif model_choice == "Ollama (Local)":
        return call_ollama(system_prompt, user_prompt, config)
    else:
        return "Error: Unknown model choice"


def generate_cv(name, email, phone, job_role, summary, education, experience, 
                skills, certifications, job_description, model_choice):
    """Generate CV using selected LLM"""
    
    config = load_config()
    
    user_prompt = f"""
Please create a professional CV with the following information:

**Personal Information:**
- Name: {name}
- Email: {email}
- Phone: {phone}

**Target Role:** {job_role}

**Professional Summary:** {summary}

**Education:**
{education}

**Work Experience:**
{experience}

**Skills:**
{skills}

**Certifications/Awards:**
{certifications}
"""
    
    if job_description:
        user_prompt += f"\n**Job Description to tailor towards:**\n{job_description}"
    
    cv_content = generate_with_llm(CV_SYSTEM_PROMPT, user_prompt, model_choice, config)
    
    # Save to memory
    history = load_memory()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "type": "CV",
        "model": model_choice,
        "inputs": {
            "name": name,
            "email": email,
            "phone": phone,
            "job_role": job_role
        },
        "output": cv_content
    })
    save_memory(history)
    
    return cv_content


def generate_cover_letter(name, email, phone, company, position, 
                          key_achievements, motivation, job_description, model_choice):
    """Generate Cover Letter using selected LLM"""
    
    config = load_config()
    
    user_prompt = f"""
Please create a professional cover letter with the following information:

**Applicant Information:**
- Name: {name}
- Email: {email}
- Phone: {phone}

**Target Position:** {position} at {company}

**Key Achievements to Highlight:**
{key_achievements}

**Motivation/Why this role:**
{motivation}
"""
    
    if job_description:
        user_prompt += f"\n**Job Description:**\n{job_description}"
    
    cover_letter_content = generate_with_llm(COVER_LETTER_SYSTEM_PROMPT, user_prompt, 
                                             model_choice, config)
    
    # Save to memory
    history = load_memory()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "type": "Cover Letter",
        "model": model_choice,
        "inputs": {
            "name": name,
            "company": company,
            "position": position
        },
        "output": cover_letter_content
    })
    save_memory(history)
    
    return cover_letter_content


def export_to_pdf(content, filename, doc_type="CV"):
    """Export content to PDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=11)
        
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('**') and line.endswith('**'):
                pdf.set_font("Helvetica", 'B', 12)
                line = line.strip('*')
                pdf.multi_cell(0, 6, line.encode('latin-1', 'replace').decode('latin-1'))
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('# '):
                pdf.set_font("Helvetica", 'B', 14)
                pdf.multi_cell(0, 8, line[2:].encode('latin-1', 'replace').decode('latin-1'))
                pdf.set_font("Helvetica", size=11)
            elif line.strip():
                pdf.multi_cell(0, 5, line.encode('latin-1', 'replace').decode('latin-1'))
            else:
                pdf.ln(3)
        
        pdf.output(filename)
        return filename
    except Exception as e:
        return f"Error creating PDF: {str(e)}"


def view_history():
    """View conversation history"""
    history = load_memory()
    if not history:
        return "No previous generations found."
    
    output = "## Previous Generations\n\n"
    for i, item in enumerate(reversed(history[-10:]), 1):
        output += f"### {i}. {item['type']} - {item['timestamp'][:10]}\n"
        output += f"**Model:** {item.get('model', 'N/A')}\n"
        output += f"**Details:** {item['inputs']}\n\n"
    
    return output


def save_configuration(default_model, claude_model, openai_model, ollama_model,
                       ollama_url, anthropic_key, openai_key, max_tokens, temperature):
    """Save configuration settings"""
    config = {
        "default_model": default_model,
        "claude_model": claude_model,
        "openai_model": openai_model,
        "ollama_model": ollama_model,
        "ollama_url": ollama_url,
        "anthropic_api_key": anthropic_key,
        "openai_api_key": openai_key,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    save_config(config)
    return "✅ Configuration saved successfully!"


def load_current_config():
    """Load current configuration for display"""
    config = load_config()
    return (
        config['default_model'],
        config['claude_model'],
        config['openai_model'],
        config['ollama_model'],
        config['ollama_url'],
        config['anthropic_api_key'],
        config['openai_api_key'],
        config['max_tokens'],
        config['temperature']
    )


def scrape_profile(url, scrape_type):
    """Scrape profile information"""
    if scrape_type == "LinkedIn (Limited)":
        info, message = scrape_linkedin(url)
    else:
        info, message = scrape_generic_profile(url)
    
    if info:
        return (
            info.get('name', ''),
            info.get('email', ''),
            info.get('phone', ''),
            info.get('headline', ''),
            info.get('summary', ''),
            info.get('education', ''),
            info.get('experience', ''),
            info.get('skills', ''),
            message
        )
    else:
        return ('', '', '', '', '', '', '', '', message)


def toggle_ollama_options(model_choice):
    """Show/hide Ollama model selector based on provider choice"""
    if model_choice == "Ollama (Local)":
        # Load current config to get Ollama URL
        config = load_config()
        models, status = get_ollama_models(config['ollama_url'])
        return (
            gr.update(visible=True),  # Show row
            gr.update(choices=models, value=models[0] if models and models[0] != "Connection Error" else None),  # Update dropdown
            gr.update(visible=True, value=status)  # Show status
        )
    else:
        return (
            gr.update(visible=False),  # Hide row
            gr.update(),  # Keep dropdown as is
            gr.update(visible=False)  # Hide status
        )


def refresh_ollama_models():
    """Refresh the list of available Ollama models"""
    config = load_config()
    models, status = get_ollama_models(config['ollama_url'])
    return gr.update(choices=models, value=models[0] if models and models[0] != "Connection Error" else None), status


def generate_cv_with_ollama(name, email, phone, job_role, summary, education, experience, 
                             skills, certifications, job_description, model_choice, ollama_model):
    """Generate CV with Ollama model selection"""
    config = load_config()
    
    # Override config with selected Ollama model if applicable
    if model_choice == "Ollama (Local)" and ollama_model:
        config['ollama_model'] = ollama_model
    
    user_prompt = f"""
Please create a professional CV with the following information:

**Personal Information:**
- Name: {name}
- Email: {email}
- Phone: {phone}

**Target Role:** {job_role}

**Professional Summary:** {summary}

**Education:**
{education}

**Work Experience:**
{experience}

**Skills:**
{skills}

**Certifications/Awards:**
{certifications}
"""
    
    if job_description:
        user_prompt += f"\n**Job Description to tailor towards:**\n{job_description}"
    
    cv_content = generate_with_llm(CV_SYSTEM_PROMPT, user_prompt, model_choice, config)
    
    # Save to memory
    history = load_memory()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "type": "CV",
        "model": model_choice,
        "specific_model": ollama_model if model_choice == "Ollama (Local)" else config.get(f"{model_choice.lower().split()[0]}_model"),
        "inputs": {
            "name": name,
            "email": email,
            "phone": phone,
            "job_role": job_role
        },
        "output": cv_content
    })
    save_memory(history)
    
    return cv_content


def generate_cover_letter_with_ollama(name, email, phone, company, position, 
                                       key_achievements, motivation, job_description, 
                                       model_choice, ollama_model):
    """Generate Cover Letter with Ollama model selection"""
    config = load_config()
    
    # Override config with selected Ollama model if applicable
    if model_choice == "Ollama (Local)" and ollama_model:
        config['ollama_model'] = ollama_model
    
    user_prompt = f"""
Please create a professional cover letter with the following information:

**Applicant Information:**
- Name: {name}
- Email: {email}
- Phone: {phone}

**Target Position:** {position} at {company}

**Key Achievements to Highlight:**
{key_achievements}

**Motivation/Why this role:**
{motivation}
"""
    
    if job_description:
        user_prompt += f"\n**Job Description:**\n{job_description}"
    
    cover_letter_content = generate_with_llm(COVER_LETTER_SYSTEM_PROMPT, user_prompt, 
                                             model_choice, config)
    
    # Save to memory
    history = load_memory()
    history.append({
        "timestamp": datetime.now().isoformat(),
        "type": "Cover Letter",
        "model": model_choice,
        "specific_model": ollama_model if model_choice == "Ollama (Local)" else config.get(f"{model_choice.lower().split()[0]}_model"),
        "inputs": {
            "name": name,
            "company": company,
            "position": position
        },
        "output": cover_letter_content
    })
    save_memory(history)
    
    return cover_letter_content


# Create Gradio Interface
with gr.Blocks(title="CV & Cover Letter Generator", theme="soft") as app:
    gr.Markdown("# 📄 Professional CV & Cover Letter Generator")
    gr.Markdown("Generate tailored CVs and cover letters using multiple LLM providers with memory storage.")
    
    # Shared state for personal information
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 👤 Personal Information (Shared)")
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=["Claude (Anthropic)", "GPT (OpenAI)", "Ollama (Local)"],
                value="Claude (Anthropic)",
                label="Select LLM Provider"
            )
    
    # Ollama model selector (conditional)
    with gr.Row(visible=False) as ollama_model_row:
        with gr.Column(scale=3):
            ollama_model_dropdown = gr.Dropdown(
                choices=["llama2"],
                value="llama2",
                label="Select Ollama Model",
                interactive=True
            )
        with gr.Column(scale=1):
            refresh_ollama_btn = gr.Button("🔄 Refresh Models", size="sm")
    
    ollama_status = gr.Textbox(label="Ollama Status", visible=False, interactive=False)
    
    # Web scraping section
    with gr.Accordion("🌐 Auto-fill from Web Profile", open=False):
        gr.Markdown("""
        **Note about LinkedIn**: LinkedIn actively blocks automated scraping. For best results:
        - Use 'Generic Profile' for personal websites, portfolios, or resume pages
        - For LinkedIn, manually copy and paste the profile information
        - Try using Google Docs, Notion, or personal website URLs instead
        """)
        with gr.Row():
            scrape_url = gr.Textbox(
                label="Profile URL", 
                placeholder="https://yourportfolio.com or https://linkedin.com/in/username"
            )
            scrape_type = gr.Radio(
                ["Generic Profile", "LinkedIn (Limited)"], 
                value="Generic Profile", 
                label="Profile Type"
            )
        scrape_btn = gr.Button("Scrape Profile", variant="secondary")
        scrape_status = gr.Textbox(label="Scrape Status", interactive=False)
    
    with gr.Row():
        name = gr.Textbox(label="Full Name", placeholder="John Doe")
        email = gr.Textbox(label="Email", placeholder="john.doe@email.com")
        phone = gr.Textbox(label="Phone", placeholder="+1234567890")
    
    job_role = gr.Textbox(label="Target Job Role", placeholder="Senior Software Engineer")
    summary = gr.Textbox(
        label="Professional Summary",
        placeholder="Experienced software engineer with 5+ years...",
        lines=3
    )
    education = gr.Textbox(
        label="Education",
        placeholder="Bachelor's in Computer Science, MIT, 2018-2022",
        lines=3
    )
    experience = gr.Textbox(
        label="Work Experience",
        placeholder="Software Engineer, Google, 2022-Present\n- Led development of...",
        lines=5
    )
    skills = gr.Textbox(
        label="Skills",
        placeholder="Python, JavaScript, React, Machine Learning",
        lines=2
    )
    certifications = gr.Textbox(
        label="Certifications & Awards",
        placeholder="AWS Certified Solutions Architect, Dean's List",
        lines=2
    )
    
    # Scraping functionality
    scrape_btn.click(
        fn=scrape_profile,
        inputs=[scrape_url, scrape_type],
        outputs=[name, email, phone, job_role, summary, education, experience, skills, scrape_status]
    )
    
    # Model choice change handler
    model_choice.change(
        fn=toggle_ollama_options,
        inputs=[model_choice],
        outputs=[ollama_model_row, ollama_model_dropdown, ollama_status]
    )
    
    # Refresh Ollama models button
    refresh_ollama_btn.click(
        fn=refresh_ollama_models,
        outputs=[ollama_model_dropdown, ollama_status]
    )
    
    with gr.Tabs():
        # CV Generator Tab
        with gr.Tab("📋 CV Generator"):
            cv_job_desc = gr.Textbox(
                label="Job Description (Optional - for tailoring)",
                placeholder="Paste the job description here...",
                lines=4
            )
            
            cv_generate_btn = gr.Button("Generate CV", variant="primary")
            cv_output = gr.Textbox(label="Generated CV", lines=20)
            
            with gr.Row():
                cv_export_btn = gr.Button("Export as PDF")
                cv_download = gr.File(label="Download CV")
            
            cv_generate_btn.click(
                fn=generate_cv_with_ollama,
                inputs=[name, email, phone, job_role, summary, education, experience, 
                       skills, certifications, cv_job_desc, model_choice, ollama_model_dropdown],
                outputs=cv_output
            )
            
            cv_export_btn.click(
                fn=lambda content: export_to_pdf(content, "cv_output.pdf", "CV"),
                inputs=cv_output,
                outputs=cv_download
            )
        
        # Cover Letter Generator Tab
        with gr.Tab("✉️ Cover Letter Generator"):
            with gr.Row():
                company = gr.Textbox(label="Company Name", placeholder="Google")
                position = gr.Textbox(label="Position", placeholder="Senior Software Engineer")
            
            key_achievements = gr.Textbox(
                label="Key Achievements to Highlight",
                placeholder="- Led team of 5 engineers\n- Increased efficiency by 40%",
                lines=4
            )
            motivation = gr.Textbox(
                label="Why This Role/Company?",
                placeholder="I am excited about this opportunity because...",
                lines=3
            )
            
            cl_job_desc = gr.Textbox(
                label="Job Description (Optional)",
                placeholder="Paste the job description here...",
                lines=4
            )
            
            cl_generate_btn = gr.Button("Generate Cover Letter", variant="primary")
            cl_output = gr.Textbox(label="Generated Cover Letter", lines=20)
            
            with gr.Row():
                cl_export_btn = gr.Button("Export as PDF")
                cl_download = gr.File(label="Download Cover Letter")
            
            cl_generate_btn.click(
                fn=generate_cover_letter_with_ollama,
                inputs=[name, email, phone, company, position, key_achievements, 
                       motivation, cl_job_desc, model_choice, ollama_model_dropdown],
                outputs=cl_output
            )
            
            cl_export_btn.click(
                fn=lambda content: export_to_pdf(content, "cover_letter_output.pdf", "Cover Letter"),
                inputs=cl_output,
                outputs=cl_download
            )
        
        # Configuration Tab
        with gr.Tab("⚙️ Configuration"):
            gr.Markdown("### Default Settings")
            
            with gr.Row():
                cfg_default_model = gr.Dropdown(
                    choices=["Claude (Anthropic)", "GPT (OpenAI)", "Ollama (Local)"],
                    label="Default Model"
                )
                cfg_max_tokens = gr.Slider(1000, 8000, step=100, label="Max Tokens")
                cfg_temperature = gr.Slider(0, 1, step=0.1, label="Temperature")
            
            gr.Markdown("### Model Configuration")
            
            with gr.Row():
                cfg_claude_model = gr.Textbox(label="Claude Model", placeholder="claude-sonnet-4-20250514")
                cfg_openai_model = gr.Textbox(label="OpenAI Model", placeholder="gpt-4-turbo-preview")
            
            with gr.Row():
                cfg_ollama_model = gr.Textbox(label="Ollama Default Model", placeholder="llama2")
                cfg_ollama_url = gr.Textbox(label="Ollama URL", placeholder="http://localhost:11434")
            
            ollama_test_btn = gr.Button("Test Ollama Connection", variant="secondary")
            ollama_test_status = gr.Textbox(label="Ollama Test Status", interactive=False)
            
            gr.Markdown("### API Keys")
            cfg_anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")
            cfg_openai_key = gr.Textbox(label="OpenAI API Key", type="password")
            
            with gr.Row():
                cfg_load_btn = gr.Button("Load Current Config", variant="secondary")
                cfg_save_btn = gr.Button("Save Configuration", variant="primary")
            
            cfg_status = gr.Textbox(label="Status", interactive=False)
            
            cfg_load_btn.click(
                fn=load_current_config,
                outputs=[cfg_default_model, cfg_claude_model, cfg_openai_model, 
                        cfg_ollama_model, cfg_ollama_url, cfg_anthropic_key, 
                        cfg_openai_key, cfg_max_tokens, cfg_temperature]
            )
            
            cfg_save_btn.click(
                fn=save_configuration,
                inputs=[cfg_default_model, cfg_claude_model, cfg_openai_model,
                       cfg_ollama_model, cfg_ollama_url, cfg_anthropic_key,
                       cfg_openai_key, cfg_max_tokens, cfg_temperature],
                outputs=cfg_status
            )
            
            ollama_test_btn.click(
                fn=lambda url: get_ollama_models(url)[1],
                inputs=[cfg_ollama_url],
                outputs=ollama_test_status
            )
        
        # History Tab
        with gr.Tab("📚 History"):
            gr.Markdown("### Previous Generations")
            history_refresh_btn = gr.Button("Refresh History")
            history_output = gr.Markdown()
            
            history_refresh_btn.click(
                fn=view_history,
                outputs=history_output
            )

if __name__ == "__main__":
    app.launch()