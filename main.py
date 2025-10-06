import gradio as gr
import json
import os
from datetime import datetime
from fpdf import FPDF
import anthropic

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Memory storage file
MEMORY_FILE = "conversation_history.json"

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


def generate_cv(name, email, phone, job_role, summary, education, experience, 
                skills, certifications, job_description):
    """Generate CV using Claude API"""
    
    # Build user prompt
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
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            system=CV_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        cv_content = message.content[0].text
        
        # Save to memory
        history = load_memory()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "CV",
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
    
    except Exception as e:
        return f"Error generating CV: {str(e)}"


def generate_cover_letter(name, email, phone, company, position, 
                          key_achievements, motivation, job_description):
    """Generate Cover Letter using Claude API"""
    
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
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            system=COVER_LETTER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        
        cover_letter_content = message.content[0].text
        
        # Save to memory
        history = load_memory()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "Cover Letter",
            "inputs": {
                "name": name,
                "company": company,
                "position": position
            },
            "output": cover_letter_content
        })
        save_memory(history)
        
        return cover_letter_content
    
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"


def export_to_pdf(content, filename, doc_type="CV"):
    """Export content to PDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Use a Unicode-compatible font
        pdf.set_font("Helvetica", size=11)
        
        # Clean content and split into lines
        lines = content.split('\n')
        
        for line in lines:
            # Handle special formatting
            if line.startswith('**') and line.endswith('**'):
                # Bold headers
                pdf.set_font("Helvetica", 'B', 12)
                line = line.strip('*')
                pdf.multi_cell(0, 6, line.encode('latin-1', 'replace').decode('latin-1'))
                pdf.set_font("Helvetica", size=11)
            elif line.startswith('# '):
                # Main headers
                pdf.set_font("Helvetica", 'B', 14)
                pdf.multi_cell(0, 8, line[2:].encode('latin-1', 'replace').decode('latin-1'))
                pdf.set_font("Helvetica", size=11)
            elif line.strip():
                # Regular text
                pdf.multi_cell(0, 5, line.encode('latin-1', 'replace').decode('latin-1'))
            else:
                # Empty line
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
    for i, item in enumerate(reversed(history[-10:]), 1):  # Show last 10
        output += f"### {i}. {item['type']} - {item['timestamp'][:10]}\n"
        output += f"**Details:** {item['inputs']}\n\n"
    
    return output


# Create Gradio Interface
with gr.Blocks(title="CV & Cover Letter Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📄 Professional CV & Cover Letter Generator")
    gr.Markdown("Generate tailored CVs and cover letters powered by AI with memory storage.")
    
    with gr.Tabs():
        # CV Generator Tab
        with gr.Tab("📋 CV Generator"):
            gr.Markdown("### Personal Information")
            with gr.Row():
                cv_name = gr.Textbox(label="Full Name", placeholder="John Doe")
                cv_email = gr.Textbox(label="Email", placeholder="john.doe@email.com")
                cv_phone = gr.Textbox(label="Phone", placeholder="+1234567890")
            
            cv_job_role = gr.Textbox(label="Target Job Role", placeholder="Senior Software Engineer")
            cv_summary = gr.Textbox(
                label="Professional Summary",
                placeholder="Experienced software engineer with 5+ years...",
                lines=3
            )
            
            gr.Markdown("### Background")
            cv_education = gr.Textbox(
                label="Education",
                placeholder="Bachelor's in Computer Science, MIT, 2018-2022",
                lines=3
            )
            cv_experience = gr.Textbox(
                label="Work Experience",
                placeholder="Software Engineer, Google, 2022-Present\n- Led development of...",
                lines=5
            )
            cv_skills = gr.Textbox(
                label="Skills",
                placeholder="Python, JavaScript, React, Machine Learning",
                lines=2
            )
            cv_certifications = gr.Textbox(
                label="Certifications & Awards",
                placeholder="AWS Certified Solutions Architect, Dean's List",
                lines=2
            )
            
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
                fn=generate_cv,
                inputs=[cv_name, cv_email, cv_phone, cv_job_role, cv_summary,
                       cv_education, cv_experience, cv_skills, cv_certifications, cv_job_desc],
                outputs=cv_output
            )
            
            cv_export_btn.click(
                fn=lambda content: export_to_pdf(content, "cv_output.pdf", "CV"),
                inputs=cv_output,
                outputs=cv_download
            )
        
        # Cover Letter Generator Tab
        with gr.Tab("✉️ Cover Letter Generator"):
            gr.Markdown("### Personal Information")
            with gr.Row():
                cl_name = gr.Textbox(label="Full Name", placeholder="John Doe")
                cl_email = gr.Textbox(label="Email", placeholder="john.doe@email.com")
                cl_phone = gr.Textbox(label="Phone", placeholder="+1234567890")
            
            with gr.Row():
                cl_company = gr.Textbox(label="Company Name", placeholder="Google")
                cl_position = gr.Textbox(label="Position", placeholder="Senior Software Engineer")
            
            cl_achievements = gr.Textbox(
                label="Key Achievements to Highlight",
                placeholder="- Led team of 5 engineers\n- Increased efficiency by 40%",
                lines=4
            )
            cl_motivation = gr.Textbox(
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
                fn=generate_cover_letter,
                inputs=[cl_name, cl_email, cl_phone, cl_company, cl_position,
                       cl_achievements, cl_motivation, cl_job_desc],
                outputs=cl_output
            )
            
            cl_export_btn.click(
                fn=lambda content: export_to_pdf(content, "cover_letter_output.pdf", "Cover Letter"),
                inputs=cl_output,
                outputs=cl_download
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