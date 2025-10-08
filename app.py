import gradio as gr

from config import (
    CLAUDE_ANTHROPIC,
    CLAUDE_MODELS,
    COVER_LETTER_SYSTEM_PROMPT,
    CV_SYSTEM_PROMPT,
    GPT_OPENAI,
    OLLAMA_LOCAL,
    OPENAI_MODELS,
    load_config,
    save_config,
)
from llm_clients import generate_with_llm, get_ollama_models
from pdf_utils import export_to_pdf
from scraping import scrape_generic_profile, scrape_linkedin
from storage import append_memory, load_memory
from utils import auto_fill_from_job_description


def build_ui():
    config = load_config()
    with gr.Blocks(title="Smart CV & Cover Letter Generator", theme="soft") as app:
        gr.Markdown("# 📄 Smart CV & Cover Letter Generator")

        # Model selection section
        with gr.Row():
            with gr.Column(scale=2):
                # Web scraping section
                with gr.Accordion("🌐 Auto-fill from Web Profile", open=False):
                    gr.Markdown(
                        "LinkedIn often blocks scraping. Use generic portfolio URLs or paste text manually."
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            scrape_status = gr.Textbox(
                                show_label=False,
                                interactive=False,
                                placeholder="Status",
                            )
                        gr.Column(scale=2)

                    with gr.Row():
                        scrape_url = gr.Textbox(label="Profile URL")

                        scrape_type = gr.Radio(
                            label="Type",
                            choices=["Generic Profile", "LinkedIn (Limited)"],
                            value="Generic Profile",
                        )

                    with gr.Row():
                        scrape_btn = gr.Button("Scrape Profile", size="md")
                        gr.Column(scale=4)

            with gr.Column(scale=1):

                model_choice = gr.Dropdown(
                    choices=[CLAUDE_ANTHROPIC, GPT_OPENAI, OLLAMA_LOCAL],
                    value=OLLAMA_LOCAL,
                    label="Select LLM Provider",
                )

                with gr.Row(
                    visible=(config.get("default_model") == config.get("ollama_model"))
                ) as ollama_row:
                    with gr.Column(scale=1):
                        ollama_dropdown = gr.Dropdown(
                            choices=get_ollama_models(config["ollama_url"])[0],
                            value=config.get("ollama_model"),
                            label="Select Ollama Model",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button(
                            "🔄 Refresh Models", size="sm", variant="primary"
                        )

                    with gr.Column(scale=1):
                        ollama_status = gr.Textbox(
                            show_label=False, placeholder="Ollama Status"
                        )

        def toggle_ollama_visibility(choice):
            return gr.update(visible=(choice == OLLAMA_LOCAL))

        def get_ollama_models_wrapper():
            models, status = get_ollama_models(config["ollama_url"])

            return gr.update(choices=models), status

        model_choice.change(
            fn=toggle_ollama_visibility, inputs=[model_choice], outputs=[ollama_row]
        )

        refresh_btn.click(
            fn=get_ollama_models_wrapper,
            inputs=[],
            outputs=[ollama_dropdown, ollama_status],
        )

        # Personal info fields
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 👤 Personal Information")

        with gr.Row():
            name = gr.Textbox(label="Full Name")
            email = gr.Textbox(label="Email")

        with gr.Row():
            phone = gr.Textbox(label="Phone")
            job_role = gr.Textbox(label="Target Job Role")

        summary = gr.Textbox(label="Professional Summary", lines=3)

        education = gr.Textbox(label="Education", lines=3)

        experience = gr.Textbox(label="Work Experience", lines=6)

        skills = gr.Textbox(label="Skills", lines=2)

        certifications = gr.Textbox(label="Certifications & Awards", lines=2)

        job_desc = gr.Textbox(label="Job Description (Optional)", lines=6)

        with gr.Row():
            with gr.Column(scale=1):
                auto_fill_btn = gr.Button(
                    "🔄 Auto-fill from Job Description", size="md", variant="primary"
                )
            gr.Column(scale=4)

        # Scrape action
        def scrape_profile(url, scrape_type):
            if not url:
                return ("", "", "", "", "", "", "", "", "⚠️ Provide a URL")
            if scrape_type == "LinkedIn (Limited)":
                info, msg = scrape_linkedin(url)
            else:
                info, msg = scrape_generic_profile(url)
            if not info:
                return ("", "", "", "", "", "", "", "", msg)
            return (
                info.get("name", ""),
                info.get("email", ""),
                info.get("phone", ""),
                info.get("headline", ""),
                info.get("summary", ""),
                info.get("education", ""),
                info.get("experience", ""),
                info.get("skills", ""),
                msg,
            )

        scrape_btn.click(
            fn=scrape_profile,
            inputs=[scrape_url, scrape_type],
            outputs=[
                name,
                email,
                phone,
                job_role,
                summary,
                education,
                experience,
                skills,
                scrape_status,
            ],
        )

        # Tabs
        with gr.Tabs():
            with gr.Tab("📋 CV Generator"):
                with gr.Row():
                    with gr.Column(scale=1):
                        cv_generate_btn = gr.Button(
                            "Generate CV", size="lg", variant="primary"
                        )
                    gr.Column(scale=4)

                cv_output = gr.Textbox(label="Generated CV", lines=20)

                with gr.Row():
                    with gr.Column(scale=1):
                        cv_export_btn = gr.Button(
                            "Export as PDF", size="md", variant="secondary"
                        )
                    gr.Column(scale=4)

                cv_download = gr.File()

                def generate_cv_local(
                    name_,
                    email_,
                    phone_,
                    job_role_,
                    summary_,
                    education_,
                    experience_,
                    skills_,
                    certifications_,
                    job_desc_,
                    provider_,
                    ollama_model_,
                ):
                    cfg = load_config()
                    if provider_ == OLLAMA_LOCAL and ollama_model_:
                        cfg["ollama_model"] = ollama_model_
                    user_prompt = f"""
                                    Create a professional CV with the following information:\n
                                    **Personal Information:**\n
                                    - Name: {name_}\n
                                    - Email: {email_}\n
                                    - Phone: {phone_}\n\n
                                    **Target Role:** {job_role_}\n\n
                                    **Professional Summary:** {summary_}\n\n
                                    **Education:**\n{education_}\n\n
                                    **Work Experience:**\n{experience_}\n\n
                                    **Skills:**\n{skills_}\n\n
                                    **Certifications/Awards:**\n{certifications_}\n
                                    """
                    if job_desc_:
                        user_prompt += (
                            f"\n**Job Description to tailor towards:**\n{job_desc_}"
                        )
                    result = generate_with_llm(
                        CV_SYSTEM_PROMPT, user_prompt, provider_, cfg
                    )
                    append_memory(
                        {
                            "type": "CV",
                            "model": provider_,
                            "inputs": {"name": name_, "job_role": job_role_},
                            "output": result,
                        }
                    )
                    return result

                def export_cv_pdf(content):
                    if not content or not content.strip():
                        return None
                    fname = export_to_pdf(content, "cv_output.pdf")
                    return fname

                cv_generate_btn.click(
                    fn=generate_cv_local,
                    inputs=[
                        name,
                        email,
                        phone,
                        job_role,
                        summary,
                        education,
                        experience,
                        skills,
                        certifications,
                        job_desc,
                        model_choice,
                        ollama_dropdown,
                    ],
                    outputs=cv_output,
                )

                cv_export_btn.click(
                    fn=export_cv_pdf, inputs=cv_output, outputs=cv_download
                )

            with gr.Tab("✉️ Cover Letter Generator"):
                company = gr.Textbox(label="Company Name")

                position = gr.Textbox(label="Position")

                key_achievements = gr.Textbox(label="Key Achievements", lines=4)

                motivation = gr.Textbox(label="Why This Role/Company?", lines=3)

                with gr.Row():
                    with gr.Column(scale=1):
                        cl_generate_btn = gr.Button(
                            "Generate Cover Letter", size="lg", variant="primary"
                        )
                    gr.Column(scale=4)

                cl_output = gr.Textbox(label="Generated Cover Letter", lines=20)

                with gr.Row():
                    with gr.Column(scale=1):
                        cl_export_btn = gr.Button("Export as PDF")
                    gr.Column(scale=4)

                cl_download = gr.File()

                def generate_cover_local(
                    name_,
                    email_,
                    phone_,
                    company_,
                    position_,
                    achievements_,
                    motivation_,
                    job_desc_,
                    provider_,
                    ollama_model_,
                ):
                    cfg = load_config()
                    if provider_ == OLLAMA_LOCAL and ollama_model_:
                        cfg["ollama_model"] = ollama_model_
                    user_prompt = f"""
                                    Please create a professional cover letter with the following information:\n
                                    **Applicant Information:**\n
                                    - Name: {name_}\n
                                    - Email: {email_}\n
                                    - Phone: {phone_}\n\n
                                    **Target Position:** {position_} at {company_}\n\n
                                    **Key Achievements to Highlight:
                                    **\n{achievements_}\n\n
                                    **Motivation/Why this role:**\n{motivation_}\n
                                    """
                    if job_desc_:
                        user_prompt += f"\n**Job Description:**\n{job_desc_}"
                    result = generate_with_llm(
                        COVER_LETTER_SYSTEM_PROMPT, user_prompt, provider_, cfg
                    )
                    append_memory(
                        {
                            "type": "Cover Letter",
                            "model": provider_,
                            "inputs": {
                                "name": name_,
                                "company": company_,
                                "position": position_,
                            },
                            "output": result,
                        }
                    )
                    return result

                def export_cl_pdf(content):
                    if not content or not content.strip():
                        return None
                    fname = export_to_pdf(content, "cover_letter_output.pdf")
                    return fname

                cl_generate_btn.click(
                    fn=generate_cover_local,
                    inputs=[
                        name,
                        email,
                        phone,
                        company,
                        position,
                        key_achievements,
                        motivation,
                        job_desc,
                        model_choice,
                        ollama_dropdown,
                    ],
                    outputs=cl_output,
                )

                cl_export_btn.click(
                    fn=export_cl_pdf, inputs=cl_output, outputs=cl_download
                )

            auto_fill_btn.click(
                fn=auto_fill_from_job_description,
                inputs=[job_desc, job_role, company, position, skills],
                outputs=[job_role, company, position, skills],
            )

        # Configuration Tab
        with gr.Tab("⚙️ Configuration"):
            with gr.Row():
                gr.Column(scale=3)
                cfg_status = gr.Textbox(
                    show_label=False, placeholder="Status", interactive=False
                )

            with gr.Row():
                cfg_default_model = gr.Dropdown(
                    choices=[CLAUDE_ANTHROPIC, GPT_OPENAI, OLLAMA_LOCAL],
                    label="Default Model",
                    value=OLLAMA_LOCAL,
                )

                cfg_claude_model = gr.Dropdown(
                    label="Claude Model",
                    choices=CLAUDE_MODELS,
                    value=config.get("claude_model"),
                )

            with gr.Row():
                cfg_openai_model = gr.Dropdown(
                    label="OpenAI Model",
                    choices=OPENAI_MODELS,
                    value=config.get("openai_model"),
                )

                cfg_ollama_models_dropdown = gr.Dropdown(
                    label="Ollama Model",
                    choices=get_ollama_models(config["ollama_url"])[0],
                    value=config.get("ollama_model"),
                )

            with gr.Row():
                cfg_max_tokens = gr.Slider(
                    1000,
                    8000,
                    step=100,
                    label="Max Tokens",
                    value=config.get("max_tokens", 4000),
                )

                cfg_temperature = gr.Slider(
                    0,
                    1,
                    step=0.1,
                    label="Temperature",
                    value=config.get("temperature", 0.7),
                )

            cfg_ollama_url = gr.Textbox(
                label="Ollama URL",
                value=config.get("ollama_url", "http://localhost:11434"),
            )

            cfg_anthropic_key = gr.Textbox(label="Anthropic API Key", type="password")

            cfg_openai_key = gr.Textbox(label="OpenAI API Key", type="password")

            with gr.Row():
                with gr.Column(scale=1):
                    cfg_save_btn = gr.Button(
                        "Save Configuration", size="lg", variant="secondary"
                    )
                gr.Column(scale=4)

            def save_cfg(
                default_model,
                claude_model,
                openai_model,
                ollama_model,
                ollama_url,
                anthropic_key,
                openai_key,
                max_tokens,
                temperature,
            ):
                cfg = {
                    "default_model": default_model,
                    "claude_model": claude_model,
                    "openai_model": openai_model,
                    "ollama_model": ollama_model,
                    "ollama_url": ollama_url,
                    "anthropic_api_key": anthropic_key,
                    "openai_api_key": openai_key,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                save_config(cfg)
                return "✅ Configuration saved successfully!"

            cfg_save_btn.click(
                fn=save_cfg,
                inputs=[
                    cfg_default_model,
                    cfg_claude_model,
                    cfg_openai_model,
                    cfg_ollama_models_dropdown,
                    cfg_ollama_url,
                    cfg_anthropic_key,
                    cfg_openai_key,
                    cfg_max_tokens,
                    cfg_temperature,
                ],
                outputs=cfg_status,
            )

        # History Tab
        with gr.Tab("📚 History"):
            history_out = gr.Markdown()

            def view_history():
                h = load_memory()
                if not h:
                    return "No history found."
                out = "### Previous Generations\n\n"
                for i, item in enumerate(reversed(h[-20:]), 1):
                    out += f"- **{item.get('type', 'N/A')}** | {item.get('timestamp', '')} | {item.get('model', '')}\\n"
                return out

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Button("Refresh History").click(
                        fn=view_history, outputs=history_out
                    )
                gr.Column(scale=4)

    return app


if __name__ == "__main__":
    build_ui().launch()
