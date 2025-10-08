def extract_job_info_from_description(job_description: str):
    if not job_description or len(job_description.strip()) < 30:
        return None, None, None, None
    job_role = ""
    company = ""
    position = ""
    skills = ""
    lines = job_description.split("\n")
    text_lower = job_description.lower()
    # company
    for line in lines[:10]:
        if "company:" in line.lower():
            company = line.split(":", 1)[1].strip()
            break
        elif " at " in line.lower() and len(line.split()) < 10:
            parts = line.split(" at ")
            if len(parts) > 1:
                company = parts[1].strip()
                break
    # position
    for line in lines[:5]:
        if any(
            k in line.lower() for k in ("position:", "role:", "title:", "job title:")
        ):
            position = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            job_role = position
            break
        elif (
            len(line.split()) <= 6
            and len(line) > 10
            and not line.startswith(("http", "www"))
        ):
            position = line.strip()
            job_role = position
            break
    # skills
    skill_keywords = [
        "required skills:",
        "skills:",
        "requirements:",
        "qualifications:",
        "technical skills:",
        "must have:",
    ]
    for keyword in skill_keywords:
        if keyword in text_lower:
            idx = text_lower.index(keyword)
            skills_section = job_description[idx : idx + 500]
            skills_lines = [
                ln.strip() for ln in skills_section.split("\n")[1:8] if ln.strip()
            ]
            skills = "\n".join(skills_lines)
            break
    return job_role, company, position, skills


def auto_fill_from_job_description(
    job_desc, current_job_role, current_company, current_position, current_skills
):
    if not job_desc:
        return current_job_role, current_company, current_position, current_skills
    extracted_role, extracted_company, extracted_position, extracted_skills = (
        extract_job_info_from_description(job_desc)
    )
    new_job_role = (
        current_job_role if current_job_role else (extracted_role or current_job_role)
    )
    new_company = (
        current_company if current_company else (extracted_company or current_company)
    )
    new_position = (
        current_position
        if current_position
        else (extracted_position or current_position)
    )
    if extracted_skills and not current_skills:
        new_skills = extracted_skills
    elif extracted_skills and current_skills:
        new_skills = (
            current_skills
            + "\n\n--- Extracted from Job Description ---\n"
            + extracted_skills
        )
    else:
        new_skills = current_skills
    return new_job_role, new_company, new_position, new_skills
