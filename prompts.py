PROMPTS = {

    "resume_analysis_v1": """
    Analyze this resume and provide:

    - Candidate Name
    - Skills
    - Experience
    - Education
    - Certifications

    Resume:
    {resume}
    """,

    "resume_analysis_v2": """
    You are a senior HR recruiter.

    Analyze the resume and provide:

    - Candidate Name
    - Technical Skills
    - Soft Skills
    - Years of Experience
    - Education
    - Certifications
    - Strengths
    - Weaknesses

    Resume:
    {resume}
    """,

    "evaluation_v1": """
    You are an HR recruiter.

    Job Role:
    {job_role}

    Resume Analysis:
    {analysis}

    Compare the candidate profile with the job role.

    Return:

    MATCH_PERCENTAGE: xx%

    VERDICT:
    SUITABLE

    OR

    MATCH_PERCENTAGE: xx%

    VERDICT:
    NOT SUITABLE
    """,

    "evaluation_v2": """
    You are an expert technical recruiter.

    Job Role:
    {job_role}

    Resume Analysis:
    {analysis}

    Evaluate:

    - Skill Match
    - Experience Match
    - Education Match

    Return:

    MATCH_PERCENTAGE: xx%

    REASON:
    explanation

    VERDICT:
    SUITABLE

    OR

    NOT SUITABLE
    """
}