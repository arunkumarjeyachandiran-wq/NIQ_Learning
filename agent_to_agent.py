from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import pypdf
import re
from langsmith import trace, traceable
from langsmith import trace
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from prompts import PROMPTS
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING_V2 = os.getenv("LANGSMITH_TRACING_V2")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING_V2"] = LANGSMITH_TRACING_V2
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT


# LLM Setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)


# Read PDF
@traceable(name="Read PDF")
def read_pdf(file_path):
    reader = pypdf.PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text

# Agent 1 - Resume Analysis
@traceable(name="Resume Analysis Agent")
def analyze_resume(
        resume_text,
        prompt_version="resume_analysis_v1"
):

    prompt_template = PROMPTS[prompt_version]

    prompt = prompt_template.format(
        resume=resume_text
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content

# Agent 2 - Role Matching
@traceable(name="Evaluation Agent")
def evaluate_candidate(
        analysis,
        job_role,
        prompt_version="evaluation_v1"
):

    prompt_template = PROMPTS[prompt_version]

    prompt = prompt_template.format(
        job_role=job_role,
        analysis=analysis
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return response.content

# Extract Email From Resume
def extract_email(resume_text):

    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"

    emails = re.findall(pattern, resume_text)

    if emails:
        return emails[0]

    return None

# Send Email
@traceable(name="Send Email")
def send_email(candidate_email, verdict, job_role):

    sender_email = SENDER_EMAIL
    sender_password = SENDER_PASSWORD

    if "NOT SUITABLE" in verdict.upper():

        subject = "Application Status"

        body = f"""
Dear Candidate,

Thank you for applying for the {job_role} position.

After reviewing your profile, we regret to inform you that you have not been shortlisted for this role.

We appreciate your interest and wish you success in your future opportunities.

Best Regards,
HR Team
"""

    else:

        subject = "Congratulations - Shortlisted"

        body = f"""
Dear Candidate,

Congratulations!

Based on our evaluation, you have been shortlisted for the {job_role} position.

Our recruitment team will contact you regarding the next steps.

Best Regards,
HR Team
"""

    msg = MIMEMultipart()

    msg["From"] = sender_email
    msg["To"] = candidate_email
    msg["Subject"] = subject

    msg.attach(
        MIMEText(body, "plain")
    )

    try:

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587
        )

        server.starttls()

        server.login(
            sender_email,
            sender_password
        )

        server.send_message(msg)

        server.quit()

        print(f"Email sent successfully to {candidate_email}")

    except Exception as e:
        print("Email sending failed:", e)

# Agent 3 - Notification Agent
@traceable(name="Notification Agent")
def notification_agent(
        resume_text,
        evaluation_result,
        job_role
):

    candidate_email = extract_email(
        resume_text
    )

    if candidate_email:

        print(
            f"Candidate Email Found: {candidate_email}"
        )

        send_email(
            candidate_email,
            evaluation_result,
            job_role
        )

    else:

        print(
            "No email address found in resume."
        )

# Main Workflow
pdf_path = r"C:\Users\prasanna.j\Downloads\Prasanna_resume.pdf"

job_role = input(
    "Enter Job Role: "
)

with trace("Resume Screening Workflow"):

    resume_text = read_pdf(pdf_path)

    analysis = analyze_resume(
        resume_text,
        prompt_version="resume_analysis_v2"
    )

    result = evaluate_candidate(
        analysis,
        job_role,
        prompt_version="evaluation_v2"
    )

    notification_agent(
        resume_text,
        result,
        job_role
    )

print("Resume Analysis")
print(analysis)

print("Final Decision")
print(result)