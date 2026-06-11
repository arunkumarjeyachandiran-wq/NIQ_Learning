"""
FastAPI server to display research reports in a neat format
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime
from urllib.parse import quote
import os
from dotenv import load_dotenv
from Agent_workflow import run_research_workflow
from llm_utils import get_llm, should_use_llm
# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Research Team Agent API",
    description="Multi-agent research workflow API with FastAPI",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class ResearchRequest(BaseModel):
    """Request model for research topic"""
    topic: str
    
    class Config:
        example = {"topic": "Compare YOLO v8 vs DETR for object detection"}


class Review(BaseModel):
    approved: bool
    confidence: int
    issues: list[str]
    suggestions: list[str]


class ResearchResponse(BaseModel):
    """Response model for research report"""
    topic: str
    research_notes: Any
    analysis: Any
    review: Review
    assets: dict[str, list[str]]
    report: dict[str, Any]
    pdf_path: str
    timestamp: str
    status: str = "success"


# In-memory storage for recent reports
reports_cache: Dict[str, Any] = {}


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page with API documentation and interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Team Agent API</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 900px;
                width: 100%;
                padding: 50px;
            }
            h1 {
                color: #333;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 40px;
                font-size: 16px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
            }
            input {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            button:active {
                transform: translateY(0);
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .api-info {
                background: #f5f5f5;
                padding: 30px;
                border-radius: 8px;
                margin-top: 40px;
            }
            .api-info h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 18px;
            }
            .endpoint {
                background: white;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .endpoint-method {
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 3px 10px;
                border-radius: 4px;
                font-weight: 600;
                margin-right: 10px;
            }
            .endpoint-path {
                color: #666;
                font-family: monospace;
                font-size: 14px;
            }
            .error {
                color: #d32f2f;
                margin-top: 15px;
                display: none;
                padding: 15px;
                background: #ffebee;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 Research Team Agent</h1>
            <p class="subtitle">Multi-agent orchestration with LangGraph & FastAPI</p>
            
            <form id="researchForm">
                <div class="form-group">
                    <label for="topic">Research Topic:</label>
                    <input 
                        type="text" 
                        id="topic" 
                        name="topic" 
                        placeholder="e.g., Compare YOLO v8 vs DETR for object detection"
                        required
                    />
                </div>
                <button type="submit">Generate Report</button>
                <div class="error" id="error"></div>
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Generating report... This may take a minute.</p>
                </div>
            </form>

            <div class="api-info">
                <h2>📚 API Endpoints</h2>
                <div class="endpoint">
                    <span class="endpoint-method">POST</span>
                    <span class="endpoint-path">/api/research</span>
                    <p style="margin-top: 8px; color: #666; font-size: 14px;">Submit a research topic and generate a report</p>
                </div>
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/api/report/&lt;topic&gt;</span>
                    <p style="margin-top: 8px; color: #666; font-size: 14px;">Get a previously generated report (HTML format)</p>
                </div>
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/api/report-json/&lt;topic&gt;</span>
                    <p style="margin-top: 8px; color: #666; font-size: 14px;">Get a previously generated report (JSON format)</p>
                </div>
                <div class="endpoint">
                    <span class="endpoint-method">GET</span>
                    <span class="endpoint-path">/docs</span>
                    <p style="margin-top: 8px; color: #666; font-size: 14px;">Interactive API documentation (Swagger UI)</p>
                </div>
            </div>
        </div>

        <script>
            /*
             * ResearchResponse schema:
             * topic: string
             * research_notes: any
             * analysis: any
             * review: { approved: boolean; confidence: number; issues: string[]; suggestions: string[] }
             * assets: { charts: string[]; tables: string[]; images: string[] }
             * report: {
             *   title: string;
             *   executive_summary: string;
             *   sections: string[];
             *   tables: { title: string; headers: string[]; rows: string[][] }[];
             *   charts: { title: string; chart_type: string; x_label: string; y_label: string; labels: string[]; values: number[] }[];
             *   images: { title: string; search_query: string; reason: string }[];
             *   conclusion: string;
             * }
             * pdf_path: string
             */
            document.getElementById('researchForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const topic = document.getElementById('topic').value;
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const submitButton = document.querySelector('button[type="submit"]');
                
                error.style.display = 'none';
                error.textContent = '';
                submitButton.disabled = true;
                loading.style.display = 'block';

                try {
                    const response = await fetch('/api/research', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ topic: topic })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }

                    const data = await response.json();
                    
                    // Redirect to report page
                    window.location.href = `/api/report/${encodeURIComponent(topic)}`;
                } catch (err) {
                    error.style.display = 'block';
                    error.textContent = `Error: ${err.message}`;
                } finally {
                    loading.style.display = 'none';
                    submitButton.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/api/research", response_model=ResearchResponse)
async def submit_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Submit a research topic and generate a report
    
    **Args:**
    - **topic**: The research topic to investigate
    
    **Returns:**
    - Topic, generated report, and timestamp
    """
    try:
        topic = request.topic.strip()
        if not topic:
            raise ValueError("Topic cannot be empty")
        
        print(f"Processing research topic: {topic}")
        result = run_research_workflow(topic)

        def serialize_value(value):
            if hasattr(value, "model_dump"):
                return value.model_dump()
            return value

        response_payload = {
            "topic": topic,
            "research_notes": serialize_value(result.get("research_notes", {})),
            "analysis": serialize_value(result.get("analysis", {})),
            "review": serialize_value(result.get("review", {
                "approved": False,
                "confidence": 0,
                "issues": [],
                "suggestions": []
            })),
            "assets": serialize_value(result.get("assets", {
                "charts": [],
                "tables": [],
                "images": []
            })),
            "report": serialize_value(result.get("report", {
                "title": "",
                "executive_summary": "",
                "sections": [],
                "tables": [],
                "charts": [],
                "images": [],
                "conclusion": ""
            })),
            "pdf_path": result.get("pdf_path", ""),
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }

        reports_cache[topic] = response_payload
        
        return response_payload
    
    except Exception as e:
        print(f"Error in research workflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating report: {str(e)}"
        )


@app.get("/api/report/{topic}", response_class=HTMLResponse)
async def get_report_html(topic: str):
    """
    Get a research report in neat HTML format
    
    **Args:**
    - **topic**: The research topic
    
    **Returns:**
    - Formatted HTML report
    """
    if topic not in reports_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found for topic: {topic}. Please submit the research first."
        )
    
    report_data = reports_cache[topic]
    timestamp = report_data["timestamp"]
    
    # Convert structured report payload to HTML
    html_report = format_report_as_html(report_data, topic, timestamp)
    
    return html_report


@app.get("/api/download-pdf/{topic}")
async def download_pdf(topic: str):
    """Download the PDF generated for this topic."""
    if topic not in reports_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found for topic: {topic}. Please submit the research first."
        )

    pdf_path = reports_cache[topic].get("pdf_path")
    if not pdf_path or not os.path.isfile(pdf_path):
        raise HTTPException(
            status_code=404,
            detail="PDF not available for this report."
        )

    return FileResponse(
        path=os.path.abspath(pdf_path),
        filename=os.path.basename(pdf_path),
        media_type="application/pdf"
    )


@app.get("/api/report-json/{topic}")
async def get_report_json(topic: str):
    """
    Get a research report in JSON format
    
    **Args:**
    - **topic**: The research topic
    
    **Returns:**
    - JSON formatted report
    """
    if topic not in reports_cache:
        raise HTTPException(
            status_code=404,
            detail=f"Report not found for topic: {topic}. Please submit the research first."
        )
    
    return JSONResponse(
        content=reports_cache[topic]
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    USE_LLM = os.getenv("USE_LLM", "true").lower() == "true"

    print("Tracing:", os.getenv("LANGSMITH_TRACING"))
    print("Project:", os.getenv("LANGSMITH_PROJECT"))
    print("Key Exists:", bool(os.getenv("LANGSMITH_API_KEY")))

    print(type(get_llm()))
    llm_status = "available" if USE_LLM else "disabled"
    return {
        "status": "healthy",
        "service": "Research Team Agent API",
        "timestamp": datetime.now().isoformat(),
        "llm_status": llm_status
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_report_as_html(report, topic: str, timestamp: str) -> str:
    """Convert report payload to formatted HTML."""
    import html
    import re

    def escape_text(text: str) -> str:
        return html.escape(str(text))

    def wrap_empty(message: str) -> str:
        return f'<div class="empty-placeholder"><p>{escape_text(message)}</p></div>'

    def format_inline(text: str) -> str:
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        return text

    def is_table_separator(line: str) -> bool:
        return bool(re.match(r'^\s*\|?\s*:?[-=]+:?(\s*\|\s*:?[-=]+:?)+\s*\|?\s*$', line))

    def render_table(header_line: str, rows: list[str]) -> str:
        header_cells = [format_inline(escape_text(cell.strip())) for cell in header_line.strip().strip('|').split('|')]
        row_cells = [
            [format_inline(escape_text(cell.strip())) for cell in row.strip().strip('|').split('|')]
            for row in rows
        ]
        header_html = ''.join(f'<th>{cell}</th>' for cell in header_cells)
        body_html = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in row_cells)
        return f'<div class="table-scroll"><table class="markdown-table"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table></div>'

    def render_markdown(raw_text: str) -> str:
        text = str(raw_text or '').rstrip('\n')
        if not text.strip():
            return wrap_empty('No analysis content is available.')

        escaped_lines = [escape_text(line) for line in text.splitlines()]
        html_lines = []
        list_stack: list[tuple[int, str]] = []
        paragraph_open = False
        in_code = False
        code_lines: list[str] = []
        i = 0

        def close_paragraph() -> None:
            nonlocal paragraph_open
            if paragraph_open:
                html_lines.append('</p>')
                paragraph_open = False

        def close_lists(min_indent: int = 0) -> None:
            nonlocal list_stack
            while list_stack and list_stack[-1][0] >= min_indent:
                _, tag = list_stack.pop()
                html_lines.append(f'</{tag}>')

        while i < len(escaped_lines):
            line = escaped_lines[i]
            raw_line = text.splitlines()[i]
            fence_match = re.match(r'^\s*```', raw_line)
            if fence_match:
                if in_code:
                    html_lines.append('<pre class="code-block"><code>' + escape_text('\n'.join(code_lines)) + '</code></pre>')
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
                i += 1
                continue

            if in_code:
                code_lines.append(raw_line)
                i += 1
                continue

            if i + 1 < len(escaped_lines) and '|' in line and is_table_separator(escaped_lines[i + 1]):
                close_paragraph()
                close_lists()
                rows = []
                j = i + 2
                while j < len(escaped_lines) and '|' in escaped_lines[j]:
                    rows.append(escaped_lines[j])
                    j += 1
                html_lines.append(render_table(line, rows))
                i = j
                continue

            heading_match = re.match(r'^(#{1,4})\s+(.*)$', raw_line)
            if heading_match:
                close_paragraph()
                close_lists()
                level = len(heading_match.group(1))
                content = format_inline(escape_text(heading_match.group(2).strip()))
                html_lines.append(f'<h{level}>{content}</h{level}>')
                i += 1
                continue

            list_match = re.match(r'^(\s*)([*+-])\s+(.*)$', raw_line)
            ordered_match = re.match(r'^(\s*)(\d+)\.\s+(.*)$', raw_line)
            if list_match or ordered_match:
                list_type = 'ul' if list_match else 'ol'
                indent = len(list_match.group(1) if list_match else ordered_match.group(1))
                content = format_inline(escape_text(list_match.group(3) if list_match else ordered_match.group(3)))
                if not list_stack or indent > list_stack[-1][0]:
                    html_lines.append(f'<{list_type}>')
                    list_stack.append((indent, list_type))
                elif indent < list_stack[-1][0]:
                    close_lists(indent)
                    if not list_stack or list_stack[-1][1] != list_type:
                        html_lines.append(f'<{list_type}>')
                        list_stack.append((indent, list_type))
                elif list_stack[-1][1] != list_type:
                    close_lists(indent)
                    html_lines.append(f'<{list_type}>')
                    list_stack.append((indent, list_type))
                html_lines.append(f'<li>{content}</li>')
                i += 1
                continue

            if not raw_line.strip():
                close_paragraph()
                close_lists()
                i += 1
                continue

            if not paragraph_open:
                html_lines.append('<p>')
                paragraph_open = True
            html_lines.append(format_inline(line))
            i += 1

        if in_code:
            html_lines.append('<pre class="code-block"><code>' + escape_text('\n'.join(code_lines)) + '</code></pre>')
        close_paragraph()
        close_lists()
        return '<div class="markdown-content">' + ''.join(html_lines) + '</div>'

    def render_research_notes(notes) -> str:
        def render_structured(node) -> str:

            if isinstance(node, str):
                return render_markdown(node)

            if isinstance(node, dict):
                if not node:
                    return wrap_empty('No research notes are available.')
                return ''.join(
                    f'<details class="note-section" open><summary><strong>{escape_text(str(key).replace("_", " ").title())}</strong></summary><div class="note-body">{render_structured(value)}</div></details>'
                    for key, value in node.items()
                )
            if isinstance(node, list):
                if not node:
                    return wrap_empty('No research notes are available.')
                if all(not isinstance(item, (dict, list)) for item in node):
                    return '<ul class="notes-list">' + ''.join(f'<li>{escape_text(item)}</li>' for item in node) + '</ul>'
                return '<div class="notes-group">' + ''.join(f'<div class="notes-list-item">{render_structured(item)}</div>' for item in node) + '</div>'
            return f'<p class="note-text">{escape_text(str(node)).replace("\n", "<br>")}</p>'

        if notes is None or (isinstance(notes, (dict, list)) and not notes):
            return wrap_empty('No research notes were generated for this topic yet.')
        return f'<div class="notes-card">{render_structured(notes)}</div>'

    def render_analysis(analysis) -> str:
        if analysis is None or (isinstance(analysis, str) and not analysis.strip()):
            return wrap_empty('No analysis content is available.')
        if isinstance(analysis, str):
            return render_markdown(analysis)
        if isinstance(analysis, dict):
            return render_markdown('\n\n'.join(f'**{k}:** {v}' for k, v in analysis.items()))
        if isinstance(analysis, list):
            return render_markdown('\n\n'.join(str(item) for item in analysis))
        return render_markdown(str(analysis))

    def render_report_text(report_text) -> str:
        if not report_text or not str(report_text).strip():
            return wrap_empty('No summary report text is available.')
        return render_markdown(str(report_text))

    def render_review(review) -> str:
        approved = review.get('approved', False)
        confidence = int(review.get('confidence', 0))
        issues = review.get('issues', []) or []
        suggestions = review.get('suggestions', []) or []
        badge_class = 'badge-approved' if approved else 'badge-rejected'
        status_text = 'Approved' if approved else 'Not Approved'

        issues_html = '<ul class="detail-list">' + ''.join(f'<li>{escape_text(item)}</li>' for item in issues) + '</ul>' if issues else '<p class="empty-list">No issues found</p>'
        suggestions_html = '<ul class="detail-list">' + ''.join(f'<li>{escape_text(item)}</li>' for item in suggestions) + '</ul>' if suggestions else '<p class="empty-list">No suggestions</p>'
        warning_html = '' if approved else '<div class="warning-banner">This report requires review before publication.</div>'

        return f'''
            {warning_html}
            <section id="review" class="review-section card-card">
                <div class="section-header">
                    <div>
                        <h2>Reviewer Feedback</h2>
                        <p class="section-subtitle">Approval status, confidence, review issues, and suggestions.</p>
                    </div>
                    <span class="badge {badge_class}">{status_text}</span>
                </div>
                <div class="sm-section">
                    <div class="metric-block">
                        <span class="metric-label">Confidence Score</span>
                        <strong>{confidence}%</strong>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {confidence}%;"></div>
                    </div>
                </div>
                <div class="review-block">
                    <div class="review-box">
                        <h3>Issues Found</h3>
                        {issues_html}
                    </div>
                    <div class="review-box">
                        <h3>Suggestions</h3>
                        {suggestions_html}
                    </div>
                </div>
            </section>
        '''

    def render_table_section(tables) -> str:
        if not tables:
            return '<div class="empty-placeholder"><p>No tables found.</p></div>'

        table_cards = []
        for table in tables:
            title = escape_text(table.get('title', 'Table'))
            headers = table.get('headers', []) or []
            rows = table.get('rows', []) or []
            header_html = ''.join(f'<th>{escape_text(str(cell))}</th>' for cell in headers)
            row_html = ''
            for row in rows:
                row_cells = ''.join(f'<td>{escape_text(str(cell))}</td>' for cell in row)
                row_html += f'<tr>{row_cells}</tr>'
            table_cards.append(f'''
                <div class="table-card">
                    <h3>{title}</h3>
                    <div class="table-scroll">
                        <table class="structured-table">
                            <thead><tr>{header_html}</tr></thead>
                            <tbody>{row_html}</tbody>
                        </table>
                    </div>
                </div>
            ''')
        return ''.join(table_cards)

    def render_chart_section(report, assets) -> str:
        chart_urls = assets.get('charts', []) or []
        chart_meta = report.get('charts', []) or []
        if not chart_urls and not chart_meta:
            return '<div class="empty-placeholder"><p>No charts available.</p></div>'

        chart_items = []
        for index, url in enumerate(chart_urls):
            title = escape_text(chart_meta[index].get('title', f'Chart {index + 1}')) if index < len(chart_meta) else f'Chart {index + 1}'
            chart_items.append(f'''
                <div class="media-card">
                    <h3>{title}</h3>
                    <img src="{escape_text(url)}" alt="{title}" />
                </div>
            ''')
        return ''.join(chart_items)

    def render_image_section(report, assets) -> str:
        image_urls = assets.get('images', []) or []
        image_meta = report.get('images', []) or []
        if not image_urls and not image_meta:
            return '<div class="empty-placeholder"><p>No images available.</p></div>'

        image_items = []
        for index, url in enumerate(image_urls):
            meta = image_meta[index] if index < len(image_meta) else {}
            title = escape_text(meta.get('title', f'Image {index + 1}'))
            reason = escape_text(meta.get('reason', ''))
            image_items.append(f'''
                <div class="media-card">
                    <h3>{title}</h3>
                    <img src="{escape_text(url)}" alt="{title}" />
                    <p class="media-caption">{reason}</p>
                </div>
            ''')
        return ''.join(image_items)

    def render_report_section(report) -> str:
        title = escape_text(report.get('title', 'Research Report'))
        executive_summary = render_markdown(report.get('executive_summary', ''))
        sections = report.get('sections', []) or []
        conclusion = render_markdown(report.get('conclusion', ''))

        sections_html = ''
        if sections:
            sections_html = '<ul class="detail-list">' + ''.join(f'<li>{escape_text(str(section))}</li>' for section in sections) + '</ul>'
        else:
            sections_html = '<p class="empty-list">No main sections were generated.</p>'

        return f'''
            <section id="report-summary" class="section-card">
                <div class="section-header">
                    <div>
                        <h2>Report</h2>
                        <p class="section-subtitle">Structured report title, executive summary, main sections, and conclusion.</p>
                    </div>
                </div>
                <div class="report-card">
                    <div class="report-title">
                        <h3>{title}</h3>
                    </div>
                    <div class="report-section-block">
                        <h4>Executive Summary</h4>
                        {executive_summary}
                    </div>
                    <div class="report-section-block">
                        <h4>Main Sections</h4>
                        {sections_html}
                    </div>
                    <div class="report-section-block">
                        <h4>Conclusion</h4>
                        {conclusion}
                    </div>
                </div>
            </section>
        '''

    def render_assets(assets) -> str:
        if not assets:
            return '<div class="empty-placeholder"><p>No assets available.</p></div>'
        items = ''.join(f'<li><strong>{escape_text(k)}:</strong> {escape_text(v)}</li>' for k, v in assets.items())
        return f'<ul class="detail-list">{items}</ul>'

    if isinstance(report, dict):
        payload_report = report.get('report', {}) or {}
        research_notes_html = render_research_notes(report.get('research_notes', {}))
        analysis_html = render_analysis(report.get('analysis', {}))
        review_html = render_review(report.get('review', {}))
        report_html = render_report_section(payload_report)
        tables_html = render_table_section(payload_report.get('tables', []))
        charts_html = render_chart_section(payload_report, report.get('assets', {}))
        images_html = render_image_section(payload_report, report.get('assets', {}))
        download_url = f"/api/download-pdf/{quote(topic)}" if report.get('pdf_path') else None
        download_button = f'<a href="{download_url}" class="download-button">Download PDF</a>' if download_url else ''

        content_html = f'''
            <div class="report-nav">
                <a href="#report-summary">Report</a>
                <a href="#review">Review</a>
                <a href="#research-notes">Research Notes</a>
                <a href="#analysis">Analysis</a>
                <a href="#tables">Tables</a>
                <a href="#charts">Charts</a>
                <a href="#images">Images</a>
            </div>
            {report_html}
            {review_html}
            <section id="research-notes" class="section-card">
                <div class="section-header">
                    <h2>Research Notes</h2>
                    <p class="section-subtitle">Detailed evidence and data captured during the research stage.</p>
                </div>
                <details class="content-panel">
                    <summary>Show Research Notes</summary>
                    {research_notes_html}
                </details>
            </section>
            <section id="analysis" class="section-card">
                <div class="section-header">
                    <h2>Analysis</h2>
                    <p class="section-subtitle">Insights synthesized from evidence and analysis.</p>
                </div>
                <details class="content-panel" open>
                    <summary>Show Analysis</summary>
                    <div class="analysis-card">{analysis_html}</div>
                </details>
            </section>
            <section id="tables" class="section-card">
                <div class="section-header">
                    <h2>Tables</h2>
                    <p class="section-subtitle">Structured tables generated from the report.</p>
                </div>
                {tables_html}
            </section>
            <section id="charts" class="section-card">
                <div class="section-header">
                    <h2>Charts</h2>
                    <p class="section-subtitle">Visual charts generated for the report content.</p>
                </div>
                <div class="media-grid">{charts_html}</div>
            </section>
            <section id="images" class="section-card">
                <div class="section-header">
                    <h2>Images</h2>
                    <p class="section-subtitle">Relevant image assets generated from the report.</p>
                </div>
                <div class="media-grid">{images_html}</div>
            </section>
        '''
    else:
        content_html = render_report_text(report)
        download_button = ''

    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Report: {escape_text(topic)}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #eef1f8;
                color: #1c1f32;
                padding: 24px;
            }}
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 20px;
                box-shadow: 0 24px 80px rgba(63, 63, 68, 0.12);
                overflow: hidden;
            }}
            .hero {{
                padding: 40px 48px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }}
            .hero h1 {{
                font-size: clamp(2rem, 2.5vw, 3rem);
                margin-bottom: 10px;
            }}
            .hero p {{
                color: rgba(255,255,255,0.85);
                max-width: 900px;
                line-height: 1.7;
            }}
            .metadata {{
                margin-top: 18px;
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                font-size: 0.95rem;
                color: rgba(255,255,255,0.9);
            }}
            .badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.7rem 1rem;
                border-radius: 999px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.04em;
                font-size: 0.85rem;
            }}
            .badge-approved {{
                background: rgba(56, 178, 124, 0.15);
                color: #2d8a5c;
            }}
            .badge-rejected {{
                background: rgba(220, 38, 38, 0.12);
                color: #b91c1c;
            }}
            .content {{
                padding: 32px 48px 48px;
                display: grid;
                gap: 28px;
            }}
            .report-nav {{
                display: flex;
                flex-wrap: wrap;
                gap: 14px;
                background: #f4f6ff;
                border: 1px solid #d9e2ff;
                border-radius: 16px;
                padding: 18px 22px;
            }}
            .report-nav a {{
                text-decoration: none;
                color: #334155;
                padding: 10px 16px;
                border-radius: 999px;
                background: #ffffff;
                border: 1px solid transparent;
                transition: all 0.2s ease;
                font-weight: 600;
            }}
            .report-nav a:hover {{
                color: #1e3a8a;
                border-color: #c7d2fe;
                background: #eff6ff;
            }}
            .section-card {{
                background: #f8fbff;
                border: 1px solid #e6ecf5;
                border-radius: 18px;
                padding: 28px;
            }}
            .section-header {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: flex-start;
                gap: 18px;
                margin-bottom: 24px;
            }}
            .section-header h2 {{
                font-size: 1.55rem;
                color: #1f2a56;
            }}
            .section-subtitle {{
                color: #5f6f8c;
                max-width: 760px;
                line-height: 1.6;
            }}
            .notes-card,
            .analysis-card,
            .report-card {{
                display: grid;
                gap: 16px;
            }}
            .note-section {{
                border: 1px solid #d9e2ef;
                border-radius: 14px;
                background: #ffffff;
                padding: 0;
                overflow: hidden;
            }}
            .note-section summary {{
                cursor: pointer;
                list-style: none;
                padding: 18px 22px;
                font-weight: 700;
                color: #1f2a56;
                outline: none;
            }}
            .note-section summary:hover {{
                background: #eef2ff;
            }}
            .note-body {{
                padding: 0 22px 18px 22px;
                border-top: 1px solid #e6ecf5;
            }}
            .notes-list,
            .detail-list {{
                list-style: disc inside;
                color: #2e405f;
            }}
            .notes-list li,
            .detail-list li {{
                margin-bottom: 10px;
                line-height: 1.7;
            }}
            .note-text {{
                color: #344054;
                line-height: 1.75;
            }}
            .content-panel {{
                border: 1px solid #d9e2ef;
                border-radius: 16px;
                background: #ffffff;
                padding: 0.75rem 1rem 1rem;
                margin-top: 16px;
            }}
            .content-panel summary {{
                cursor: pointer;
                font-weight: 700;
                color: #1f2a56;
                padding: 16px 0;
                list-style: none;
                outline: none;
            }}
            .media-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
            }}
            .media-card,
            .table-card {{
                background: #ffffff;
                border: 1px solid #d9e2ef;
                border-radius: 18px;
                padding: 22px;
            }}
            .media-card img {{
                width: 100%;
                max-height: 320px;
                object-fit: contain;
                border-radius: 14px;
                margin-top: 14px;
            }}
            .media-caption {{
                margin-top: 12px;
                color: #475569;
                line-height: 1.7;
            }}
            .report-title h3 {{
                margin-bottom: 8px;
                color: #111827;
            }}
            .report-section-block {{
                margin-top: 18px;
            }}
            .report-section-block h4 {{
                margin-bottom: 10px;
                color: #1f2a56;
            }}
            .structured-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.95rem;
                background: #ffffff;
            }}
            .structured-table th,
            .structured-table td {{
                border: 1px solid #d2dbe6;
                padding: 14px 16px;
                text-align: left;
            }}
            .structured-table th {{
                background: #f4f7ff;
                font-weight: 700;
            }}
            .empty-placeholder {{
                padding: 18px 20px;
                background: #f8fafc;
                border: 1px dashed #cbd5e1;
                border-radius: 14px;
                color: #475569;
            }}
            .detail-list {{
                list-style: disc inside;
                color: #2e405f;
            }}
            .detail-list li {{
                margin-bottom: 10px;
                line-height: 1.7;
            }}
            .markdown-content h1,
            .markdown-content h2,
            .markdown-content h3,
            .markdown-content h4 {{
                color: #1f2a56;
                margin-top: 1.4em;
                margin-bottom: 0.6em;
            }}
            .markdown-content p {{
                margin: 0.8em 0;
                color: #384059;
                line-height: 1.75;
            }}
            .markdown-content ul,
            .markdown-content ol {{
                margin: 0.8em 0 0.8em 1.3em;
                color: #384059;
            }}
            .markdown-content li {{
                margin-bottom: 0.5em;
            }}
            .markdown-content table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                table-layout: fixed;
            }}
            .markdown-content th,
            .markdown-content td {{
                border: 1px solid #d2dbe6;
                padding: 12px 14px;
                text-align: left;
                word-break: break-word;
            }}
            .markdown-content th {{
                background: #f4f7ff;
                font-weight: 700;
            }}
            .code-block,
            .markdown-content pre {{
                background: #0f172a;
                color: #f8fafc;
                border-radius: 14px;
                padding: 18px;
                overflow-x: auto;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 0.95rem;
                line-height: 1.65;
            }}
            .markdown-content code {{
                background: rgba(56, 189, 248, 0.12);
                color: #0f172a;
                padding: 0.15em 0.35em;
                border-radius: 8px;
                font-family: Consolas, 'Courier New', monospace;
            }}
            .table-scroll {{
                overflow-x: auto;
            }}
            .review-section {{
                border: 1px solid #dbe5f2;
                border-radius: 18px;
                padding: 24px;
                background: #ffffff;
            }}
            .review-card {{
                display: grid;
                gap: 20px;
            }}
            .review-header {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 20px;
                flex-wrap: wrap;
            }}
            .metric-block {{
                display: grid;
                gap: 6px;
            }}
            .metric-label {{
                color: #687195;
                font-size: 0.95rem;
            }}
            .progress-bar {{
                width: 100%;
                height: 14px;
                background: #e8edf9;
                border-radius: 999px;
                overflow: hidden;
                margin-top: 6px;
            }}
            .progress-fill {{
                height: 100%;
                background: linear-gradient(90deg, #4f46e5, #7c3aed);
                border-radius: 999px;
            }}
            .review-block {{
                display: grid;
                gap: 18px;
            }}
            .review-box {{
                padding: 20px;
                border-radius: 16px;
                background: #f4f7ff;
                border: 1px solid #d9e2f5;
            }}
            .review-box h3 {{
                margin-bottom: 12px;
                color: #2f3a6a;
            }}
            .warning-banner {{
                padding: 16px 20px;
                border-radius: 14px;
                background: #fff1f0;
                color: #9f1239;
                border: 1px solid #fecdd3;
                margin-bottom: 20px;
                font-weight: 600;
            }}
            .download-button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                margin-top: 18px;
                padding: 14px 24px;
                background: #4f46e5;
                color: white;
                border-radius: 12px;
                text-decoration: none;
                transition: transform 0.2s ease, background 0.2s ease;
            }}
            .download-button:hover {{
                transform: translateY(-1px);
                background: #4338ca;
            }}
            .back-button {{
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 12px 20px;
                color: #334155;
                border: 1px solid #cbd5e1;
                border-radius: 999px;
                text-decoration: none;
                background: #ffffff;
                font-weight: 600;
            }}
            .back-button:hover {{
                background: #f8fafc;
            }}
            .footer {{
                padding: 24px 48px;
                background: #f8fafc;
                color: #64748b;
                font-size: 0.95rem;
                border-top: 1px solid #e2e8f0;
            }}
            @media (max-width: 860px) {{
                .hero,
                .content,
                .footer {{
                    padding-left: 24px;
                    padding-right: 24px;
                }}
                .section-header {{
                    flex-direction: column;
                    align-items: stretch;
                }}
            }}
            @media (max-width: 640px) {{
                body {{ padding: 16px; }}
                .hero {{ padding: 28px 20px; }}
                .content {{ padding: 24px 20px 32px; }}
                .section-card {{ padding: 22px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero">
                <h1>📄 Research Report</h1>
                <p>Review the research workflow output with notes, analysis, review feedback, and PDF export support.</p>
                <div class="metadata">
                    <span><strong>Topic:</strong> {escape_text(topic)}</span>
                    <span><strong>Generated:</strong> {escape_text(timestamp)}</span>
                    <span><strong>PDF available:</strong> {'Yes' if report.get('pdf_path') else 'No'}</span>
                </div>
            </div>
            <div class="content">
                <a href="/" class="back-button">← Back to Home</a>
                {download_button}
                {content_html}
            </div>
            <div class="footer">
                Generated by Research Team Agent API | Multi-agent orchestration with LangGraph
            </div>
        </div>
    </body>
    </html>
    '''


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("RESEARCH TEAM AGENT - FastAPI Server")
    print("="*70)
    print("\nStarting server...")
    print("📍 Home: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
