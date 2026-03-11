#!/usr/bin/env python3
"""
gleanings: research pipeline for under-covered AI breakthroughs.

Runs three research agents in parallel, synthesizes findings, writes output/index.html.

Usage:
    python run.py
    python run.py --model claude-opus-4-6   # use opus for higher quality
    python run.py --no-open                  # don't auto-open browser
"""

import asyncio
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent
AGENTS_DIR = ROOT / "agents"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RESEARCH_AGENTS = [
    "reasoning_inference.md",
    "architecture_training.md",
    "agents_science.md",
]

# Practitioner agent runs separately — different output schema, different section in HTML
PRACTITIONER_AGENT = "practitioner_discoveries.md"

DEFAULT_MODEL = "claude-sonnet-4-6"


def load_prompt(filename: str) -> str:
    return (AGENTS_DIR / filename).read_text()


async def run_research_agent(
    client: anthropic.AsyncAnthropic,
    model: str,
    prompt_file: str,
) -> list[dict]:
    """Run a single research agent and return its findings as a list."""
    prompt = load_prompt(prompt_file)
    label = prompt_file.replace(".md", "")
    print(f"  [{label}] starting...")

    message = await client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown fences if the model wrapped in them
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    try:
        findings = json.loads(raw)
        print(f"  [{label}] done — {len(findings)} findings")
        return findings
    except json.JSONDecodeError as e:
        print(f"  [{label}] WARNING: could not parse JSON: {e}")
        print(f"  [{label}] raw output:\n{raw[:500]}")
        return []


async def run_synthesizer(
    client: anthropic.AsyncAnthropic,
    model: str,
    all_findings: list[dict],
) -> dict:
    """Synthesize all findings into a structured report."""
    print("  [synthesizer] starting...")

    prompt = load_prompt("synthesizer.md")
    findings_json = json.dumps(all_findings, indent=2)

    message = await client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n---\n\nAgent findings:\n\n{findings_json}",
            }
        ],
    )

    raw = message.content[0].text.strip()

    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])

    try:
        report = json.loads(raw)
        theme_count = len(report.get("themes", []))
        print(f"  [synthesizer] done — {theme_count} themes")
        return report
    except json.JSONDecodeError as e:
        print(f"  [synthesizer] WARNING: could not parse JSON: {e}")
        return {
            "generated_at": datetime.now().isoformat(),
            "themes": [],
            "meta": {"total_findings": len(all_findings), "error": str(e)},
        }


def coverage_badge(level: str) -> str:
    labels = {"low": "low coverage", "medium": "moderate coverage", "high": "covered"}
    classes = {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}
    return f'<span class="badge {classes.get(level, "badge-low")}">{labels.get(level, level)}</span>'


def source_link(source: str) -> str:
    if not source:
        return ""
    if source.startswith("http"):
        url = source
    elif source.startswith("arXiv:") or source.startswith("arxiv:"):
        arxiv_id = source.split(":")[-1].strip()
        url = f"https://arxiv.org/abs/{arxiv_id}"
    elif source[:4].isdigit() and "." in source:
        # Bare arXiv ID like 2502.03387
        url = f"https://arxiv.org/abs/{source}"
    else:
        return f'<span class="source-text">{source}</span>'
    return f'<a href="{url}" target="_blank" rel="noopener" class="source-link">{source}</a>'


def render_html(report: dict, practitioner_findings: list[dict] | None = None) -> str:
    generated_at = report.get("generated_at", datetime.now().isoformat())
    try:
        dt = datetime.fromisoformat(generated_at)
        date_str = dt.strftime("%B %d, %Y")
    except Exception:
        date_str = generated_at

    meta = report.get("meta", {})
    total = meta.get("total_findings", 0)
    low_coverage = meta.get("papers_with_low_coverage", 0)
    search_date = meta.get("search_date", date_str)

    headline = report.get("headline_finding", {})
    themes = report.get("themes", [])

    practitioner_findings = practitioner_findings or []

    # Build theme nav
    nav_items = "".join(
        f'<a href="#{t["id"]}" class="nav-item">{t["title"]}</a>'
        for t in themes
        if t.get("id") and t.get("title")
    )
    if practitioner_findings:
        nav_items += '<a href="#practitioner" class="nav-item nav-item--practitioner">In the wild</a>'

    # Build headline section
    if headline:
        headline_html = f"""
        <section class="headline-section">
            <div class="headline-label">Most important finding</div>
            <h2 class="headline-title">{headline.get("title", "")}</h2>
            <p class="headline-summary">{headline.get("summary", "")}</p>
            <div class="headline-source">{source_link(headline.get("source", ""))}</div>
        </section>"""
    else:
        headline_html = ""

    # Build themes
    themes_html = ""
    for theme in themes:
        tid = theme.get("id", "")
        title = theme.get("title", "")
        synthesis = theme.get("synthesis", "")
        findings = theme.get("findings", [])

        cards = ""
        for f in findings:
            ftitle = f.get("title", "")
            fsource = f.get("source", "")
            ffinding = f.get("finding", "")
            fmatters = f.get("why_it_matters", "")
            foverlooked = f.get("why_overlooked", "")
            fcoverage = f.get("heuristic_coverage", "low")

            cards += f"""
            <article class="finding-card">
                <div class="card-header">
                    <h4 class="card-title">{ftitle}</h4>
                    {coverage_badge(fcoverage)}
                </div>
                <p class="card-finding">{ffinding}</p>
                <div class="card-meta">
                    <div class="card-meta-row">
                        <span class="meta-label">Why it matters</span>
                        <span class="meta-value">{fmatters}</span>
                    </div>
                    <div class="card-meta-row">
                        <span class="meta-label">Why overlooked</span>
                        <span class="meta-value">{foverlooked}</span>
                    </div>
                    <div class="card-meta-row source-row">
                        <span class="meta-label">Source</span>
                        <span class="meta-value">{source_link(fsource)}</span>
                    </div>
                </div>
            </article>"""

        themes_html += f"""
        <section class="theme-section" id="{tid}">
            <h3 class="theme-title">{title}</h3>
            <p class="theme-synthesis">{synthesis}</p>
            <div class="findings-grid">
                {cards}
            </div>
        </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gleanings — {date_str}</title>
    <style>
        :root {{
            --bg: #0c0c0e;
            --surface: #111114;
            --surface-raised: #18181d;
            --border: #242428;
            --border-subtle: #1c1c20;
            --text: #e8e8ee;
            --text-secondary: #8a8a96;
            --text-tertiary: #55555f;
            --accent: #6b7fff;
            --accent-dim: rgba(107, 127, 255, 0.12);
            --low: #4ade80;
            --low-bg: rgba(74, 222, 128, 0.1);
            --medium: #fbbf24;
            --medium-bg: rgba(251, 191, 36, 0.1);
            --high: #64748b;
            --high-bg: rgba(100, 116, 139, 0.1);
            --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
            --font-mono: "SF Mono", "Fira Code", "Cascadia Code", monospace;
        }}

        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        html {{ scroll-behavior: smooth; }}

        body {{
            background: var(--bg);
            color: var(--text);
            font-family: var(--font-sans);
            font-size: 15px;
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }}

        /* Layout */
        .page {{
            max-width: 900px;
            margin: 0 auto;
            padding: 0 24px 80px;
        }}

        /* Header */
        .site-header {{
            padding: 48px 0 32px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 40px;
        }}

        .site-title {{
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--accent);
            font-family: var(--font-mono);
            margin-bottom: 12px;
        }}

        .site-desc {{
            font-size: 13px;
            color: var(--text-tertiary);
            max-width: 480px;
            line-height: 1.5;
        }}

        /* Stats bar */
        .stats-bar {{
            display: flex;
            gap: 32px;
            padding: 20px 0;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 40px;
            flex-wrap: wrap;
        }}

        .stat {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .stat-value {{
            font-size: 22px;
            font-weight: 600;
            font-family: var(--font-mono);
            color: var(--text);
            letter-spacing: -0.02em;
        }}

        .stat-label {{
            font-size: 11px;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* Nav */
        .theme-nav {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 48px;
        }}

        .nav-item {{
            font-size: 12px;
            color: var(--text-secondary);
            text-decoration: none;
            padding: 5px 10px;
            border: 1px solid var(--border);
            border-radius: 4px;
            transition: border-color 0.15s, color 0.15s;
        }}

        .nav-item:hover {{
            border-color: var(--accent);
            color: var(--accent);
        }}

        /* Headline */
        .headline-section {{
            background: var(--accent-dim);
            border: 1px solid rgba(107, 127, 255, 0.2);
            border-radius: 8px;
            padding: 28px 32px;
            margin-bottom: 56px;
        }}

        .headline-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--accent);
            font-family: var(--font-mono);
            margin-bottom: 12px;
        }}

        .headline-title {{
            font-size: 19px;
            font-weight: 600;
            line-height: 1.35;
            margin-bottom: 12px;
        }}

        .headline-summary {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.65;
            margin-bottom: 14px;
        }}

        .headline-source {{
            font-size: 12px;
        }}

        /* Theme sections */
        .theme-section {{
            margin-bottom: 60px;
        }}

        .theme-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--text);
        }}

        .theme-synthesis {{
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.65;
            margin-bottom: 24px;
            padding-left: 14px;
            border-left: 2px solid var(--border);
        }}

        /* Findings grid */
        .findings-grid {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        /* Finding card */
        .finding-card {{
            background: var(--surface);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 20px 22px;
            transition: border-color 0.15s;
        }}

        .finding-card:hover {{
            border-color: var(--border);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 10px;
        }}

        .card-title {{
            font-size: 14px;
            font-weight: 600;
            line-height: 1.4;
            flex: 1;
        }}

        .card-finding {{
            font-size: 13.5px;
            color: var(--text-secondary);
            line-height: 1.65;
            margin-bottom: 16px;
        }}

        /* Badge */
        .badge {{
            font-size: 10px;
            font-family: var(--font-mono);
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 3px 7px;
            border-radius: 3px;
            white-space: nowrap;
            flex-shrink: 0;
        }}

        .badge-low {{
            color: var(--low);
            background: var(--low-bg);
        }}

        .badge-medium {{
            color: var(--medium);
            background: var(--medium-bg);
        }}

        .badge-high {{
            color: var(--high);
            background: var(--high-bg);
        }}

        /* Card meta */
        .card-meta {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding-top: 14px;
            border-top: 1px solid var(--border-subtle);
        }}

        .card-meta-row {{
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 8px;
            font-size: 12.5px;
            line-height: 1.5;
        }}

        .meta-label {{
            color: var(--text-tertiary);
            font-family: var(--font-mono);
            font-size: 11px;
            padding-top: 1px;
        }}

        .meta-value {{
            color: var(--text-secondary);
        }}

        /* Source links */
        .source-link {{
            color: var(--accent);
            text-decoration: none;
            font-family: var(--font-mono);
            font-size: 11.5px;
        }}

        .source-link:hover {{
            text-decoration: underline;
        }}

        .source-text {{
            color: var(--text-tertiary);
            font-family: var(--font-mono);
            font-size: 11.5px;
        }}

        /* Footer */
        .site-footer {{
            padding: 32px 0;
            border-top: 1px solid var(--border-subtle);
            font-size: 12px;
            color: var(--text-tertiary);
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
        }}

        /* Practitioner section */
        .practitioner-section {{
            margin-bottom: 60px;
            padding-top: 48px;
            border-top: 1px solid var(--border-subtle);
        }}

        .section-eyebrow {{
            font-size: 11px;
            font-family: var(--font-mono);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #c084fc;
            margin-bottom: 8px;
        }}

        .practitioner-card {{
            border-color: #2a1f33;
        }}

        .practitioner-card:hover {{
            border-color: #3d2b4d;
        }}

        .nav-item--practitioner {{
            border-color: #3d2b4d;
            color: #c084fc;
        }}

        .nav-item--practitioner:hover {{
            border-color: #c084fc;
            color: #c084fc;
        }}

        @media (max-width: 640px) {{
            .card-meta-row {{
                grid-template-columns: 1fr;
                gap: 2px;
            }}
            .meta-label {{
                margin-bottom: 0;
            }}
            .stats-bar {{
                gap: 20px;
            }}
        }}
    </style>
</head>
<body>
<div class="page">

    <header class="site-header">
        <div class="site-title">Gleanings</div>
        <p class="site-desc">Under-covered AI breakthroughs. Ranked by significance to current practice, not by coverage.</p>
    </header>

    <div class="stats-bar">
        <div class="stat">
            <span class="stat-value">{total}</span>
            <span class="stat-label">Findings</span>
        </div>
        <div class="stat">
            <span class="stat-value">{low_coverage}</span>
            <span class="stat-label">Low-coverage papers</span>
        </div>
        <div class="stat">
            <span class="stat-value">{len(themes)}</span>
            <span class="stat-label">Themes</span>
        </div>
        <div class="stat">
            <span class="stat-value">{search_date}</span>
            <span class="stat-label">Search date</span>
        </div>
    </div>

    {headline_html}

    <nav class="theme-nav">
        {nav_items}
    </nav>

    {themes_html}

    {render_practitioner_section(practitioner_findings)}

    <footer class="site-footer">
        <span>Generated {date_str} — 3 research agents + synthesizer</span>
        <span>gleanings</span>
    </footer>

</div>
</body>
</html>"""


def render_practitioner_section(discoveries: list[dict]) -> str:
    if not discoveries:
        return ""

    cards = ""
    for d in discoveries:
        title = d.get("title", "")
        where = d.get("where_shared", "")
        discovery = d.get("discovery", "")
        matters = d.get("why_it_matters", "")
        overlooked = d.get("why_overlooked", "")
        dtype = d.get("discoverer_type", "")
        confirmations = d.get("independent_confirmations", "none")
        coverage = d.get("heuristic_coverage", "low")

        confirm_label = {
            "none": "No independent confirmations",
            "a few": "A few independent confirmations",
            "many": "Many independent confirmations",
        }.get(confirmations, confirmations)

        cards += f"""
        <article class="finding-card practitioner-card">
            <div class="card-header">
                <h4 class="card-title">{title}</h4>
                {coverage_badge(coverage)}
            </div>
            <p class="card-finding">{discovery}</p>
            <div class="card-meta">
                <div class="card-meta-row">
                    <span class="meta-label">Why it matters</span>
                    <span class="meta-value">{matters}</span>
                </div>
                <div class="card-meta-row">
                    <span class="meta-label">Why overlooked</span>
                    <span class="meta-value">{overlooked}</span>
                </div>
                <div class="card-meta-row">
                    <span class="meta-label">Discovered by</span>
                    <span class="meta-value">{dtype} &middot; {confirm_label}</span>
                </div>
                <div class="card-meta-row source-row">
                    <span class="meta-label">Shared on</span>
                    <span class="meta-value source-text">{where}</span>
                </div>
            </div>
        </article>"""

    return f"""
    <section class="practitioner-section" id="practitioner">
        <div class="section-eyebrow">In the wild</div>
        <h3 class="theme-title">Practitioner discoveries</h3>
        <p class="theme-synthesis">Breakthroughs that happened when people actually used the tools — not in labs, but in real work. Most were shared once in a community and then disappeared. The people who found them often didn't realize how significant the discovery was.</p>
        <div class="findings-grid">
            {cards}
        </div>
    </section>"""


async def main(model: str, open_browser: bool) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    print(f"Running with model: {model}")
    print("Phase 1: research agents (parallel)...")

    research_tasks = [
        run_research_agent(client, model, agent_file)
        for agent_file in RESEARCH_AGENTS
    ] + [run_research_agent(client, model, PRACTITIONER_AGENT)]

    *research_results, practitioner_findings = await asyncio.gather(*research_tasks)

    all_research_findings = [f for agent_findings in research_results for f in agent_findings]
    print(f"  research findings: {len(all_research_findings)}")
    print(f"  practitioner discoveries: {len(practitioner_findings)}")

    print("Phase 2: synthesis...")
    report = await run_synthesizer(client, model, all_research_findings)

    # Fill in meta if synthesizer didn't
    if "meta" not in report:
        report["meta"] = {}
    report["meta"].setdefault("total_findings", len(all_research_findings) + len(practitioner_findings))
    report["meta"].setdefault(
        "papers_with_low_coverage",
        sum(1 for f in all_research_findings if f.get("heuristic_coverage") == "low"),
    )
    report["practitioner_discoveries"] = practitioner_findings
    report["meta"].setdefault("search_date", datetime.now().strftime("%Y-%m-%d"))
    report.setdefault("generated_at", datetime.now().isoformat())

    print("Phase 3: rendering HTML...")
    html = render_html(report, practitioner_findings)

    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"  written: {output_path}")

    # Required for GitHub Pages to serve without Jekyll processing
    (OUTPUT_DIR / ".nojekyll").touch()

    # Also save the raw JSON for inspection
    json_path = OUTPUT_DIR / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  written: {json_path}")

    if open_browser:
        subprocess.run(["open", str(output_path)], check=False)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gleanings research pipeline")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't auto-open the HTML output in a browser",
    )
    args = parser.parse_args()

    asyncio.run(main(model=args.model, open_browser=not args.no_open))
