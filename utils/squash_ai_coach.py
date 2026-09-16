"""
Squash AI Coach — Generative AI Coaching Layer

Architecture:
  Computer Vision & Kinematics Engine -> Objective Metrics Dict -> AI Squash Coach (Gemini) -> Structured JSON -> Reports (MD/HTML/TXT)

Guarantees:
  1. 100% additive & non-breaking.
  2. Strict fallback to rule-based coaching on any error, missing key, or timeout.
  3. Grounded strictly in supplied metrics — no hallucinated stats, benchmarks, or history.
  4. Returns structured JSON with personalized training drills targeting detected weaknesses.
"""

import os
import json
import time
import logging

logger = logging.getLogger("SquashAICoach")

# Target models in priority order
PREFERRED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

COACHING_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "executive_summary": {"type": "string"},
        "playing_style": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "areas_to_improve": {"type": "array", "items": {"type": "string"}},
        "tactical_analysis": {"type": "array", "items": {"type": "string"}},
        "movement_analysis": {"type": "array", "items": {"type": "string"}},
        "training_plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drill": {"type": "string"},
                    "duration": {"type": "string"},
                    "sets": {"type": "string"},
                    "focus": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["drill", "duration", "sets", "focus", "reason"]
            }
        },
        "coach_summary": {"type": "string"}
    },
    "required": [
        "executive_summary",
        "playing_style",
        "strengths",
        "areas_to_improve",
        "tactical_analysis",
        "movement_analysis",
        "training_plan",
        "coach_summary"
    ]
}


def generate_rule_based_coaching(metrics: dict) -> dict:
    """
    Deterministic rule-based coaching feedback.
    Acts as the reliable 100% offline fallback whenever Gemini is not available.
    """
    t_rate = float(metrics.get("t_recovery_rate", 0.0))
    dist_per_shot = float(metrics.get("dist_per_shot", 0.0))
    lunges = int(metrics.get("num_lunges", 0))
    strikes = int(metrics.get("num_strikes", 0))
    avg_speed = float(metrics.get("avg_speed_kmh", 0.0))
    peak_speed = float(metrics.get("peak_speed_kmh", 0.0))

    strengths = []
    areas_to_improve = []
    tactical_analysis = []
    movement_analysis = []
    training_plan = []

    # 1. T-Recovery Analysis
    if t_rate >= 70.0:
        strengths.append(f"High T-Zone recovery consistency ({t_rate:.1f}%).")
        tactical_analysis.append(
            f"Strong positional dominance with a {t_rate:.1f}% T-recovery rate. "
            "You consistently reset to the central T, putting defensive pressure on rallies."
        )
    elif t_rate >= 40.0:
        areas_to_improve.append(f"Moderate T-Zone recovery ({t_rate:.1f}%).")
        tactical_analysis.append(
            f"Moderate T-recovery rate ({t_rate:.1f}%). While you reset on routine shots, "
            "you occasionally linger in the deep corners after striking the ball."
        )
        training_plan.append({
            "drill": "Ghosting: Deep Corner to T-Split Step",
            "duration": "12 minutes",
            "sets": "3 sets x 8 reps",
            "focus": "Explosive first recovery step back to the T-hub",
            "reason": f"Observed T-recovery rate of {t_rate:.1f}% indicates delay in recovering from corners."
        })
    else:
        areas_to_improve.append(f"Low T-Zone recovery ({t_rate:.1f}%).")
        tactical_analysis.append(
            f"Low T-recovery rate ({t_rate:.1f}%). You frequently remain in corner zones after hitting, "
            "leaving large portions of the court vulnerable."
        )
        training_plan.append({
            "drill": "Ghosting: 4-Corner Rapid Recovery to T",
            "duration": "15 minutes",
            "sets": "4 sets x 10 reps",
            "focus": "Immediate change of direction upon racket follow-through",
            "reason": f"T-recovery rate is {t_rate:.1f}%, showing a consistent delay returning to the T-zone."
        })

    # 2. Movement & Distance Economy
    if dist_per_shot > 4.5:
        areas_to_improve.append(f"High distance traveled per shot ({dist_per_shot:.1f}m/shot).")
        movement_analysis.append(
            f"High movement distance per stroke ({dist_per_shot:.1f} meters/shot). "
            "This suggests late anticipation or indirect running paths to the ball."
        )
        training_plan.append({
            "drill": "Early Preparation & Linear Footwork Routine",
            "duration": "12 minutes",
            "sets": "3 sets x 6 minutes",
            "focus": "Preparing racket shoulder early and taking direct diagonal movement lines",
            "reason": f"Player averaged {dist_per_shot:.1f} meters per shot, signaling inefficient court travel."
        })
    else:
        strengths.append(f"Efficient court movement economy ({dist_per_shot:.1f}m/shot).")
        movement_analysis.append(
            f"Good movement economy at {dist_per_shot:.1f} meters per shot. "
            "You are moving directly to hitting positions without excessive lateral wasted steps."
        )

    # 3. Biomechanics & Lunges
    if lunges > 0:
        strengths.append(f"Active lower-body lunge engagement ({lunges} lunges detected).")
        movement_analysis.append(
            f"{lunges} lunges detected. Focus on maintaining a 90-degree knee angle on low lunges "
            "to ensure optimal balance and fast push-off recovery."
        )
    else:
        areas_to_improve.append("Low dynamic knee flexion / zero lunges detected.")
        movement_analysis.append(
            "No lunges detected. When retrieving low corner balls, bend at the knees and hips "
            "rather than reaching purely with the upper torso."
        )
        training_plan.append({
            "drill": "Front Court Low Lunge & Push-Off Drill",
            "duration": "10 minutes",
            "sets": "3 sets x 8 reps per leg",
            "focus": "Deep front knee bend with upright spine and immediate backward push-off",
            "reason": "Zero lunges were registered, indicating potential bending from the waist on low shots."
        })

    # Default drill if training plan has room
    if len(training_plan) < 2:
        training_plan.append({
            "drill": "Length & Straight Drive Consistency Practice",
            "duration": "15 minutes",
            "sets": "3 sets x 20 consecutive drives",
            "focus": "Tight ball path hugging the side wall and controlled strike tempo",
            "reason": "Baseline skill reinforcement to complement movement and recovery mechanics."
        })

    # Style classification
    if avg_speed > 7.0 or peak_speed > 13.0:
        playing_style = "High-Intensity Aggressive / High-Tempo"
    elif dist_per_shot < 3.5 and t_rate > 60:
        playing_style = "Economical Tactical / Central Control"
    else:
        playing_style = "Developing All-Court Player"

    return {
        "executive_summary": (
            f"Analyzed {metrics.get('duration_sec', 0):.1f}s of footage with {strikes} shots executed. "
            f"Player covered {metrics.get('total_distance_m', 0):.1f}m with an average velocity of "
            f"{avg_speed:.1f} km/h and peak of {peak_speed:.1f} km/h. "
            f"T-zone recovery consistency was {t_rate:.1f}%."
        ),
        "playing_style": playing_style,
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "tactical_analysis": tactical_analysis,
        "movement_analysis": movement_analysis,
        "training_plan": training_plan,
        "coach_summary": (
            "Continue prioritizing quick recovery steps toward the central T-zone after each stroke. "
            "Focusing on the prescribed drills will directly reduce unnecessary running distance."
        )
    }


def call_gemini_coach(metrics: dict, api_key: str) -> dict:
    """
    Invokes Google Gemini with structured JSON output schema.
    Uses the modern google.genai SDK or falls back to google.generativeai.
    """
    prompt = f"""You are the AI Squash Coach in an automated computer vision squash training system.

CRITICAL INSTRUCTIONS & GUARDRAILS:
1. Do NOT invent measurements, biomechanical observations, benchmarks, player history, or performance facts.
2. You may ONLY reason from the exact metrics provided below.
3. Call yourself "AI Squash Coach". Do NOT claim any human or professional certifications (do NOT say "PSA certified").
4. Output strictly valid JSON conforming to the requested schema. No markdown wrappers around the JSON.
5. In the "training_plan", generate personalized drills specifically targeting the detected weaknesses.
   - Low T-recovery -> T-recovery / ghosting drills
   - High distance per shot -> movement-efficiency / anticipation drills
   - Zero or low lunges -> lunge mobility & knee-bend conditioning
   - Low shot volume -> rally consistency drills
   Every drill must have a data-driven "reason" referencing the provided metrics.

SESSION METRICS MEASURED BY COMPUTER VISION & KINEMATICS ENGINE:
- Total Session Duration: {metrics.get('duration_sec', 0.0):.1f} seconds
- Total Shots Hit: {metrics.get('num_strikes', 0)}
- T-Zone Recovery Rate: {metrics.get('t_recovery_rate', 0.0):.1f}% (percentage of shots where player recovered to central T)
- Total Distance Covered: {metrics.get('total_distance_m', 0.0):.2f} meters
- Distance Covered Per Shot: {metrics.get('dist_per_shot', 0.0):.2f} meters/shot
- Average Speed: {metrics.get('avg_speed_kmh', 0.0):.2f} km/h
- Peak Speed: {metrics.get('peak_speed_kmh', 0.0):.2f} km/h
- Lunges Detected: {metrics.get('num_lunges', 0)}

Generate the structured JSON response now."""

    # Method 1: Try modern google.genai SDK
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        for model_name in PREFERRED_GEMINI_MODELS:
            try:
                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=COACHING_JSON_SCHEMA,
                    temperature=0.2,
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                raw_text = response.text.strip()
                parsed = json.loads(raw_text)
                return {
                    "source": "AI_GEMINI",
                    "model": model_name,
                    "status": "success",
                    "data": parsed
                }
            except Exception as model_err:
                logger.debug("Model %s failed: %s, trying next", model_name, model_err)
                continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[AI Coach] google.genai client error: %s", e)

    # Method 2: Try google.generativeai (legacy SDK fallback)
    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=api_key)
        for model_name in PREFERRED_GEMINI_MODELS:
            try:
                model = legacy_genai.GenerativeModel(
                    model_name=model_name,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.2}
                )
                response = model.generate_content(prompt)
                raw_text = response.text.strip()
                parsed = json.loads(raw_text)
                return {
                    "source": "AI_GEMINI",
                    "model": model_name,
                    "status": "success",
                    "data": parsed
                }
            except Exception as model_err:
                logger.debug("Legacy model %s failed: %s", model_name, model_err)
                continue
    except Exception as e:
        logger.warning("[AI Coach] legacy google.generativeai error: %s", e)

    raise RuntimeError("All Gemini model generation attempts failed.")


def get_coaching_insights(metrics: dict, api_key: str = None) -> dict:
    """
    Main entrypoint for obtaining coaching insights.
    Guarantees non-breaking execution with automatic fallback.
    """
    # Check for API key in parameter or environment variable
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")

    if not resolved_key:
        print("[AI Coach] No GEMINI_API_KEY found. Utilizing deterministic rule-based coaching feedback.")
        fallback_data = generate_rule_based_coaching(metrics)
        return {
            "source": "RULE_BASED_FALLBACK",
            "model": "Deterministic Heuristics Engine",
            "status": "fallback",
            "reason": "No GEMINI_API_KEY supplied",
            "data": fallback_data
        }

    print("[AI Coach] Querying Google Gemini for personalized tactical and training recommendations...")
    start_time = time.time()
    try:
        result = call_gemini_coach(metrics, resolved_key)
        elapsed = time.time() - start_time
        print(f"[AI Coach] Gemini coaching plan generated successfully ({result['model']}) in {elapsed:.2f}s.")
        return result
    except Exception as exc:
        print(f"[AI Coach] Warning: Gemini request failed ({exc}). Seamlessly falling back to rule-based analysis.")
        fallback_data = generate_rule_based_coaching(metrics)
        return {
            "source": "RULE_BASED_FALLBACK",
            "model": "Deterministic Heuristics Engine",
            "status": "fallback",
            "reason": f"API Failure: {str(exc)}",
            "data": fallback_data
        }


def build_markdown_report(metrics: dict, coaching_result: dict, video_path: str) -> str:
    """
    Formats both Objective CV Metrics and AI Coaching Insights into structured Markdown.
    """
    data = coaching_result.get("data", {})
    source = coaching_result.get("source", "RULE_BASED_FALLBACK")
    model_name = coaching_result.get("model", "Heuristics Engine")
    
    duration = metrics.get("duration_sec", 0.0)
    strikes = metrics.get("num_strikes", 0)
    t_rate = metrics.get("t_recovery_rate", 0.0)
    distance = metrics.get("total_distance_m", 0.0)
    avg_speed = metrics.get("avg_speed_kmh", 0.0)
    peak_speed = metrics.get("peak_speed_kmh", 0.0)
    lunges = metrics.get("num_lunges", 0)
    dist_per_shot = metrics.get("dist_per_shot", 0.0)

    t_status = "🟢 Excellent" if t_rate >= 70 else "🟡 Moderate" if t_rate >= 40 else "🔴 Action Required"
    source_tag = "🤖 Generated by Google Gemini (" + model_name + ")" if source == "AI_GEMINI" else "⚙️ Rule-Based Heuristic Engine (Offline Fallback)"

    md = []
    md.append("# 🏆 Squash Performance & AI Coaching Report\n")
    md.append(f"**Source Video:** `{os.path.basename(video_path)}` | **Duration Analyzed:** `{duration:.1f} seconds`\n")
    md.append("> *\"Computer vision measures what the player actually did; Generative AI turns those measurements into personalized coaching and training recommendations.\"*\n\n")

    md.append("---\n")
    md.append("## 📊 PART 1: OBJECTIVE SESSION METRICS\n")
    md.append("*Measured directly via YOLOv8 Pose Estimation, Custom Ball Tracking & Planar Court Homography.*\n\n")
    md.append("| Metric Category | Metric Measured | Value | Status / Context |\n")
    md.append("| :--- | :--- | :--- | :--- |\n")
    md.append(f"| 🎯 Tactical | **Total Shots Hit** | `{strikes}` | Stroke count from speed/trajectory reversals |\n")
    md.append(f"| ⏱️ Tactical | **T-Recovery Rate** | `{t_rate:.1f}%` | {t_status} |\n")
    md.append(f"| 🏃 Kinematics | **Total Distance Covered** | `{distance:.2f} m` | Calibrated via 6.4m × 9.75m court coordinates |\n")
    md.append(f"| 📏 Kinematics | **Distance Per Shot** | `{dist_per_shot:.2f} m/shot` | Movement economy indicator |\n")
    md.append(f"| ⚡ Velocity | **Average / Peak Speed** | `{avg_speed:.1f} / {peak_speed:.1f} km/h` | Glitch-filtered velocity tracking |\n")
    md.append(f"| 🦵 Biomechanics | **Lunges Completed** | `{lunges}` | Knee-to-torso angle flexion detector |\n\n")

    md.append("---\n")
    md.append(f"## 🤖 PART 2: AI SQUASH COACH INSIGHTS\n")
    md.append(f"*{source_tag}*\n\n")

    md.append(f"### 📋 Executive Summary\n{data.get('executive_summary', '')}\n\n")
    md.append(f"**Observed Playing Style:** `{data.get('playing_style', 'All-Court')}`\n\n")

    # Strengths & Areas for Improvement
    md.append("### 🌟 Observed Strengths\n")
    for s in data.get("strengths", []):
        md.append(f"- {s}\n")
    md.append("\n")

    md.append("### ⚠️ Areas for Improvement\n")
    for a in data.get("areas_to_improve", []):
        md.append(f"- {a}\n")
    md.append("\n")

    # Tactical & Movement Breakdown
    md.append("### 🧭 Tactical & Court Craft Analysis\n")
    for t in data.get("tactical_analysis", []):
        md.append(f"- {t}\n")
    md.append("\n")

    md.append("### 🏃 Biomechanical & Movement Dynamics\n")
    for m in data.get("movement_analysis", []):
        md.append(f"- {m}\n")
    md.append("\n")

    # Personalized Training Plan
    md.append("### 🏋️ Personalized Prescribed Training Plan\n")
    md.append("*Custom drill routine generated strictly from detected performance bottlenecks:*\n\n")
    md.append("| Drill Name | Duration | Sets / Reps | Target Focus | Data-Driven Justification |\n")
    md.append("| :--- | :--- | :--- | :--- | :--- |\n")
    for drill in data.get("training_plan", []):
        md.append(
            f"| **{drill.get('drill')}** | `{drill.get('duration')}` | `{drill.get('sets')}` | "
            f"{drill.get('focus')} | *{drill.get('reason')}* |\n"
        )
    md.append("\n")

    md.append(f"### 💬 Coach Closing Note\n> {data.get('coach_summary', '')}\n\n")
    md.append("---\n*Report generated by Squash Personal Trainer with Google Gemini AI Coach Layer.*")

    return "".join(md)


def build_html_report(metrics: dict, coaching_result: dict, video_path: str) -> str:
    """
    Renders an interactive, responsive HTML5 dashboard for session review.
    """
    data = coaching_result.get("data", {})
    source = coaching_result.get("source", "RULE_BASED_FALLBACK")
    model_name = coaching_result.get("model", "Heuristics Engine")
    
    duration = metrics.get("duration_sec", 0.0)
    strikes = metrics.get("num_strikes", 0)
    t_rate = metrics.get("t_recovery_rate", 0.0)
    distance = metrics.get("total_distance_m", 0.0)
    avg_speed = metrics.get("avg_speed_kmh", 0.0)
    peak_speed = metrics.get("peak_speed_kmh", 0.0)
    lunges = metrics.get("num_lunges", 0)
    dist_per_shot = metrics.get("dist_per_shot", 0.0)
    t_status = "🟢 Excellent" if t_rate >= 70 else "🟡 Moderate" if t_rate >= 40 else "🔴 Action Required"

    badge_class = "badge-ai" if source == "AI_GEMINI" else "badge-fallback"
    badge_text = f"AI Generated: {model_name}" if source == "AI_GEMINI" else f"Rule-Based Engine ({model_name})"

    drills_html = ""
    for d in data.get("training_plan", []):
        drills_html += f"""
        <div class="drill-card">
            <div class="drill-title">{d.get('drill', '')}</div>
            <div class="drill-meta">
                <span>⏱️ {d.get('duration', '')}</span>
                <span>🔁 {d.get('sets', '')}</span>
            </div>
            <div class="drill-focus"><strong>Focus:</strong> {d.get('focus', '')}</div>
            <div class="drill-reason"><strong>Why Prescribed:</strong> {d.get('reason', '')}</div>
        </div>
        """

    strengths_html = "".join([f"<li>{s}</li>" for s in data.get("strengths", [])])
    improvements_html = "".join([f"<li>{a}</li>" for a in data.get("areas_to_improve", [])])
    tactical_html = "".join([f"<li>{t}</li>" for t in data.get("tactical_analysis", [])])
    movement_html = "".join([f"<li>{m}</li>" for m in data.get("movement_analysis", [])])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Squash Trainer — AI Performance Report</title>
    <style>
        :root {{
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --bg-card-hover: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #10b981;
            --accent-blue: #38bdf8;
            --accent-purple: #a855f7;
            --accent-amber: #f59e0b;
            --border-color: #334155;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            padding: 30px 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .header h1 {{ font-size: 1.8rem; font-weight: 700; color: #fff; }}
        .header p {{ color: var(--text-muted); font-size: 0.95rem; margin-top: 4px; }}
        .badge {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }}
        .badge-ai {{ background: rgba(168, 85, 247, 0.2); color: var(--accent-purple); border: 1px solid var(--accent-purple); }}
        .badge-fallback {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
        
        .hook-banner {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
            border-left: 4px solid var(--accent-green);
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            font-size: 0.95rem;
            color: #cbd5e1;
        }}

        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--accent-blue);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 35px;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.2s ease;
        }}
        .metric-card:hover {{ transform: translateY(-2px); }}
        .metric-label {{ color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        .metric-value {{ font-size: 1.75rem; font-weight: 700; margin: 8px 0 4px 0; color: #fff; }}
        .metric-sub {{ font-size: 0.85rem; color: var(--accent-green); }}

        .coach-section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 35px;
        }}
        .coach-summary-box {{
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 24px;
            border-left: 3px solid var(--accent-purple);
        }}
        .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
        @media (max-width: 768px) {{ .columns {{ grid-template-columns: 1fr; }} }}
        
        .box {{
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 18px;
        }}
        .box h4 {{ margin-bottom: 12px; font-size: 1rem; color: #f1f5f9; }}
        .box ul {{ padding-left: 20px; }}
        .box li {{ margin-bottom: 8px; font-size: 0.92rem; color: #cbd5e1; }}

        .drills-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .drill-card {{
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .drill-title {{ font-weight: 700; font-size: 1.05rem; color: #38bdf8; }}
        .drill-meta {{ display: flex; gap: 14px; font-size: 0.85rem; color: #a855f7; }}
        .drill-focus {{ font-size: 0.9rem; color: #e2e8f0; }}
        .drill-reason {{ font-size: 0.85rem; color: var(--text-muted); font-style: italic; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px; }}

        .footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🎾 Squash Trainer AI Coach Report</h1>
                <p>Video: <code>{os.path.basename(video_path)}</code> &bull; Analyzed Duration: <strong>{duration:.1f}s</strong></p>
            </div>
            <div>
                <span class="badge {badge_class}">{badge_text}</span>
            </div>
        </div>

        <div class="hook-banner">
            💡 <strong>Pipeline Principle:</strong> Computer vision measures what the player actually did; Generative AI turns those measurements into personalized coaching and training recommendations.
        </div>

        <div class="section-title">📊 Part 1: Objective Session Telemetry (CV & Kinematics)</div>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Shots Hit</div>
                <div class="metric-value">{strikes}</div>
                <div class="metric-sub">Stroke detection</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">T-Recovery Rate</div>
                <div class="metric-value">{t_rate:.1f}%</div>
                <div class="metric-sub">{t_status}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Distance Covered</div>
                <div class="metric-value">{distance:.1f} m</div>
                <div class="metric-sub">Calibrated meters</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Distance / Shot</div>
                <div class="metric-value">{dist_per_shot:.1f} m</div>
                <div class="metric-sub">Movement economy</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average / Peak Speed</div>
                <div class="metric-value">{avg_speed:.1f} <span style="font-size:1rem;color:#94a3b8">/ {peak_speed:.1f}</span></div>
                <div class="metric-sub">km/h</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lunges Detected</div>
                <div class="metric-value">{lunges}</div>
                <div class="metric-sub">Pose angle analysis</div>
            </div>
        </div>

        <div class="section-title">🤖 Part 2: AI Squash Coach Insights & Prescribed Drills</div>
        <div class="coach-section">
            <div class="coach-summary-box">
                <div style="font-size:0.85rem; text-transform:uppercase; color:var(--accent-purple); font-weight:700; margin-bottom:4px;">Executive Assessment</div>
                <p style="color:#e2e8f0; font-size:1rem;">{data.get('executive_summary', '')}</p>
                <div style="margin-top:10px; font-size:0.9rem; color:#94a3b8;">
                    <strong>Identified Style Archetype:</strong> <span style="color:#38bdf8;">{data.get('playing_style', 'Developing Player')}</span>
                </div>
            </div>

            <div class="columns">
                <div class="box">
                    <h4 style="color:var(--accent-green);">🌟 Key Strengths Observed</h4>
                    <ul>{strengths_html}</ul>
                </div>
                <div class="box">
                    <h4 style="color:var(--accent-amber);">⚠️ Areas for Improvement</h4>
                    <ul>{improvements_html}</ul>
                </div>
            </div>

            <div class="columns">
                <div class="box">
                    <h4>🧭 Tactical T-Zone & Court Craft</h4>
                    <ul>{tactical_html}</ul>
                </div>
                <div class="box">
                    <h4>🏃 Movement Economy & Biomechanics</h4>
                    <ul>{movement_html}</ul>
                </div>
            </div>

            <div style="margin-top: 24px;">
                <h3 style="font-size:1.15rem; color:#fff; margin-bottom:12px;">🏋️ Prescribed Corrective Training Plan</h3>
                <p style="color:var(--text-muted); font-size:0.9rem; margin-bottom:14px;">
                    Targeted practice routines addressing the measured bottlenecks from this session.
                </p>
                <div class="drills-grid">
                    {drills_html}
                </div>
            </div>

            <div style="margin-top: 24px; padding: 16px; background: rgba(16, 185, 129, 0.08); border-left: 3px solid var(--accent-green); border-radius: 6px;">
                <strong style="color:var(--accent-green);">Coach Closing Note:</strong>
                <p style="margin-top:4px; font-size:0.92rem; color:#e2e8f0;">{data.get('coach_summary', '')}</p>
            </div>
        </div>

        <div class="footer">
            Squash Personal Trainer &bull; Computer Vision &bull; Kinematics Homography &bull; Google Gemini Generative AI
        </div>
    </div>
</body>
</html>
"""
    return html


def build_text_report(metrics: dict, coaching_result: dict, video_path: str) -> str:
    """
    Plain-text formatted report.
    """
    data = coaching_result.get("data", {})
    source = coaching_result.get("source", "RULE_BASED_FALLBACK")
    
    text = [
        "==================================================",
        "          SQUASH PERSONAL TRAINER REPORT          ",
        "==================================================",
        f"\nSource Video: {os.path.basename(video_path)}",
        f"Total Duration Analyzed: {metrics.get('duration_sec', 0):.1f} seconds\n",
        "WORKOUT METRICS (COMPUTER VISION):",
        f"- Total Distance Covered: {metrics.get('total_distance_m', 0):.2f} meters",
        f"- Average Speed: {metrics.get('avg_speed_kmh', 0):.2f} km/h",
        f"- Peak Speed: {metrics.get('peak_speed_kmh', 0):.2f} km/h",
        f"- Total Lunges Detected: {metrics.get('num_lunges', 0)}",
        f"- Distance Per Shot: {metrics.get('dist_per_shot', 0):.2f} m/shot\n",
        "TACTICAL METRICS (COMPUTER VISION):",
        f"- Total Shots Hit: {metrics.get('num_strikes', 0)}",
        f"- T-Recovery Rate: {metrics.get('t_recovery_rate', 0):.1f}%\n",
        "==================================================",
        f"AI SQUASH COACH FEEDBACK [{source}]:",
        "==================================================",
        f"\nExecutive Summary:\n{data.get('executive_summary', '')}\n",
        f"Playing Style Archetype: {data.get('playing_style', '')}\n",
        "Key Strengths:",
    ]
    for s in data.get("strengths", []):
        text.append(f"  * {s}")
    text.append("\nAreas to Improve:")
    for a in data.get("areas_to_improve", []):
        text.append(f"  * {a}")
    text.append("\nPrescribed Training Plan:")
    for d in data.get("training_plan", []):
        text.append(f"  - Drill: {d.get('drill')} ({d.get('duration')}, {d.get('sets')})")
        text.append(f"    Focus: {d.get('focus')}")
        text.append(f"    Reason: {d.get('reason')}")
    text.append(f"\nCoach Summary:\n{data.get('coach_summary', '')}\n")
    text.append("==================================================")
    return "\n".join(text)
