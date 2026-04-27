"""
CSS for the Gradio app. Anchored on UNC Charlotte's official palette to
match the project's analysis figures.
"""

from __future__ import annotations

from config import (
    CHARLOTTE_GREEN, NINER_GOLD, ATHLETIC_GREEN, ATHLETIC_GOLD,
    NEAR_BLACK, GREY_DARK, GREY, GREY_LIGHT, BG_TINT, WHITE,
    SPEAKER_REP_COLOR, SPEAKER_CALLER_COLOR,
    FLAG_THREAT_COLOR, FLAG_PRED_COLOR, FLAG_NEUTRAL_COLOR,
)


CSS = f"""
/* ------------------------------------------------------------------- */
/* Page-level                                                          */
/* ------------------------------------------------------------------- */

:root {{
    --uncc-green: {CHARLOTTE_GREEN};
    --uncc-gold: {NINER_GOLD};
    --uncc-green-bright: {ATHLETIC_GREEN};
    --uncc-gold-bright: {ATHLETIC_GOLD};
    --uncc-bg: {BG_TINT};
    --uncc-text: {NEAR_BLACK};
    --uncc-grey-dark: {GREY_DARK};
    --uncc-grey: {GREY};
    --uncc-grey-light: {GREY_LIGHT};
    --speaker-rep: {SPEAKER_REP_COLOR};
    --speaker-caller: {SPEAKER_CALLER_COLOR};
    --flag-threat: {FLAG_THREAT_COLOR};
    --flag-pred: {FLAG_PRED_COLOR};
    --flag-neutral: {FLAG_NEUTRAL_COLOR};
}}

.gradio-container {{
    max-width: 1280px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

/* Header banner */
.header-banner {{
    background: transparent;
    color: var(--uncc-green);
    padding: 18px 24px;
    margin-bottom: 16px;
    border-left: 5px solid var(--uncc-gold);
}}
.header-banner h1 {{
    margin: 0;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--uncc-green);
}}
.header-banner .subtitle {{
    font-size: 13px;
    color: var(--uncc-green);
    opacity: 0.75;
    margin-top: 2px;
}}

/* ------------------------------------------------------------------- */
/* Chat window — replaces Gradio's default Chatbot styling              */
/* ------------------------------------------------------------------- */

.chat-window {{
    background: {WHITE};
    border: 1px solid var(--uncc-grey-light);
    border-radius: 6px;
    padding: 14px 16px;
    min-height: 600px;
    max-height: 760px;
    overflow-y: auto;
    font-size: 14px;
    line-height: 1.5;
}}

.turn {{
    margin-bottom: 14px;
    padding-bottom: 14px;
    border-bottom: 1px dashed var(--uncc-grey-light);
}}
.turn:last-child {{ border-bottom: none; }}

.turn-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.turn-header .speaker-rep {{ color: var(--speaker-rep); }}
.turn-header .speaker-caller {{ color: var(--speaker-caller); }}
.turn-header .turn-index {{
    color: var(--uncc-grey);
    font-weight: 400;
    font-size: 11px;
}}

.turn-text {{
    color: var(--uncc-text);
    margin: 4px 0 6px 0;
    white-space: pre-wrap;
}}

/* Two-column layout: text on left, detection on right.
   Flex with align-items: start so they share top edge. */
.turn-body {{
    display: flex;
    gap: 16px;
    align-items: flex-start;
    margin-top: 4px;
}}
.turn-body-left {{
    flex: 1 1 auto;
    min-width: 0;       /* allow text wrapping inside flex */
}}
.turn-body-right {{
    flex: 0 0 280px;    /* fixed width for detection panel */
    max-width: 320px;
}}
.turn-body-right:empty {{
    display: none;
}}
/* On narrow viewports, stack instead of side-by-side */
@media (max-width: 900px) {{
    .turn-body {{
        flex-direction: column;
    }}
    .turn-body-right {{
        flex: 0 0 auto;
        max-width: 100%;
    }}
}}

/* Flag chips */
.flags-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}}
.flag {{
    display: inline-flex;
    align-items: center;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 12px;
    border: 1px solid;
    font-weight: 500;
    letter-spacing: 0.02em;
}}
.flag-actual {{
    color: var(--flag-threat);
    border-color: var(--flag-threat);
    background: rgba(163, 58, 58, 0.06);
}}
.flag-predicted {{
    color: var(--flag-pred);
    border-color: var(--flag-pred);
    background: rgba(122, 92, 38, 0.08);
}}
.flag-actual::before  {{ content: '● '; }}
.flag-predicted::before {{ content: '◆ '; }}

/* Detection panel underneath each turn */
.detection-panel {{
    margin-top: 8px;
    padding: 8px 10px;
    background: var(--uncc-bg);
    border-left: 3px solid var(--uncc-gold);
    border-radius: 3px;
    font-size: 12px;
    color: var(--uncc-grey-dark);
}}
.detection-panel .det-title {{
    font-weight: 600;
    color: var(--uncc-green);
    margin-bottom: 4px;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.detection-panel table {{
    width: 100%;
    border-collapse: collapse;
}}
.detection-panel td, .detection-panel th {{
    padding: 3px 6px;
    text-align: left;
    border-bottom: 1px solid var(--uncc-grey-light);
}}
.detection-panel th {{
    font-weight: 600;
    font-size: 10px;
    color: var(--uncc-grey-dark);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.det-pred-Y {{ color: var(--flag-threat); font-weight: 600; }}
.det-pred-S {{ color: var(--flag-pred); font-weight: 600; }}
.det-pred-N {{ color: var(--uncc-grey); }}
.det-latency {{ color: var(--uncc-grey); font-variant-numeric: tabular-nums; }}

/* The "transcribing" cursor */
.transcribing-cursor {{
    display: inline-block;
    width: 6px;
    height: 14px;
    background: var(--uncc-green);
    margin-left: 2px;
    animation: blink 0.9s steps(2, start) infinite;
    vertical-align: text-bottom;
}}
@keyframes blink {{
    to {{ visibility: hidden; }}
}}

/* Generating-conversation placeholder shown in the chat area while the
   dynamic-mode generator's first API call is in flight. */
.generating-indicator {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px;
    color: var(--uncc-green);
}}
.generating-spinner {{
    width: 38px;
    height: 38px;
    border: 4px solid var(--uncc-grey-light);
    border-top: 4px solid var(--uncc-gold);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 14px;
}}
.generating-text {{
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 0.01em;
}}
@keyframes spin {{
    to {{ transform: rotate(360deg); }}
}}

/* Detection result boxes — large, status-driven typography */
.det-box {{
    border-radius: 6px;
    padding: 10px 12px;
    margin: 0 0 6px 0;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}
.det-box:last-child {{ margin-bottom: 0; }}
.detection-panel-v2 {{
    margin: 0;
}}
.det-box-detected {{
    background: rgba(220, 38, 38, 0.06);
    border-left: 4px solid #dc2626;
}}
.det-box-clear {{
    background: rgba(37, 99, 235, 0.04);
    border-left: 4px solid #2563eb;
}}
.det-box-pending {{
    background: rgba(120, 120, 120, 0.06);
    border-left: 4px solid var(--uncc-grey);
}}
.det-box-failed {{
    background: rgba(120, 120, 120, 0.10);
    border-left: 4px solid var(--uncc-grey);
}}
.det-box-headline {{
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.005em;
    line-height: 1.25;
}}
.det-box-headline-detected {{ color: #dc2626; }}
.det-box-headline-clear    {{ color: #2563eb; opacity: 0.85; }}
.det-box-headline-pending  {{ color: var(--uncc-grey-dark); font-weight: 500; }}
.det-box-headline-failed   {{ color: var(--uncc-grey-dark); font-weight: 500; }}
.det-box-sub {{
    margin-top: 4px;
    font-size: 12px;
    color: var(--uncc-grey-dark);
    opacity: 0.75;
    font-variant-numeric: tabular-nums;
}}
.det-box-stance {{
    font-size: 11px;
    color: var(--uncc-grey-dark);
    opacity: 0.55;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}}
.det-pending-spinner {{
    display: inline-block;
    width: 10px;
    height: 10px;
    border: 2px solid var(--uncc-grey-light);
    border-top: 2px solid var(--uncc-grey);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 6px;
    vertical-align: -1px;
}}

/* ------------------------------------------------------------------- */
/* Sidebar / control panel                                              */
/* ------------------------------------------------------------------- */

.section-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--uncc-green);
    margin: 12px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 2px solid var(--uncc-gold);
}}

/* Status pill */
.status-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.status-idle    {{ background: var(--uncc-grey-light); color: var(--uncc-grey-dark); }}
.status-running {{ background: var(--uncc-green); color: {WHITE}; }}
.status-done    {{ background: var(--uncc-gold); color: {NEAR_BLACK}; }}
.status-error   {{ background: var(--flag-threat); color: {WHITE}; }}

/* Button polish */
.primary-btn {{
    background: var(--uncc-green) !important;
    color: {WHITE} !important;
    border: none !important;
}}
.primary-btn:hover {{
    background: var(--uncc-green-bright) !important;
}}
.secondary-btn {{
    background: {WHITE} !important;
    color: var(--uncc-green) !important;
    border: 1px solid var(--uncc-green) !important;
}}
.secondary-btn:hover {{
    background: var(--uncc-bg) !important;
}}
"""
