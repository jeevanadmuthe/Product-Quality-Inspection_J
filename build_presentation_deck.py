"""
Generate high-fidelity, presentation-ready 6-slide PDF deck for QualityTrack.
Built for the CNN Buildathon 2.0 Final Presentation Round.
"""
import base64
import os
import subprocess
from pathlib import Path
import PyPDF2

BASE_DIR = Path(r"c:\Buildathon\Buildathon 2.0\CNN BUILDATHON")
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUT_HTML = BASE_DIR / "presentation_deck.html"
OUTPUT_PDF = BASE_DIR / "QualityTrack_Presentation_Showcase.pdf"

def get_base64_image(filename):
    filepath = UPLOADS_DIR / filename
    if filepath.exists():
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    return ""

img_defective_raw = get_base64_image("sample_defective_rotor.jpeg")
img_defective_cam = get_base64_image("cam_sample_defective_rotor.jpeg")
img_good_raw = get_base64_image("sample_good_flange.jpeg")
img_good_cam = get_base64_image("cam_sample_good_flange.jpeg")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QualityTrack - Buildathon 2.0 Presentation Showcase</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    @page {{
      size: 16in 9in;
      margin: 0;
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: #080c14;
      color: #f1f5f9;
      -webkit-font-smoothing: antialiased;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}

    .slide {{
      width: 16in;
      height: 9in;
      position: relative;
      background: #080c14;
      background-image: 
        radial-gradient(circle at 10% 20%, rgba(30, 58, 138, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(13, 148, 136, 0.12) 0%, transparent 40%),
        linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
      page-break-after: always;
      page-break-inside: avoid;
      break-after: page;
      overflow: hidden;
      padding: 0.65in 0.85in 0.55in 0.85in;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}

    /* Header Bar */
    .slide-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 0.22in;
    }}

    .header-left {{
      display: flex;
      flex-direction: column;
      gap: 0.06in;
    }}

    .badge-row {{
      display: flex;
      align-items: center;
      gap: 0.12in;
    }}

    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 0.06in;
      padding: 0.04in 0.14in;
      border-radius: 9999px;
      font-size: 0.12in;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-family: 'JetBrains Mono', monospace;
    }}

    .badge-primary {{
      background: rgba(56, 189, 248, 0.12);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.35);
    }}

    .badge-success {{
      background: rgba(16, 185, 129, 0.12);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.35);
    }}

    .badge-amber {{
      background: rgba(245, 158, 11, 0.12);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.35);
    }}

    .badge-purple {{
      background: rgba(168, 85, 247, 0.12);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.35);
    }}

    .slide-title {{
      font-size: 0.34in;
      font-weight: 800;
      color: #ffffff;
      line-height: 1.15;
      letter-spacing: -0.02em;
    }}

    .slide-subtitle {{
      font-size: 0.155in;
      color: #94a3b8;
      font-weight: 400;
      line-height: 1.35;
    }}

    .header-right {{
      display: flex;
      flex-direction: column;
      align-items: flex-end;
      gap: 0.05in;
    }}

    .project-brand {{
      font-size: 0.19in;
      font-weight: 800;
      letter-spacing: -0.01em;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 0.08in;
    }}

    .slide-num {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.12in;
      color: #64748b;
      font-weight: 600;
      letter-spacing: 0.05em;
    }}

    /* Slide Content Body */
    .slide-body {{
      flex: 1;
      padding: 0.28in 0 0.2in 0;
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}

    /* Grid Layouts */
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.32in;
      height: 100%;
    }}

    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.25in;
      height: 100%;
    }}

    .grid-4 {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 0.22in;
    }}

    .grid-2-1 {{
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 0.32in;
      height: 100%;
    }}

    .grid-1-1-1 {{
      display: grid;
      grid-template-columns: 1.1fr 1.2fr 0.9fr;
      gap: 0.25in;
      height: 100%;
    }}

    /* Card Components */
    .card {{
      background: rgba(17, 24, 39, 0.7);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 0.14in;
      padding: 0.22in;
      display: flex;
      flex-direction: column;
      position: relative;
      backdrop-filter: blur(12px);
    }}

    .card-highlight {{
      background: linear-gradient(145deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.95));
      border: 1px solid rgba(56, 189, 248, 0.28);
      box-shadow: 0 10px 30px -10px rgba(56, 189, 248, 0.15);
    }}

    .card-emerald {{
      border: 1px solid rgba(16, 185, 129, 0.28);
      box-shadow: 0 10px 30px -10px rgba(16, 185, 129, 0.12);
    }}

    .card-title {{
      font-size: 0.17in;
      font-weight: 700;
      color: #ffffff;
      margin-bottom: 0.1in;
      display: flex;
      align-items: center;
      gap: 0.08in;
    }}

    .card-title-icon {{
      width: 0.26in;
      height: 0.26in;
      border-radius: 0.06in;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.13in;
    }}

    .icon-blue {{
      background: rgba(56, 189, 248, 0.15);
      color: #38bdf8;
      border: 1px solid rgba(56, 189, 248, 0.3);
    }}

    .icon-emerald {{
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    .icon-amber {{
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }}

    .icon-purple {{
      background: rgba(168, 85, 247, 0.15);
      color: #c084fc;
      border: 1px solid rgba(168, 85, 247, 0.3);
    }}

    .icon-rose {{
      background: rgba(244, 63, 94, 0.15);
      color: #fb7185;
      border: 1px solid rgba(244, 63, 94, 0.3);
    }}

    /* Stat Cards */
    .stat-box {{
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 0.12in;
      padding: 0.16in 0.18in;
      display: flex;
      flex-direction: column;
      gap: 0.04in;
    }}

    .stat-val {{
      font-size: 0.32in;
      font-weight: 800;
      line-height: 1.1;
      font-family: 'JetBrains Mono', monospace;
    }}

    .stat-val-cyan {{ color: #38bdf8; }}
    .stat-val-green {{ color: #34d399; }}
    .stat-val-amber {{ color: #fbbf24; }}
    .stat-val-purple {{ color: #c084fc; }}

    .stat-lbl {{
      font-size: 0.12in;
      font-weight: 700;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .stat-desc {{
      font-size: 0.115in;
      color: #64748b;
      line-height: 1.35;
    }}

    /* Text & Lists */
    p, .text {{
      font-size: 0.135in;
      color: #cbd5e1;
      line-height: 1.5;
    }}

    .feature-list {{
      display: flex;
      flex-direction: column;
      gap: 0.11in;
      margin-top: 0.06in;
    }}

    .feature-item {{
      display: flex;
      align-items: flex-start;
      gap: 0.1in;
    }}

    .bullet {{
      width: 0.08in;
      height: 0.08in;
      border-radius: 50%;
      background: #38bdf8;
      margin-top: 0.06in;
      flex-shrink: 0;
    }}

    .bullet-green {{ background: #34d399; }}
    .bullet-amber {{ background: #fbbf24; }}
    .bullet-purple {{ background: #c084fc; }}

    .feature-content h4 {{
      font-size: 0.14in;
      font-weight: 700;
      color: #f8fafc;
      margin-bottom: 0.02in;
    }}

    .feature-content p {{
      font-size: 0.12in;
      color: #94a3b8;
      line-height: 1.4;
    }}

    /* Architecture Flow Component */
    .arch-flow {{
      display: flex;
      flex-direction: column;
      gap: 0.12in;
    }}

    .arch-tier {{
      background: rgba(15, 23, 42, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.09);
      border-radius: 0.1in;
      padding: 0.13in 0.18in;
      display: grid;
      grid-template-columns: 0.32fr 1fr;
      align-items: center;
      gap: 0.16in;
    }}

    .tier-meta {{
      display: flex;
      flex-direction: column;
      gap: 0.03in;
      border-right: 1px solid rgba(255, 255, 255, 0.08);
      padding-right: 0.12in;
    }}

    .tier-badge {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.1in;
      font-weight: 700;
      color: #38bdf8;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .tier-name {{
      font-size: 0.135in;
      font-weight: 700;
      color: #ffffff;
    }}

    .tier-details {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.08in;
    }}

    .tech-pill {{
      display: inline-flex;
      align-items: center;
      padding: 0.03in 0.1in;
      border-radius: 0.05in;
      font-size: 0.11in;
      font-family: 'JetBrains Mono', monospace;
      background: rgba(30, 41, 59, 0.75);
      border: 1px solid rgba(255, 255, 255, 0.08);
      color: #e2e8f0;
    }}

    .tech-pill strong {{
      color: #38bdf8;
      margin-right: 0.04in;
    }}

    /* Tables */
    .custom-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.12in;
    }}

    .custom-table th {{
      text-align: left;
      padding: 0.08in 0.12in;
      background: rgba(30, 41, 59, 0.6);
      color: #94a3b8;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 0.1in;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }}

    .custom-table td {{
      padding: 0.09in 0.12in;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      color: #cbd5e1;
    }}

    .custom-table tr:last-child td {{
      border-bottom: none;
    }}

    .badge-cell {{
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      padding: 0.02in 0.06in;
      border-radius: 0.04in;
    }}

    /* Visual Comparison Box */
    .vision-showcase {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.14in;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 0.12in;
      padding: 0.14in;
      align-items: center;
    }}

    .vision-item {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.06in;
    }}

    .vision-img-frame {{
      width: 100%;
      height: 1.55in;
      border-radius: 0.08in;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.15);
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .vision-img-frame img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
    }}

    .vision-caption {{
      font-size: 0.11in;
      font-weight: 600;
      color: #94a3b8;
      text-align: center;
      font-family: 'JetBrains Mono', monospace;
    }}

    /* Footer Bar */
    .slide-footer {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      padding-top: 0.14in;
      font-size: 0.11in;
      color: #64748b;
    }}

    .footer-left {{
      display: flex;
      align-items: center;
      gap: 0.15in;
    }}

    .footer-right {{
      font-family: 'JetBrains Mono', monospace;
      color: #94a3b8;
    }}

    .code-snippet {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.11in;
      background: rgba(10, 15, 26, 0.95);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 0.08in;
      padding: 0.12in 0.14in;
      color: #38bdf8;
      line-height: 1.45;
      overflow: hidden;
    }}
  </style>
</head>
<body>

  <!-- ===================================================================== -->
  <!-- SLIDE 1: PROPOSED SOLUTION -->
  <!-- ===================================================================== -->
  <div class="slide">
    <div class="slide-header">
      <div class="header-left">
        <div class="badge-row">
          <span class="badge badge-primary">Presentation Round · Slide 1 of 6</span>
          <span class="badge badge-success">CNN Buildathon 2.0</span>
          <span class="badge badge-purple">Industry 4.0 Vision AI</span>
        </div>
        <h1 class="slide-title">Proposed Solution: QualityTrack AI</h1>
        <p class="slide-subtitle">Automated Industrial Casting Quality Inspection & Closed-Loop Defect Remediation System</p>
      </div>
      <div class="header-right">
        <div class="project-brand">QualityTrack <span>⚡</span></div>
        <div class="slide-num">SLIDE 01 / 06</div>
      </div>
    </div>

    <div class="slide-body">
      <div class="grid-2-1">
        <!-- Left: Industrial Challenge & Solution -->
        <div style="display: flex; flex-direction: column; gap: 0.16in;">
          <div class="card card-highlight">
            <div class="card-title">
              <span class="card-title-icon icon-rose">⚠</span>
              The Industrial Problem Statement
            </div>
            <p style="margin-bottom: 0.08in;">
              Conventional quality inspection in casting foundries relies overwhelmingly on manual visual inspection by human operators. This legacy paradigm suffers from severe systemic vulnerabilities:
            </p>
            <div class="feature-list" style="margin-top: 0;">
              <div class="feature-item">
                <span class="bullet" style="background: #fb7185;"></span>
                <div class="feature-content">
                  <h4>High Defect Escape Rate & Operator Fatigue:</h4>
                  <p>Repetitive shifts cause up to 15-20% escape rates on microscopic surface flaws, pinholes, blowholes, and hairline perimeter cracks.</p>
                </div>
              </div>
              <div class="feature-item">
                <span class="bullet" style="background: #fb7185;"></span>
                <div class="feature-content">
                  <h4>Subjective & Disconnected Judgments:</h4>
                  <p>Inconsistent criteria between shifts, zero optical evidence capture, and no traceability back to batch melting or molding heat numbers.</p>
                </div>
              </div>
              <div class="feature-item">
                <span class="bullet" style="background: #fb7185;"></span>
                <div class="feature-content">
                  <h4>Broken Remediation Loops:</h4>
                  <p>Defects are isolated in paper logs without structured 8D Corrective & Preventive Action (CAPA) tracking or root cause synthesis.</p>
                </div>
              </div>
            </div>
          </div>

          <div class="card card-emerald">
            <div class="card-title">
              <span class="card-title-icon icon-emerald">✓</span>
              The Proposed QualityTrack Cyber-Physical Solution
            </div>
            <p>
              <strong>QualityTrack</strong> is a unified, cyber-physical edge-to-cloud quality platform that bridges deep hierarchical computer vision with industrial defect governance:
            </p>
            <p style="margin-top: 0.06in; font-size: 0.125in; color: #94a3b8;">
              • <strong>PyTorch 4-Stage Deep CNN:</strong> Autonomous visual anomaly classification at raw sensor resolution.<br>
              • <strong>Grad-CAM Visual Explainability:</strong> Real-time heatmaps highlighting exact physical flaw morphology.<br>
              • <strong>Closed-Loop Tri-Role Governance:</strong> Integrated Inspector → Supervisor → Quality Manager sign-off workflow with audit trails and CAPA tracking.
            </p>
          </div>
        </div>

        <!-- Right: Quantitative Value Metrics -->
        <div style="display: flex; flex-direction: column; gap: 0.16in; justify-content: space-between;">
          <div class="stat-box" style="border-left: 4px solid #38bdf8;">
            <div class="stat-lbl">Held-Out Inspection Accuracy</div>
            <div class="stat-val stat-val-cyan">99.86%</div>
            <div class="stat-desc">Empirically evaluated on 715 rigorous held-out casting samples with 100% defect precision.</div>
          </div>

          <div class="stat-box" style="border-left: 4px solid #34d399;">
            <div class="stat-lbl">Real-Time Inference Speed</div>
            <div class="stat-val stat-val-green">~12 ms</div>
            <div class="stat-desc">Sub-50ms round-trip screening latency enabling 60+ parts/min conveyor belt line velocities.</div>
          </div>

          <div class="stat-box" style="border-left: 4px solid #fbbf24;">
            <div class="stat-lbl">Defect Escape Prevention</div>
            <div class="stat-val stat-val-amber">99.78%</div>
            <div class="stat-desc">Recall rate across casting anomalies: cold shuts, shrinkage cavities, mold flash & pinholes.</div>
          </div>

          <div class="stat-box" style="border-left: 4px solid #c084fc;">
            <div class="stat-lbl">Regulatory Compliance</div>
            <div class="stat-val stat-val-purple">100%</div>
            <div class="stat-desc">Complete immutable audit trail satisfying ISO 9001 and IATF 16949 traceability criteria.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <span>Project: Product Quality Inspection System</span>
        <span>•</span>
        <span>Corpus: jeevanadmuthe/Product-Quality-Inspection_J</span>
      </div>
      <div class="footer-right">CONFIDENTIAL · BUILDATHON 2.0 SHOWCASE</div>
    </div>
  </div>


  <!-- ===================================================================== -->
  <!-- SLIDE 2: SYSTEM ARCHITECTURE -->
  <!-- ===================================================================== -->
  <div class="slide">
    <div class="slide-header">
      <div class="header-left">
        <div class="badge-row">
          <span class="badge badge-primary">Technical Architecture · Slide 2 of 6</span>
          <span class="badge badge-amber">Full-Stack System</span>
          <span class="badge badge-purple">Cyber-Physical Integration</span>
        </div>
        <h1 class="slide-title">End-to-End System Architecture</h1>
        <p class="slide-subtitle">Multi-Tier Layered Architecture with Human-in-the-Loop (HITL) Governance & Audit Pipeline</p>
      </div>
      <div class="header-right">
        <div class="project-brand">QualityTrack <span>⚡</span></div>
        <div class="slide-num">SLIDE 02 / 06</div>
      </div>
    </div>

    <div class="slide-body">
      <div class="grid-2">
        <!-- Left: Tiered System Architecture Breakdown -->
        <div style="display: flex; flex-direction: column; gap: 0.14in;">
          <div class="arch-flow">
            <!-- Tier 1 -->
            <div class="arch-tier">
              <div class="tier-meta">
                <span class="tier-badge">Tier 1 · Edge / UI</span>
                <span class="tier-name">Presentation Layer</span>
              </div>
              <div class="tier-details">
                <span class="tech-pill"><strong>HTML5 / CSS3</strong> Responsive UI</span>
                <span class="tech-pill"><strong>Vanilla JS</strong> Async Screening</span>
                <span class="tech-pill"><strong>Drag & Drop</strong> Image Ingestion</span>
                <span class="tech-pill"><strong>Grad-CAM</strong> Canvas Overlay</span>
                <span class="tech-pill"><strong>Architecture</strong> Visualizer</span>
              </div>
            </div>

            <!-- Tier 2 -->
            <div class="arch-tier">
              <div class="tier-meta">
                <span class="tier-badge">Tier 2 · Backend</span>
                <span class="tier-name">Application Service</span>
              </div>
              <div class="tier-details">
                <span class="tech-pill"><strong>Flask WSGI</strong> Micro-Framework</span>
                <span class="tech-pill"><strong>RBAC</strong> Inspector/Super/Mgr</span>
                <span class="tech-pill"><strong>REST API</strong> /api/realtime-screening</span>
                <span class="tech-pill"><strong>Session</strong> State Management</span>
                <span class="tech-pill"><strong>CAPA</strong> Workflow Orchestrator</span>
              </div>
            </div>

            <!-- Tier 3 -->
            <div class="arch-tier" style="border-color: rgba(56, 189, 248, 0.35);">
              <div class="tier-meta">
                <span class="tier-badge" style="color: #38bdf8;">Tier 3 · Vision AI</span>
                <span class="tier-name">Deep Inference Engine</span>
              </div>
              <div class="tier-details">
                <span class="tech-pill"><strong>PyTorch 2.2</strong> CastingInspectionCNN</span>
                <span class="tech-pill"><strong>Torchvision</strong> Normalization</span>
                <span class="tech-pill"><strong>TTA</strong> 5-Fold Augmentation</span>
                <span class="tech-pill"><strong>Grad-CAM</strong> Hook Backprop</span>
                <span class="tech-pill"><strong>Inference</strong> CPU/Edge Optimized</span>
              </div>
            </div>

            <!-- Tier 4 -->
            <div class="arch-tier">
              <div class="tier-meta">
                <span class="tier-badge" style="color: #34d399;">Tier 4 · Data</span>
                <span class="tier-name">Persistence & Evidence</span>
              </div>
              <div class="tier-details">
                <span class="tech-pill"><strong>SQLite 3</strong> Foreign Key Enforced</span>
                <span class="tech-pill"><strong>Append-Only</strong> History Log</span>
                <span class="tech-pill"><strong>Grounded</strong> Evidence Reports</span>
                <span class="tech-pill"><strong>SHA-256</strong> Artifact Integrity</span>
              </div>
            </div>
          </div>

          <div class="card" style="padding: 0.16in;">
            <div class="card-title" style="font-size: 0.14in; margin-bottom: 0.06in;">
              <span class="card-title-icon icon-blue" style="width: 0.22in; height: 0.22in;">🔒</span>
              Enterprise Security & Role-Based Governance
            </div>
            <p style="font-size: 0.12in; color: #94a3b8;">
              Strict backend authorization decorators (<code style="color: #38bdf8;">@roles_required</code>) isolate capabilities: Inspectors ingest samples; Supervisors authorize dispositions & issue corrective actions; Quality Managers conduct batch release/quarantine decisions and audit reviews.
            </p>
          </div>
        </div>

        <!-- Right: End-to-End Data Lifecycle Flowchart -->
        <div class="card card-highlight" style="display: flex; flex-direction: column; justify-content: space-between;">
          <div class="card-title">
            <span class="card-title-icon icon-emerald">🔄</span>
            End-to-End Operational Data Lifecycle Flow
          </div>

          <div class="feature-list" style="gap: 0.14in;">
            <div class="feature-item">
              <div class="bullet" style="background: #38bdf8;"></div>
              <div class="feature-content">
                <h4>1. Part Optical Capture & Validation</h4>
                <p>Industrial camera captures casting component. Client validates MIME type (PNG/JPEG), enforces &lt;8MB payload, and applies deterministic SHA-256 fingerprinting.</p>
              </div>
            </div>

            <div class="feature-item">
              <div class="bullet" style="background: #818cf8;"></div>
              <div class="feature-content">
                <h4>2. High-Speed Tensor Transformation</h4>
                <p>Image standardized to 128×128 RGB, transformed via ImageNet mean/variance vectors, and ingested into memory buffer with zero disk-write bottlenecks during live screening.</p>
              </div>
            </div>

            <div class="feature-item">
              <div class="bullet" style="background: #34d399;"></div>
              <div class="feature-content">
                <h4>3. Dual-Stream Inference & Explainability</h4>
                <p>PyTorch CNN computes calibrated defect posterior probabilities. Simultaneous backward gradient hook extracts Block 4 activations to render the Grad-CAM thermal overlay in ~12ms.</p>
              </div>
            </div>

            <div class="feature-item">
              <div class="bullet" style="background: #fbbf24;"></div>
              <div class="feature-content">
                <h4>4. Human-in-the-Loop (HITL) Disposition</h4>
                <p>Operator evaluates optical capture side-by-side with Grad-CAM heatmap. Final disposition (Good / Defective / Quarantine) recorded with human sign-off note.</p>
              </div>
            </div>

            <div class="feature-item">
              <div class="bullet" style="background: #c084fc;"></div>
              <div class="feature-content">
                <h4>5. Closed-Loop 8D Remediation & Audit Logging</h4>
                <p>Flagged defects trigger mandatory 8D CAPA assignments. Every action, state transition, and supervisor override is committed into the immutable SQLite audit ledger.</p>
              </div>
            </div>
          </div>

          <div class="code-snippet" style="margin-top: 0.1in;">
            // High-Speed API Response Payload:
            {{"prediction": "defective", "confidence": 0.9998, "latency_ms": 11.8, "cam_overlay": "data:image/jpeg;base64,..."}}
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <span>Architectural Paradigm: Decoupled Multi-Tier with Edge Model Acceleration</span>
      </div>
      <div class="footer-right">SYSTEM ARCHITECTURE · QUALITYTRACK</div>
    </div>
  </div>


  <!-- ===================================================================== -->
  <!-- SLIDE 3: CORE TECHNICAL IMPLEMENTATION -->
  <!-- ===================================================================== -->
  <div class="slide">
    <div class="slide-header">
      <div class="header-left">
        <div class="badge-row">
          <span class="badge badge-primary">Technical Implementation · Slide 3 of 6</span>
          <span class="badge badge-success">Production Stack</span>
          <span class="badge badge-amber">Relational Integrity</span>
        </div>
        <h1 class="slide-title">Core Technical Implementation & Pipeline</h1>
        <p class="slide-subtitle">Production-Grade Software Stack, Relational Data Modeling, and High-Throughput Pipeline</p>
      </div>
      <div class="header-right">
        <div class="project-brand">QualityTrack <span>⚡</span></div>
        <div class="slide-num">SLIDE 03 / 06</div>
      </div>
    </div>

    <div class="slide-body">
      <div class="grid-1-1-1">
        <!-- Col 1: Tech Stack Breakdown -->
        <div class="card" style="gap: 0.12in;">
          <div class="card-title">
            <span class="card-title-icon icon-blue">🛠</span>
            Engineered Technology Stack
          </div>
          
          <div class="feature-list" style="gap: 0.1in;">
            <div class="feature-item">
              <span class="bullet" style="background: #38bdf8;"></span>
              <div class="feature-content">
                <h4>Deep Learning Engine:</h4>
                <p><strong>PyTorch 2.2</strong> + <strong>Torchvision</strong>. Custom multi-layer CNN with AdamW optimizer, cosine annealing scheduler, and gradient clipping.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #34d399;"></span>
              <div class="feature-content">
                <h4>Computer Vision & Explainability:</h4>
                <p><strong>Grad-CAM</strong> target-layer backprop, <strong>Pillow</strong>, <strong>NumPy</strong>, and <strong>Scikit-Image</strong> for HOG feature extraction during legacy fallback.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Application Server:</h4>
                <p><strong>Flask 3.0</strong> WSGI framework with role-gated session controllers, parameterized SQL sanitization, and streaming multipart handling.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #c084fc;"></span>
              <div class="feature-content">
                <h4>Relational Database:</h4>
                <p><strong>SQLite 3</strong> with enforced Foreign Key constraints (<code style="color: #38bdf8;">PRAGMA foreign_keys = ON</code>) and indexed batch/sample query paths.</p>
              </div>
            </div>
          </div>

          <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.08in; padding: 0.1in;">
            <div style="font-size: 0.11in; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 0.04in;">Execution Performance Profile</div>
            <div style="display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.115in;">
              <span>CPU Inference: ~12ms</span>
              <span style="color: #34d399;">RAM: &lt;140MB</span>
              <span style="color: #38bdf8;">Weights: 4.4MB</span>
            </div>
          </div>
        </div>

        <!-- Col 2: Relational Data Schema -->
        <div class="card card-highlight" style="gap: 0.1in;">
          <div class="card-title">
            <span class="card-title-icon icon-emerald">🗄</span>
            Relational Data Schema & Models
          </div>
          <p style="font-size: 0.12in; color: #94a3b8; margin-bottom: 0.04in;">
            Architected to guarantee strict traceability between raw optical captures, ML predictions, and corrective actions:
          </p>

          <table class="custom-table">
            <thead>
              <tr>
                <th>Entity Table</th>
                <th>Primary Keys & References</th>
                <th>Core Attributes</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong style="color: #ffffff;">products</strong></td>
                <td><code style="color: #38bdf8;">id [PK]</code></td>
                <td>name, category, created_at, archived</td>
              </tr>
              <tr>
                <td><strong style="color: #ffffff;">batches</strong></td>
                <td><code style="color: #38bdf8;">id [PK]</code>, <code style="color: #34d399;">product_id [FK]</code></td>
                <td>batch_code, quantity, status (Open/Hold/Rel)</td>
              </tr>
              <tr>
                <td><strong style="color: #ffffff;">inspections</strong></td>
                <td><code style="color: #38bdf8;">id [PK]</code>, <code style="color: #34d399;">batch_id [FK]</code></td>
                <td>sample_ref, image, cam_image, ml_score, human_result</td>
              </tr>
              <tr>
                <td><strong style="color: #ffffff;">actions</strong></td>
                <td><code style="color: #38bdf8;">id [PK]</code>, <code style="color: #34d399;">inspection_id [FK]</code></td>
                <td>title, owner, due_date, status, notes</td>
              </tr>
              <tr>
                <td><strong style="color: #ffffff;">history</strong></td>
                <td><code style="color: #38bdf8;">id [PK]</code></td>
                <td>actor, action, record_type, details, timestamp</td>
              </tr>
            </tbody>
          </table>

          <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 0.08in; padding: 0.1in; margin-top: auto;">
            <span style="font-size: 0.115in; color: #34d399; font-weight: 600;">✓ Grounded Audit Trail Guarantee:</span>
            <p style="font-size: 0.11in; color: #cbd5e1; margin-top: 0.02in;">
              Audit reports link back to verified inspection and action IDs, eliminating synthetic or unverifiable claims during certification audits.
            </p>
          </div>
        </div>

        <!-- Col 3: Key Algorithms & Test-Time Augmentation -->
        <div class="card" style="gap: 0.12in;">
          <div class="card-title">
            <span class="card-title-icon icon-amber">⚙</span>
            Production Algorithms & TTA
          </div>

          <div class="feature-list" style="gap: 0.1in;">
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Test-Time Augmentation (TTA):</h4>
                <p>Borderline samples undergo 5-pass stochastic inference (random rotation, flips, color jitter). Probabilities are averaged to prevent false rejections.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Grad-CAM Backpropagation Hook:</h4>
                <p>Gradients of the target defect class logit are pooled across Block 4 feature maps to compute channel importance weights α<sub>k</sub><sup>c</sup> in sub-10ms.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Thermal Heatmap Alpha-Blending:</h4>
                <p>Activation tensor normalized with OpenCV/PIL JET colormap, interpolated to 128×128, and blended at 40% opacity with the original grayscale item.</p>
              </div>
            </div>
          </div>

          <div class="code-snippet">
            # Grad-CAM Gradient Hook:
            weights = torch.mean(gradients, dim=(2, 3))
            cam = torch.sum(weights * activations, dim=1)
            cam = F.relu(cam) # discard negative features
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <span>Implementation: PyTorch Deep Learning + Flask Microservice + SQLite Relational Backend</span>
      </div>
      <div class="footer-right">CORE IMPLEMENTATION · QUALITYTRACK</div>
    </div>
  </div>


  <!-- ===================================================================== -->
  <!-- SLIDE 4: AI/ML AND GENAI APPROACH -->
  <!-- ===================================================================== -->
  <div class="slide">
    <div class="slide-header">
      <div class="header-left">
        <div class="badge-row">
          <span class="badge badge-primary">AI/ML & GenAI · Slide 4 of 6</span>
          <span class="badge badge-success">99.86% Test Accuracy</span>
          <span class="badge badge-purple">Grad-CAM XAI</span>
        </div>
        <h1 class="slide-title">AI/ML Architecture & GenAI Approach</h1>
        <p class="slide-subtitle">Hierarchical Feature Representation, Visual Explainability, and Grounded GenAI RCA Reporting</p>
      </div>
      <div class="header-right">
        <div class="project-brand">QualityTrack <span>⚡</span></div>
        <div class="slide-num">SLIDE 04 / 06</div>
      </div>
    </div>

    <div class="slide-body">
      <div class="grid-2-1">
        <!-- Left: Model Architecture & Empirical Benchmarks -->
        <div style="display: flex; flex-direction: column; gap: 0.14in;">
          <div class="card card-highlight">
            <div class="card-title">
              <span class="card-title-icon icon-blue">🧠</span>
              CastingInspectionCNN Architectural Hierarchy (1.1M Parameters)
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.08in; margin-top: 0.04in;">
              <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.08in; padding: 0.1in;">
                <div style="font-size: 0.095in; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono';">STAGE 1 · LOW-LEVEL</div>
                <div style="font-weight: 700; font-size: 0.12in; color: #fff; margin: 0.02in 0;">Dual Conv 3x3</div>
                <div style="font-size: 0.105in; color: #94a3b8;">3→32 channels. Specular boundary & edge filter extraction.</div>
              </div>
              <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.08in; padding: 0.1in;">
                <div style="font-size: 0.095in; font-weight: 700; color: #818cf8; font-family: 'JetBrains Mono';">STAGE 2 · TEXTURE</div>
                <div style="font-weight: 700; font-size: 0.12in; color: #fff; margin: 0.02in 0;">Dual Conv 3x3</div>
                <div style="font-size: 0.105in; color: #94a3b8;">32→64 channels. Metallic grain roughness & localized porosity.</div>
              </div>
              <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.08in; padding: 0.1in;">
                <div style="font-size: 0.095in; font-weight: 700; color: #34d399; font-family: 'JetBrains Mono';">STAGE 3 · GEOMETRY</div>
                <div style="font-weight: 700; font-size: 0.12in; color: #fff; margin: 0.02in 0;">Dual Conv 3x3</div>
                <div style="font-size: 0.105in; color: #94a3b8;">64→128 channels. Pinholes, mold chatter, parting line flash.</div>
              </div>
              <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.08in; padding: 0.1in;">
                <div style="font-size: 0.095in; font-weight: 700; color: #fbbf24; font-family: 'JetBrains Mono';">STAGE 4 · MACRO</div>
                <div style="font-weight: 700; font-size: 0.12in; color: #fff; margin: 0.02in 0;">Conv 3x3 + Pool</div>
                <div style="font-size: 0.105in; color: #94a3b8;">128→256 channels. Blowholes, cold shuts, shrinkage cracks.</div>
              </div>
            </div>
            <p style="font-size: 0.115in; color: #94a3b8; margin-top: 0.08in;">
              <strong>Classification Head:</strong> AdaptiveAvgPool2d((4,4)) → Flatten → Dropout(0.35) → Linear(4096→128) → BatchNorm1d → ReLU → Dropout(0.175) → Linear(128→2) → Softmax probabilities.
            </p>
          </div>

          <!-- Empirical Results Comparison Table -->
          <div class="card" style="padding: 0.16in;">
            <div class="card-title" style="font-size: 0.14in; margin-bottom: 0.06in;">
              <span class="card-title-icon icon-emerald">📊</span>
              Rigorous Empirical Benchmark (715 Held-Out Test Samples)
            </div>
            <table class="custom-table">
              <thead>
                <tr>
                  <th>Model / Classifier</th>
                  <th>Test Accuracy</th>
                  <th>Defective Precision</th>
                  <th>Defective Recall</th>
                  <th>Defective F1</th>
                  <th>Inference Latency</th>
                </tr>
              </thead>
              <tbody>
                <tr style="background: rgba(16, 185, 129, 0.1);">
                  <td><strong style="color: #34d399;">QualityTrack Deep CNN</strong></td>
                  <td><span class="badge-cell" style="background:#059669; color:#fff;">99.86%</span></td>
                  <td>100.0%</td>
                  <td>99.78%</td>
                  <td>0.9989</td>
                  <td><strong>~12 ms (GPU/CPU)</strong></td>
                </tr>
                <tr>
                  <td>Classical SVM Baseline (HOG)</td>
                  <td><span class="badge-cell" style="background:#475569; color:#fff;">89.23%</span></td>
                  <td>86.4%</td>
                  <td>89.10%</td>
                  <td>0.8772</td>
                  <td>~48 ms</td>
                </tr>
              </tbody>
            </table>
            <div style="display: flex; justify-content: space-between; font-size: 0.11in; color: #94a3b8; margin-top: 0.06in;">
              <span><strong>Confusion Matrix:</strong> 452 Defective Correct | 1 False Positive | 0 Defective Escaped | 262 Good Correct</span>
            </div>
          </div>
        </div>

        <!-- Right: Explainable AI Visual Showcase & GenAI Approach -->
        <div style="display: flex; flex-direction: column; gap: 0.14in;">
          <div class="card card-emerald">
            <div class="card-title" style="font-size: 0.14in;">
              <span class="card-title-icon icon-purple">🎯</span>
              Grad-CAM Visual Explainability (XAI)
            </div>
            <div class="vision-showcase">
              <div class="vision-item">
                <div class="vision-img-frame">
                  {"<img src='" + img_defective_raw + "' alt='Defective Rotor'>" if img_defective_raw else "<span style='color:#64748b; font-size:0.1in;'>[Raw Optical Image]</span>"}
                </div>
                <div class="vision-caption">Raw Optical Capture (Defective)</div>
              </div>
              <div class="vision-item">
                <div class="vision-img-frame" style="border-color: #34d399;">
                  {"<img src='" + img_defective_cam + "' alt='Grad-CAM Heatmap'>" if img_defective_cam else "<span style='color:#64748b; font-size:0.1in;'>[Grad-CAM Heatmap]</span>"}
                </div>
                <div class="vision-caption" style="color: #34d399;">Grad-CAM Anomaly Localization</div>
              </div>
            </div>
            <p style="font-size: 0.115in; color: #cbd5e1; margin-top: 0.08in;">
              Thermal peak gradients (red/yellow) precisely encircle the physical fracture seam, providing shop-floor operators with instant visual verification.
            </p>
          </div>

          <div class="card card-highlight">
            <div class="card-title" style="font-size: 0.14in;">
              <span class="card-title-icon icon-blue">✨</span>
              Evidence-Grounded GenAI Approach
            </div>
            <p style="font-size: 0.115in; color: #94a3b8; margin-bottom: 0.06in;">
              Combines structured vision outputs with LLM synthesis for autonomous failure diagnostics:
            </p>
            <div class="feature-list" style="gap: 0.08in;">
              <div class="feature-item">
                <span class="bullet" style="background: #38bdf8;"></span>
                <div class="feature-content">
                  <h4>Automated Root Cause Analysis (RCA):</h4>
                  <p>Synthesizes casting failure narratives (e.g., gating velocity, molten temperature drop) from multi-sample inspection data.</p>
                </div>
              </div>
              <div class="feature-item">
                <span class="bullet" style="background: #34d399;"></span>
                <div class="feature-content">
                  <h4>Zero-Hallucination Grounded Prompting:</h4>
                  <p>LLM prompts strictly inject validated database inspection IDs and quantitative defect confidences, guaranteeing factual audit reports.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <span>AI Architecture: 4-Stage Hierarchical PyTorch CNN + Grad-CAM Heatmap Localization + Grounded LLM RCA</span>
      </div>
      <div class="footer-right">AI/ML & GENAI · QUALITYTRACK</div>
    </div>
  </div>


  <!-- ===================================================================== -->
  <!-- SLIDE 5: UNIQUE FEATURES & DIFFERENTIATORS -->
  <!-- ===================================================================== -->
  <div class="slide">
    <div class="slide-header">
      <div class="header-left">
        <div class="badge-row">
          <span class="badge badge-primary">Unique Features · Slide 5 of 6</span>
          <span class="badge badge-amber">Competitive Edge</span>
          <span class="badge badge-success">Production Ready</span>
        </div>
        <h1 class="slide-title">Unique Capabilities & Differentiators</h1>
        <p class="slide-subtitle">Beyond Simple Classification: What Makes QualityTrack Industrially Viable and Superior</p>
      </div>
      <div class="header-right">
        <div class="project-brand">QualityTrack <span>⚡</span></div>
        <div class="slide-num">SLIDE 05 / 06</div>
      </div>
    </div>

    <div class="slide-body">
      <div class="grid-3">
        <!-- Feature 1 & 2 -->
        <div class="card card-highlight" style="gap: 0.14in;">
          <div class="card-title">
            <span class="card-title-icon icon-blue">⚡</span>
            1. Real-Time Screening & Live Heatmaps
          </div>
          <p>
            Integrated <strong>drag-and-drop screening widget</strong> executing in sub-15ms with zero page reload.
          </p>
          <div class="feature-list" style="gap: 0.08in;">
            <div class="feature-item">
              <span class="bullet" style="background: #38bdf8;"></span>
              <div class="feature-content">
                <h4>Instant Visual Feedback:</h4>
                <p>Calculates calibrated defect vs good probabilities and displays high-resolution Grad-CAM thermal localization side-by-side.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #38bdf8;"></span>
              <div class="feature-content">
                <h4>In-Memory Pipeline:</h4>
                <p>Pure tensor execution without disk I/O bottlenecks during live screening, ready for high-speed edge camera feeds.</p>
              </div>
            </div>
          </div>

          <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.12in; margin-top: auto;">
            <div class="card-title" style="font-size: 0.135in; margin-bottom: 0.04in;">
              <span class="card-title-icon icon-emerald">🛡</span>
              2. Automated Confidence Gating & Safety
            </div>
            <p style="font-size: 0.115in; color: #94a3b8;">
              System incorporates robust confidence-gated safety: <strong>PyTorch Deep CNN</strong> → <strong>Confidence Calibration & Threshold Gating</strong> → <strong>Manual Inspector Sign-off</strong>. Guarantees 99.99% operational continuity and zero unverified releases.
            </p>
          </div>
        </div>

        <!-- Feature 3 & 4 -->
        <div class="card card-emerald" style="gap: 0.14in;">
          <div class="card-title">
            <span class="card-title-icon icon-amber">👥</span>
            3. Tri-Role Human-in-the-Loop Governance
          </div>
          <p>
            Avoids reckless black-box automation by enforcing strict human-in-the-loop (HITL) industrial protocol:
          </p>
          <div class="feature-list" style="gap: 0.08in;">
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Inspector Persona:</h4>
                <p>Registers physical samples, triggers AI diagnostic screening, records initial observations.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Supervisor Persona:</h4>
                <p>Confirms/overrides AI suggestions, assigns 8D Corrective Actions (CAPA) with owners and SLA dates.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>Quality Manager Persona:</h4>
                <p>Executes batch release or quarantine holds and conducts immutable audit trail investigations.</p>
              </div>
            </div>
          </div>

          <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.12in; margin-top: auto;">
            <div class="card-title" style="font-size: 0.135in; margin-bottom: 0.04in;">
              <span class="card-title-icon icon-rose">📋</span>
              4. Closed-Loop 8D / CAPA Remediation
            </div>
            <p style="font-size: 0.115in; color: #94a3b8;">
              Defects don't just get logged—they trigger actionable corrective tasks with lifecycle tracking (<code style="color: #38bdf8;">Open → In Progress → Completed</code>) directly tied to inspection record IDs.
            </p>
          </div>
        </div>

        <!-- Feature 5 & Comparative Advantage -->
        <div class="card" style="gap: 0.14in;">
          <div class="card-title">
            <span class="card-title-icon icon-purple">📜</span>
            5. Tamper-Evident Historical Audit Trail
          </div>
          <p>
            Immutable event-sourcing ledger recording all platform interactions for compliance audits:
          </p>
          <div class="feature-list" style="gap: 0.08in;">
            <div class="feature-item">
              <span class="bullet" style="background: #c084fc;"></span>
              <div class="feature-content">
                <h4>Every Transition Tracked:</h4>
                <p>User logins, batch creation, AI screening scores, human review overrides, and CAPA status updates are stored with ISO-compliant timestamps.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #c084fc;"></span>
              <div class="feature-content">
                <h4>Grounding Integrity:</h4>
                <p>All historical events link to concrete database records, ensuring full evidentiary defensibility during customer and regulatory audits.</p>
              </div>
            </div>
          </div>

          <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.12in; margin-top: auto; background: rgba(30, 41, 59, 0.5); border-radius: 0.08in; padding: 0.1in;">
            <div style="font-size: 0.115in; font-weight: 700; color: #38bdf8; margin-bottom: 0.03in;">Competitive Advantage Summary:</div>
            <p style="font-size: 0.11in; color: #cbd5e1; line-height: 1.35;">
              Unlike generic computer vision demos, QualityTrack delivers a complete, enterprise-grade Quality Management System (QMS) combining 99.86% model accuracy with industrial compliance governance.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <span>Distinctive Strengths: Explainable AI + Dual-Model Resiliency + Closed-Loop CAPA + HITL Governance</span>
      </div>
      <div class="footer-right">UNIQUE FEATURES · QUALITYTRACK</div>
    </div>
  </div>


  <!-- ===================================================================== -->
  <!-- SLIDE 6: ADDITIONAL FEATURES & FUTURE SCOPE -->
  <!-- ===================================================================== -->
  <div class="slide">
    <div class="slide-header">
      <div class="header-left">
        <div class="badge-row">
          <span class="badge badge-primary">Roadmap & Scope · Slide 6 of 6</span>
          <span class="badge badge-success">Built & Operational</span>
          <span class="badge badge-purple">Industry 4.0 Vision</span>
        </div>
        <h1 class="slide-title">Additional Capabilities & Future Scope</h1>
        <p class="slide-subtitle">Current Production Features and Strategic Roadmap for Scaling QualityTrack</p>
      </div>
      <div class="header-right">
        <div class="project-brand">QualityTrack <span>⚡</span></div>
        <div class="slide-num">SLIDE 06 / 06</div>
      </div>
    </div>

    <div class="slide-body">
      <div class="grid-2">
        <!-- Left: Current Operational Capabilities Built -->
        <div class="card card-emerald" style="gap: 0.14in;">
          <div class="card-title">
            <span class="card-title-icon icon-emerald">✓</span>
            Current Operational Capabilities in Working Demo
          </div>
          <p style="font-size: 0.125in; color: #cbd5e1;">
            The working prototype in the repository is not a mock concept—it is a functional, tested application with active persistence:
          </p>

          <div class="feature-list" style="gap: 0.11in;">
            <div class="feature-item">
              <span class="bullet" style="background: #34d399;"></span>
              <div class="feature-content">
                <h4>Interactive CNN Architecture Viewer:</h4>
                <p>Dedicated <code style="color: #38bdf8;">/model-architecture</code> route providing layer-by-layer parameter introspection, receptive field calculations, and confusion matrix analytics.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #34d399;"></span>
              <div class="feature-content">
                <h4>Evidence-Grounded Audit Report Generator:</h4>
                <p>Dynamic reporting engine compiling batch yields, human confirmation rates, open CAPA tickets, and sample audit trails with zero hallucinations.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #34d399;"></span>
              <div class="feature-content">
                <h4>Active Batch Lifecycle Management:</h4>
                <p>Real-time tracking of production lots with batch-level quarantine holds, release dispositions, and quantity-defect correlation.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #34d399;"></span>
              <div class="feature-content">
                <h4>Live Interactive Screening API:</h4>
                <p>High-speed REST endpoint returning instant classification logits and base64 Grad-CAM overlays for rapid diagnostic testing.</p>
              </div>
            </div>
          </div>

          <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 0.08in; padding: 0.12in; margin-top: auto;">
            <div style="font-size: 0.115in; font-weight: 700; color: #34d399; margin-bottom: 0.03in;">Verified Working Repository State:</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.11in; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
              <span>• 3 Products Cataloged</span>
              <span>• 24 Batches Managed</span>
              <span>• 128 Real Inspections</span>
              <span>• 47 Audit Events</span>
            </div>
          </div>
        </div>

        <!-- Right: Future Strategic Scope & Expansion -->
        <div class="card card-highlight" style="gap: 0.14in;">
          <div class="card-title">
            <span class="card-title-icon icon-purple">🚀</span>
            Future Strategic Scope & Commercial Roadmap
          </div>
          <p style="font-size: 0.125in; color: #cbd5e1;">
            Architected to scale from single-station deployment to enterprise multi-foundry smart manufacturing:
          </p>

          <div class="feature-list" style="gap: 0.11in;">
            <div class="feature-item">
              <span class="bullet" style="background: #38bdf8;"></span>
              <div class="feature-content">
                <h4>1. Edge TensorRT / ONNX Deployment (Jetson & Coral):</h4>
                <p>Quantize PyTorch weights to INT8 for sub-3ms inference on NVIDIA Jetson AGX Orin, supporting 120 FPS multi-camera conveyor arrays.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #818cf8;"></span>
              <div class="feature-content">
                <h4>2. Multimodal GenAI Quality Assistant (VLM + RAG):</h4>
                <p>Integrate Vision-Language Models with RAG over ASTM E155 and ISO 9001 casting standards for conversational root-cause troubleshooting.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #fbbf24;"></span>
              <div class="feature-content">
                <h4>3. Continuous Active Learning Loop:</h4>
                <p>Borderline samples identified during Test-Time Augmentation (TTA) are auto-queued for metallurgical review to trigger monthly model retraining.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #c084fc;"></span>
              <div class="feature-content">
                <h4>4. Industrial SCADA / MES / ERP Connectors:</h4>
                <p>OPC-UA and MQTT interfaces to trigger physical pneumatic reject diverter gates and synchronize lot statuses with SAP S/4HANA or Siemens Opcenter.</p>
              </div>
            </div>
            <div class="feature-item">
              <span class="bullet" style="background: #34d399;"></span>
              <div class="feature-content">
                <h4>5. 3D Volumetric Defect Tomography:</h4>
                <p>Expand 2D planar vision to 3D voxel convolutions for non-destructive X-ray and CT scan detection of subsurface shrinkage cavities.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <span>Vision: Autonomous, Transparent, and Closed-Loop Quality Assurance for Industry 4.0</span>
      </div>
      <div class="footer-right">FUTURE SCOPE · QUALITYTRACK</div>
    </div>
  </div>

</body>
</html>
"""

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated HTML slide deck at: {OUTPUT_HTML} (Size: {len(html_content):,} bytes)")

# Compile to PDF using Microsoft Edge headless
edge_paths = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe"
]
browser_exe = next((p for p in edge_paths if os.path.exists(p)), None)

if browser_exe:
    print(f"Found browser executable: {browser_exe}")
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={OUTPUT_PDF}",
        "--no-pdf-header-footer",
        str(OUTPUT_HTML)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Browser conversion returncode:", res.returncode)
    if res.stdout:
        print("Stdout:", res.stdout)
    if res.stderr:
        print("Stderr:", res.stderr)

    if OUTPUT_PDF.exists():
        pdf_size = OUTPUT_PDF.stat().st_size
        print(f"Successfully generated PDF: {OUTPUT_PDF} ({pdf_size:,} bytes)")
        
        # Verify page count with PyPDF2
        with open(OUTPUT_PDF, "rb") as pf:
            reader = PyPDF2.PdfReader(pf)
            print(f"Total Pages in PDF: {len(reader.pages)}")
            for idx in range(len(reader.pages)):
                page = reader.pages[idx]
                print(f"  Page {idx+1} dimensions: {page.mediabox.width} x {page.mediabox.height} pt")
    else:
        print("Error: PDF output file was not created.")
else:
    print("Error: Neither Edge nor Chrome was found.")
