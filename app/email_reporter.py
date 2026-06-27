"""Email reporter for the Prompt Optimizer.

Sends a formatted comparison report to the admin after every successful
prompt optimization cycle.

Uses Python stdlib smtplib (no external deps required).
"""
from __future__ import annotations

import logging
import smtplib
import os
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger("prompt-optimizer.email")

# ---------------------------------------------------------------------------
# Configuration — loaded from environment variables
# ---------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USER = os.getenv("SMTP_USER") or "rickdev.ma100@gmail.com"
SMTP_PASS = os.getenv("SMTP_PASS") or "lszf keer clvm mnvz"       # set via K8s secret
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL") or "rickdev.ma100@gmail.com"


def _build_html_report(
    alert_name: str,
    old_prompt: str,
    new_prompt: str,
    winner: dict,
    current: dict,
    all_candidates: list[dict],
    cluster_applied: bool = False,
) -> str:
    """Build an HTML email body with the full benchmark comparison."""

    rows = ""
    for c in all_candidates:
        highlight = "background:#d4edda;" if c["name"] == winner["name"] else ""
        rows += f"""
        <tr style="{highlight}">
          <td>{c['name']}</td>
          <td>{c['score']:.3f}</td>
          <td>{c['latency_s']:.1f}s</td>
          <td>{c['metrics'].get('avg_dialogue_turns', '?')}</td>
          <td>{c['metrics'].get('a1_ratio', '?')}</td>
        </tr>"""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""
<!DOCTYPE html>
<html>
<head>
  <style>
    body {{ font-family: Arial, sans-serif; color: #333; max-width: 800px; margin: auto; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #16a085; border-bottom: 2px solid #16a085; padding-bottom: 5px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th {{ background: #2c3e50; color: white; padding: 8px; text-align: left; }}
    td {{ padding: 8px; border: 1px solid #ddd; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .prompt-box {{ background: #f4f4f4; border-left: 4px solid #16a085;
                  padding: 12px; white-space: pre-wrap; font-family: monospace;
                  font-size: 13px; margin: 10px 0; }}
    .winner {{ color: green; font-weight: bold; }}
    .badge {{ display:inline-block; padding:4px 10px; border-radius:12px;
              background:#16a085; color:white; font-size:13px; }}
  </style>
</head>
<body>
  <h1>🤖 Lang-Learn Prompt Optimization Report</h1>
  <p>Generated at: <strong>{timestamp}</strong></p>

  <h2>⚠️ Alert Trigger</h2>
  <p><span class="badge">{alert_name}</span></p>

  <h2>📊 Benchmark Comparison</h2>
  <table>
    <tr>
      <th>Candidate</th><th>Score</th><th>Latency</th>
      <th>Dialogue Turns</th><th>A1 Ratio</th>
    </tr>
    {rows}
  </table>

  <h2>🏆 Winner: <span class="winner">{winner['name']}</span></h2>
  <table>
    <tr><td><strong>Score</strong></td><td>{winner['score']:.3f}</td>
        <td><strong>Old Score</strong></td><td>{current['score']:.3f}</td></tr>
    <tr><td><strong>Latency</strong></td><td>{winner['latency_s']:.1f}s</td>
        <td><strong>Old Latency</strong></td><td>{current['latency_s']:.1f}s</td></tr>
    <tr><td><strong>Dialogue Turns</strong></td>
        <td>{winner['metrics'].get('avg_dialogue_turns', '?')}</td>
        <td><strong>A1 Ratio</strong></td>
        <td>{winner['metrics'].get('a1_ratio', '?')}</td></tr>
  </table>

  <h2>📝 Old Prompt</h2>
  <div class="prompt-box">{old_prompt}</div>

  <h2>✅ New Prompt</h2>
  <div class="prompt-box">{new_prompt}</div>

  <h2>🚀 Cluster Deployment</h2>
  {_cluster_status_html(cluster_applied)}

  <h2>📋 Recommendation</h2>
  {_recommendation_html(winner['name'] != 'current' and old_prompt != new_prompt)}


  <hr/>
  <p style="color:#999;font-size:12px;">Sent by lang-learn-prompt-optimizer service</p>
</body>
</html>
"""


def _recommendation_html(prompt_updated: bool) -> str:
    """Generate HTML snippet for the recommendation status."""
    if prompt_updated:
        return (
            '<p><strong>Prompt Updated Successfully</strong> — The new prompt has been written to '
            '<code>prompts/scenario_dialogue.txt</code> in the optimizer service. The previous prompt has been '
            'archived.</p>'
        )
    return (
        '<p><strong>No Update Performed</strong> — The current prompt is already optimal (none of the candidate '
        'variations improved the baseline by the minimum threshold of 0.02).</p>'
    )


def _cluster_status_html(cluster_applied: bool) -> str:
    """Generate HTML snippet for the cluster deployment status."""
    if cluster_applied:
        return (
            '<p style="color:green;"><strong>✅ Prompt auto-applied to inference service.</strong></p>'
            '<p>The <code>prompts</code> ConfigMap has been updated and inference pods '
            'have been restarted with the new prompt.</p>'
        )
    return (
        '<p style="color:orange;"><strong>⚠️ Auto-apply not performed.</strong></p>'
        '<p>The prompt was saved locally but was NOT applied to the cluster. '
        'This may be because the prompt did not improve, or because the Kubernetes '
        'API was unreachable. To apply manually, run:<br/>'
        '<code>kubectl edit cm prompts -n lang-learn</code></p>'
    )


def send_optimization_report(
    alert_name: str,
    old_prompt: str,
    new_prompt: str,
    winner: dict,
    current: dict,
    all_candidates: list[dict],
    cluster_applied: bool = False,
) -> bool:
    """Send the benchmark comparison + updated prompt email to the admin.

    Returns True on success, False on failure.
    """
    if not SMTP_PASS:
        logger.warning(
            "SMTP_PASS not set — skipping email. "
            "Set the SMTP_PASS env var (Gmail App Password) to enable email."
        )
        # Log the report to stdout so it is still visible in pod logs
        logger.info("=== PROMPT OPTIMIZATION REPORT ===")
        logger.info(f"Alert: {alert_name}")
        logger.info(f"Winner: {winner['name']} (score={winner['score']:.3f})")
        logger.info(f"Old score: {current['score']:.3f}")
        logger.info(f"New prompt:\n{new_prompt}")
        return False

    subject = f"[PROMPT OPTIMIZATION] New Prompt Selected — {alert_name}"
    html_body = _build_html_report(
        alert_name, old_prompt, new_prompt, winner, current,
        all_candidates, cluster_applied,
    )
    text_body = (
        f"Alert: {alert_name}\n"
        f"Winner: {winner['name']} (score={winner['score']:.3f})\n"
        f"Old score: {current['score']:.3f}\n\n"
        f"New prompt:\n{new_prompt}\n\n"
        f"Old prompt:\n{old_prompt}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ADMIN_EMAIL
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        logger.info(f"Sending report email to {ADMIN_EMAIL} via {SMTP_HOST}:{SMTP_PORT}")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [ADMIN_EMAIL], msg.as_string())
        logger.info("Email sent successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
