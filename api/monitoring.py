"""
Static HTML monitoring dashboard generator.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_dashboard(path: Path, stats: dict) -> None:
    """Write a self-refreshing HTML dashboard from a metrics stats dict."""
    error_rows = "".join(
        f"<tr><td>{e['timestamp']}</td>"
        f"<td><code>{e['endpoint']}</code></td>"
        f"<td>{e['detail']}</td></tr>"
        for e in stats.get("recent_errors", [])
    )
    errors_section = (
        '<p class="empty">No errors recorded.</p>'
        if not stats.get("recent_errors")
        else (
            "<table><thead><tr>"
            "<th>Timestamp</th><th>Endpoint</th><th>Detail</th>"
            "</tr></thead>"
            f"<tbody>{error_rows}</tbody></table>"
        )
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="refresh" content="300" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>API Monitoring Dashboard</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: system-ui, -apple-system, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      padding: 2rem 1rem;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ color: #38bdf8; font-size: 1.5rem; margin-bottom: .25rem; }}
    .subtitle {{ font-size: .8rem; color: #64748b; margin-bottom: 1.5rem; }}
    h2 {{ font-size: 1rem; color: #94a3b8; text-transform: uppercase;
          letter-spacing: .05em; margin: 1.75rem 0 .75rem; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: .75rem;
    }}
    .card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 1.1rem 1.25rem;
    }}
    .card .label {{
      font-size: .7rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .card .value {{
      font-size: 1.6rem;
      font-weight: 700;
      color: #38bdf8;
      margin-top: .2rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #1e293b;
      border-radius: 8px;
      overflow: hidden;
    }}
    th {{
      background: #334155;
      padding: .6rem 1rem;
      text-align: left;
      font-size: .75rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    td {{
      padding: .6rem 1rem;
      border-top: 1px solid #334155;
      font-size: .85rem;
    }}
    code {{ background: #0f172a; padding: .1rem .35rem; border-radius: 4px; font-size: .8rem; }}
    .empty {{ color: #64748b; font-size: .9rem; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>API Monitoring Dashboard</h1>
    <p class="subtitle">
      Last updated: {stats['timestamp']} &bull; Auto-refreshes every 5&nbsp;minutes
    </p>

    <h2>Overview</h2>
    <div class="cards">
      <div class="card">
        <div class="label">Uptime</div>
        <div class="value">{stats['uptime_human']}</div>
      </div>
      <div class="card">
        <div class="label">Total Predictions</div>
        <div class="value">{stats['total_predictions']:,}</div>
      </div>
      <div class="card">
        <div class="label">Predictions / hr</div>
        <div class="value">{stats['predictions_per_hour']:.1f}</div>
      </div>
      <div class="card">
        <div class="label">Avg Inference</div>
        <div class="value">{stats['avg_inference_time_ms']:.1f}&nbsp;ms</div>
      </div>
      <div class="card">
        <div class="label">Error Rate</div>
        <div class="value">{stats['error_rate']:.2%}</div>
      </div>
      <div class="card">
        <div class="label">Total Errors</div>
        <div class="value">{stats['total_errors']:,}</div>
      </div>
    </div>

    <h2>Recent Errors</h2>
    {errors_section}
  </div>
</body>
</html>"""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        logger.debug("Dashboard saved to %s", path)
    except Exception as exc:
        logger.error("Failed to save dashboard: %s", exc, exc_info=True)
