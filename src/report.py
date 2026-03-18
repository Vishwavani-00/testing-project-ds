"""
HTML Report Generator
Produces a self-contained HTML report with:
- Executive Summary
- EDA charts + ADF stationarity results
- Model comparison table + metric charts
- Best model forecast plot
- Forecast table
"""
import base64
import io
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _img_tag(b64: str, width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:1000px;margin:12px 0;" />'


def generate_html_report(
    eda_results: dict,
    metrics_df: pd.DataFrame,
    best_model_name: str,
    forecast_df: pd.DataFrame,
    eval_figs: dict,
    forecast_fig: plt.Figure,
    output_path: str
) -> str:
    """
    Build and write HTML report. Returns the output path.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    adf = eda_results["adf_stats"]
    stats = eda_results["summary_stats"]
    stationarity = "Stationary ✓" if adf["is_stationary"] else "Non-Stationary ✗"

    # Convert figures to base64
    trend_b64 = _fig_to_b64(eda_results["trend_fig"])
    season_b64 = _fig_to_b64(eda_results["seasonality_fig"])
    decomp_b64 = _fig_to_b64(eda_results["decomposition_fig"])
    pred_b64 = _fig_to_b64(eval_figs["predictions_fig"])
    metrics_b64 = _fig_to_b64(eval_figs["metrics_fig"])
    residuals_b64 = _fig_to_b64(eval_figs["residuals_fig"])
    forecast_b64 = _fig_to_b64(forecast_fig)

    # Metrics table HTML
    metrics_html = metrics_df.to_html(index=False, border=0,
        classes="table", justify="center")

    # Forecast table HTML (first 10 rows)
    forecast_html = forecast_df.head(10).to_html(index=False, border=0, classes="table")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Time Series Forecasting Report — Retail Sales Demand</title>
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css"/>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa; color: #2c3e50; }}
    .navbar {{ background: #2c3e50; border: none; border-radius: 0; }}
    .navbar-brand, .navbar-nav > li > a {{ color: #ecf0f1 !important; }}
    .container-fluid {{ max-width: 1200px; margin: auto; padding: 30px 20px; }}
    h1 {{ font-size: 2rem; font-weight: 700; color: #2c3e50; }}
    h2 {{ font-size: 1.4rem; font-weight: 700; color: #34495e; border-left: 4px solid #4E79A7; padding-left: 10px; margin-top: 40px; }}
    h3 {{ font-size: 1.1rem; font-weight: 600; color: #4E79A7; margin-top: 20px; }}
    .card {{ background: #fff; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
    .kpi-grid {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }}
    .kpi-box {{ background: #fff; border-radius: 8px; padding: 16px 20px; min-width: 140px; flex: 1; box-shadow: 0 2px 6px rgba(0,0,0,0.08); text-align: center; }}
    .kpi-box .val {{ font-size: 1.6rem; font-weight: 700; color: #4E79A7; }}
    .kpi-box .lbl {{ font-size: 0.82rem; color: #7f8c8d; margin-top: 4px; }}
    .table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    .table th {{ background: #2c3e50; color: #fff; padding: 9px 12px; text-align: center; }}
    .table td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #ecf0f1; }}
    .table tr:nth-child(even) {{ background: #f4f6f7; }}
    .badge-best {{ background: #27ae60; color: #fff; padding: 3px 9px; border-radius: 12px; font-size: 0.78rem; }}
    .footer {{ text-align: center; padding: 20px; color: #95a5a6; font-size: 0.82rem; margin-top: 40px; }}
    img {{ border-radius: 6px; }}
    .toc a {{ color: #4E79A7; text-decoration: none; display: block; padding: 3px 0; font-size: 0.92rem; }}
    .toc a:hover {{ color: #2c3e50; }}
  </style>
</head>
<body>
<nav class="navbar navbar-default">
  <div class="container-fluid">
    <div class="navbar-header">
      <span class="navbar-brand">📊 Time Series Forecasting — Retail Sales</span>
    </div>
  </div>
</nav>

<div class="container-fluid">

  <div style="margin-bottom:10px; color:#7f8c8d; font-size:0.85rem;">Generated: {now}</div>
  <h1>Time Series Forecasting for Retail Sales Demand</h1>
  <p style="color:#7f8c8d;">A complete forecasting pipeline — EDA → Model Training → Evaluation → 30-Day Forecast</p>

  <!-- TOC -->
  <div class="card toc" style="max-width:360px;">
    <strong>Table of Contents</strong>
    <a href="#summary">1. Executive Summary</a>
    <a href="#eda">2. Exploratory Data Analysis</a>
    <a href="#models">3. Model Evaluation</a>
    <a href="#forecast">4. 30-Day Forecast</a>
  </div>

  <!-- 1. Executive Summary -->
  <h2 id="summary">1. Executive Summary</h2>
  <div class="card">
    <p>This report presents a full time series forecasting pipeline for retail sales demand. Historical sales data was cleaned, analysed for trend and seasonality, and fed into multiple forecasting models. The best-performing model — <strong>{best_model_name}</strong> — was used to generate a 30-day forward forecast.</p>
    <div class="kpi-grid">
      <div class="kpi-box"><div class="val">{stats.get('count', 'N/A'):.0f}</div><div class="lbl">Total Days</div></div>
      <div class="kpi-box"><div class="val">{stats.get('mean', 0):.1f}</div><div class="lbl">Avg Daily Sales</div></div>
      <div class="kpi-box"><div class="val">{stats.get('max', 0):.1f}</div><div class="lbl">Peak Sales</div></div>
      <div class="kpi-box"><div class="val">{stats.get('min', 0):.1f}</div><div class="lbl">Min Sales</div></div>
      <div class="kpi-box"><div class="val">{stationarity}</div><div class="lbl">ADF Stationarity</div></div>
      <div class="kpi-box"><div class="val">{best_model_name}</div><div class="lbl">Best Model</div></div>
    </div>
    <p><strong>RMSE (Best):</strong> {metrics_df[metrics_df['Model']==best_model_name]['RMSE'].values[0]:.4f} &nbsp;|&nbsp;
       <strong>MAE:</strong> {metrics_df[metrics_df['Model']==best_model_name]['MAE'].values[0]:.4f} &nbsp;|&nbsp;
       <strong>MAPE:</strong> {metrics_df[metrics_df['Model']==best_model_name]['MAPE (%)'].values[0]:.2f}%</p>
  </div>

  <!-- 2. EDA -->
  <h2 id="eda">2. Exploratory Data Analysis</h2>
  <div class="card">
    <h3>2.1 Sales Trend</h3>
    {_img_tag(trend_b64)}
    <h3>2.2 Seasonality — Day of Week &amp; Month</h3>
    {_img_tag(season_b64)}
    <h3>2.3 Seasonal Decomposition</h3>
    {_img_tag(decomp_b64)}
    <h3>2.4 Stationarity — ADF Test</h3>
    <table class="table" style="max-width:500px;">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>ADF Statistic</td><td>{adf['adf_statistic']}</td></tr>
      <tr><td>p-value</td><td>{adf['p_value']}</td></tr>
      <tr><td>Lags Used</td><td>{adf['lags_used']}</td></tr>
      <tr><td>Verdict</td><td><strong>{stationarity}</strong></td></tr>
      {"".join(f"<tr><td>Critical Value ({k})</td><td>{v}</td></tr>" for k,v in adf['critical_values'].items())}
    </table>
    <p style="color:#7f8c8d;font-size:0.85rem;">H₀: Series has a unit root (non-stationary). p &lt; 0.05 → reject H₀ → stationary.</p>
  </div>

  <!-- 3. Model Evaluation -->
  <h2 id="models">3. Model Evaluation</h2>
  <div class="card">
    <h3>3.1 Metrics Comparison</h3>
    {metrics_html}
    <br/>
    {_img_tag(metrics_b64)}
    <h3>3.2 Predictions vs Actual (Test Set)</h3>
    {_img_tag(pred_b64)}
    <h3>3.3 Residual Analysis — {best_model_name}</h3>
    {_img_tag(residuals_b64)}
  </div>

  <!-- 4. Forecast -->
  <h2 id="forecast">4. 30-Day Forecast</h2>
  <div class="card">
    <p>Forecast generated using <strong>{best_model_name}</strong> — the model with the lowest RMSE on the test set.</p>
    {_img_tag(forecast_b64)}
    <h3>Forecast Values (First 10 Days)</h3>
    {forecast_html}
    <p style="color:#7f8c8d;font-size:0.85rem;">Full forecast exported to <code>outputs/forecasts/</code> as CSV.</p>
  </div>

  <div class="footer">Time Series Forecasting Pipeline &nbsp;|&nbsp; Generated {now}</div>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] HTML report saved → {output_path}")
    return output_path
