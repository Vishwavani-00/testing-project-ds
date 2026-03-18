"""
report.py — Self-contained HTML report generator (MuSigma template style).
PRD Deliverable: HTML report
"""
import base64, io, os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _img(b64: str, width="100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};max-width:960px;margin:10px 0;" />'


def generate_html_report(eda_results, metrics_df, best_model_name, forecast_df, eval_figs, forecast_fig, output_path) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    adf = eda_results["adf_stats"]
    stats = eda_results["summary_stats"]
    stationary_label = "Stationary ✓" if adf["is_stationary"] else "Non-Stationary ✗"

    trend_b64 = _fig_to_b64(eda_results["trend_fig"])
    season_b64 = _fig_to_b64(eda_results["seasonality_fig"])
    decomp_b64 = _fig_to_b64(eda_results["decomposition_fig"])
    corr_b64 = _fig_to_b64(eda_results["correlation_fig"])
    pred_b64 = _fig_to_b64(eval_figs["predictions_fig"])
    metrics_b64 = _fig_to_b64(eval_figs["metrics_fig"])
    residuals_b64 = _fig_to_b64(eval_figs["residuals_fig"])
    forecast_b64 = _fig_to_b64(forecast_fig)

    best_row = metrics_df[metrics_df["Model"] == best_model_name].iloc[0]
    metrics_table = metrics_df.to_html(index=False, border=0, classes="tbl")
    forecast_table = forecast_df.head(10).to_html(index=False, border=0, classes="tbl")
    crit_rows = "".join(f"<tr><td>Critical Value ({k})</td><td>{v}</td></tr>" for k, v in adf["critical_values"].items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Time Series Forecasting — Retail Sales Demand</title>
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css"/>
  <style>
    body{{font-family:'Segoe UI',Arial,sans-serif;background:#f7f8fa;color:#2c3e50;}}
    .navbar{{background:#2c3e50;border:none;border-radius:0;}}
    .navbar-brand,.navbar-nav>li>a{{color:#ecf0f1!important;}}
    .wrap{{max-width:1180px;margin:auto;padding:28px 18px;}}
    h1{{font-size:1.9rem;font-weight:700;color:#2c3e50;}}
    h2{{font-size:1.35rem;font-weight:700;color:#34495e;border-left:4px solid #4E79A7;padding-left:10px;margin-top:38px;}}
    h3{{font-size:1.05rem;font-weight:600;color:#4E79A7;margin-top:18px;}}
    .card{{background:#fff;border-radius:8px;padding:20px 22px;margin-bottom:22px;box-shadow:0 2px 7px rgba(0,0,0,.07);}}
    .kpi{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:18px;}}
    .kpi-box{{background:#fff;border-radius:8px;padding:14px 18px;min-width:130px;flex:1;box-shadow:0 2px 6px rgba(0,0,0,.08);text-align:center;}}
    .kpi-box .val{{font-size:1.5rem;font-weight:700;color:#4E79A7;}}
    .kpi-box .lbl{{font-size:.8rem;color:#7f8c8d;margin-top:3px;}}
    .tbl{{width:100%;border-collapse:collapse;font-size:.88rem;}}
    .tbl th{{background:#2c3e50;color:#fff;padding:8px 11px;text-align:center;}}
    .tbl td{{padding:7px 11px;text-align:center;border-bottom:1px solid #ecf0f1;}}
    .tbl tr:nth-child(even){{background:#f4f6f7;}}
    .footer{{text-align:center;padding:18px;color:#95a5a6;font-size:.8rem;margin-top:36px;}}
    img{{border-radius:5px;}}
    .toc a{{color:#4E79A7;text-decoration:none;display:block;padding:2px 0;font-size:.9rem;}}
  </style>
</head>
<body>
<nav class="navbar navbar-default">
  <div class="container-fluid">
    <div class="navbar-header">
      <span class="navbar-brand">📊 Time Series Forecasting — Retail Sales Demand</span>
    </div>
  </div>
</nav>
<div class="wrap">
  <div style="color:#95a5a6;font-size:.82rem;margin-bottom:6px;">Generated: {now}</div>
  <h1>Time Series Forecasting for Retail Sales Demand</h1>
  <p style="color:#7f8c8d;">End-to-end forecasting pipeline: EDA → Modelling → Evaluation → 30-Day Forecast</p>

  <div class="card toc" style="max-width:340px;">
    <strong>Contents</strong>
    <a href="#summary">1. Executive Summary</a>
    <a href="#eda">2. Exploratory Data Analysis</a>
    <a href="#models">3. Model Evaluation</a>
    <a href="#forecast">4. 30-Day Forecast</a>
  </div>

  <h2 id="summary">1. Executive Summary</h2>
  <div class="card">
    <p>Historical retail sales data was cleaned, analysed for trend and seasonality, and used to train five forecasting models. The best-performing model — <strong>{best_model_name}</strong> (lowest RMSE on held-out 30-day test window) — was used to generate the final 30-day demand forecast.</p>
    <div class="kpi">
      <div class="kpi-box"><div class="val">{int(stats.get('count',0))}</div><div class="lbl">Total Days</div></div>
      <div class="kpi-box"><div class="val">{stats.get('mean',0):.1f}</div><div class="lbl">Avg Daily Sales</div></div>
      <div class="kpi-box"><div class="val">{stats.get('max',0):.1f}</div><div class="lbl">Peak Sales</div></div>
      <div class="kpi-box"><div class="val">{stats.get('min',0):.1f}</div><div class="lbl">Min Sales</div></div>
      <div class="kpi-box"><div class="val">{stationary_label}</div><div class="lbl">Stationarity</div></div>
      <div class="kpi-box"><div class="val">{best_model_name}</div><div class="lbl">Best Model</div></div>
    </div>
    <p><strong>Best Model:</strong> RMSE = <strong>{best_row['RMSE']}</strong> &nbsp;|&nbsp; MAE = <strong>{best_row['MAE']}</strong> &nbsp;|&nbsp; MAPE = <strong>{best_row['MAPE (%)']:.2f}%</strong></p>
  </div>

  <h2 id="eda">2. Exploratory Data Analysis</h2>
  <div class="card">
    <h3>2.1 Sales Trend</h3>{_img(trend_b64)}
    <h3>2.2 Seasonality — Day of Week & Month</h3>{_img(season_b64)}
    <h3>2.3 Seasonal Decomposition</h3>{_img(decomp_b64)}
    <h3>2.4 Feature Correlation with Sales</h3>{_img(corr_b64, "50%")}
    <h3>2.5 Stationarity — ADF Test</h3>
    <table class="tbl" style="max-width:480px;">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>ADF Statistic</td><td>{adf['adf_statistic']}</td></tr>
      <tr><td>p-value</td><td>{adf['p_value']}</td></tr>
      <tr><td>Lags Used</td><td>{adf['lags_used']}</td></tr>
      <tr><td>Verdict</td><td><strong>{stationary_label}</strong></td></tr>
      {crit_rows}
    </table>
    <p style="color:#7f8c8d;font-size:.82rem;margin-top:8px;">H₀: Series has a unit root (non-stationary). p &lt; 0.05 → reject H₀ → stationary.</p>
  </div>

  <h2 id="models">3. Model Evaluation</h2>
  <div class="card">
    <h3>3.1 Metrics Comparison</h3>{metrics_table}<br/>{_img(metrics_b64)}
    <h3>3.2 Predictions vs Actual (Test Set)</h3>{_img(pred_b64)}
    <h3>3.3 Residual Analysis — {best_model_name}</h3>{_img(residuals_b64)}
  </div>

  <h2 id="forecast">4. 30-Day Forecast</h2>
  <div class="card">
    <p>Forward forecast generated using <strong>{best_model_name}</strong> (best RMSE on test set). Shaded band represents ±10% uncertainty range.</p>
    {_img(forecast_b64)}
    <h3>Forecast Values (First 10 Days)</h3>{forecast_table}
    <p style="color:#7f8c8d;font-size:.82rem;">Full 30-day forecast exported to <code>outputs/forecasts/</code> as CSV.</p>
  </div>

  <div class="footer">Time Series Forecasting Pipeline &nbsp;|&nbsp; Generated {now}</div>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] HTML report → {output_path}")
    return output_path
