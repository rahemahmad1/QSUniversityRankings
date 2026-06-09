# QS Rankings Dashboard - run with: python qsranking_dashboard.py
# (Do not run the .ipynb file directly with Python; use this script or Jupyter.)

"""
QS Rankings Dashboard — Employability & Predictive Analytics Focus
Predicts graduate employability (QS Employment Outcomes / EO_Score) via Random Forest.
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─── DATA LOADING ────────────────────────────────────────────────────────────

try:
    _DATA_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _DATA_DIR = os.getcwd()
DATA_PATH = os.path.join(_DATA_DIR, "CLEANED QS RANKINGS.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

# ── FIX: read CSV without skipping any rows; strip BOM + whitespace ──────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# ── FIX: force-cast Year to numeric so 2024 rows are not dropped ─────────────
numeric_cols = [c for c in df.columns if
                c.endswith("_Score") or c.endswith("_Rank") or
                c in ["Year", "Rank_Current", "Rank_Previous", "Overall_Score"]]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows where Year is still NaN (truly missing), but keep 2024
df = df[df["Year"].notna()]
df["Year"] = df["Year"].astype(int)

# ── Normalise text fields ─────────────────────────────────────────────────────
if "Status" in df.columns:
    df["Status"] = df["Status"].replace({
        "Private not for Profit": "Private not-for-profit",
        "Private for Profit": "Private for-profit"
    })

if "Country" in df.columns:
    df["Country"] = df["Country"].replace({"United States of America": "United States"})

# 2024 rows have blank Region in the CSV; backfill from 2025/2026 Country→Region map
if "Region" in df.columns and "Country" in df.columns and "Year" in df.columns:
    _region_missing = df["Region"].isna() | (df["Region"].astype(str).str.strip() == "")
    if _region_missing.any():
        _country_region = (
            df.loc[~_region_missing, ["Country", "Region"]]
            .dropna(subset=["Country", "Region"])
            .drop_duplicates("Country")
            .set_index("Country")["Region"]
        )
        df.loc[_region_missing, "Region"] = df.loc[_region_missing, "Country"].map(_country_region)
        _still_missing = df["Region"].isna().sum()
        if _still_missing:
            print(f"Warning: {_still_missing} rows still missing Region after backfill")

if "Size" in df.columns:
    size_order = {"S": "Small", "M": "Medium", "L": "Large", "XL": "Extra Large"}
    df["Size_Label"] = df["Size"].map(size_order)

if "Research" in df.columns:
    research_order = {"VH": "Very High", "HI": "High", "MD": "Medium", "LO": "Low"}
    df["Research_Label"] = df["Research"].map(research_order)

if "Rank_Current" in df.columns and "Rank_Previous" in df.columns:
    df["Rank_Change"] = df["Rank_Previous"] - df["Rank_Current"]
else:
    df["Rank_Change"] = np.nan

# ── Employability features (QS Employment Outcomes pillar) ───────────────────
EMPLOYABILITY_COL = "EO_Score"
if EMPLOYABILITY_COL in df.columns:
    df["Employability_Score"] = df[EMPLOYABILITY_COL]
    df["Employability_Rate_Pct"] = df["Employability_Score"]
    df["Employability_Tier"] = pd.cut(
        df["Employability_Score"],
        bins=[-0.1, 25, 50, 75, 100.1],
        labels=["Low (0–25)", "Moderate (26–50)", "Strong (51–75)", "Elite (76–100)"],
    )

score_cols = [c for c in [
    "AR_Score", "ER_Score", "FS_Score", "CPF_Score", "IFR_Score",
    "ISR_Score", "IRN_Score", "EO_Score", "SUS_Score"
] if c in df.columns]

# ── DIAGNOSTIC: print year distribution so you can verify 2024 is present ────
print("\n===== QS DASHBOARD — DATA LOADED =====")
print(f"Total rows : {len(df):,}")
if "Year" in df.columns:
    year_counts = df["Year"].value_counts().sort_index()
    print("Rows per year:")
    for yr, cnt in year_counts.items():
        print(f"  {yr}: {cnt:,} rows")
if "Region" in df.columns and (df["Year"] == 2024).any():
    n24 = (df["Year"] == 2024).sum()
    r24 = df.loc[df["Year"] == 2024, "Region"].notna().sum()
    print(f"2024 Region filled: {r24:,} / {n24:,} rows")
print("=======================================\n")

# ─── COLOUR PALETTE & THEME ──────────────────────────────────────────────────

NAVY      = "#0B1437"
DARK_CARD = "#0F1B4C"
MID_CARD  = "#132257"
ACCENT    = "#00C6FF"
ACCENT2   = "#7B61FF"
GOLD      = "#F5A623"
GREEN     = "#2ECC71"
RED_SOFT  = "#E74C3C"
WHITE     = "#FFFFFF"
LIGHT_TXT = "#CBD5E1"
MUTED_TXT = "#64748B"
BORDER    = "rgba(0,198,255,0.18)"

CHART_TEMPLATE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,20,55,0.6)",
    font=dict(color=LIGHT_TXT, family="'DM Sans', sans-serif"),
    margin=dict(l=30, r=20, t=50, b=30),
    hovermode="x unified",
    colorway=[ACCENT, ACCENT2, GOLD, GREEN, RED_SOFT, "#F39C12", "#9B59B6", "#1ABC9C"],
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=LIGHT_TXT)),
    title_font=dict(color=WHITE, size=15),
    xaxis=dict(gridcolor="rgba(100,116,139,0.2)", zeroline=False, tickfont=dict(color=LIGHT_TXT)),
    yaxis=dict(gridcolor="rgba(100,116,139,0.2)", zeroline=False, tickfont=dict(color=LIGHT_TXT)),
    hoverlabel=dict(
        bgcolor="#050e2a",
        bordercolor=ACCENT,
        font=dict(color=WHITE, size=13, family="'DM Sans', sans-serif"),
    ),
)

def styled_fig(fig, height=380):
    fig.update_layout(**CHART_TEMPLATE, height=height)
    fig.update_xaxes(CHART_TEMPLATE["xaxis"])
    fig.update_yaxes(CHART_TEMPLATE["yaxis"])
    return fig

# ─── FILTER HELPER ───────────────────────────────────────────────────────────

def df_from_store(data):
    """Rebuild dataframe from dcc.Store; keep Year as plain int (2024, 2025, 2026)."""
    d = pd.DataFrame(data)
    if "Year" in d.columns:
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
        d = d[d["Year"].notna()].copy()
        d["Year"] = d["Year"].astype(int)
    return d


def apply_filters(d, years, regions, countries, statuses, sizes, research, rank_range):
    f = d.copy()
    if "Year" in f.columns:
        f["Year"] = pd.to_numeric(f["Year"], errors="coerce")
        f = f[f["Year"].notna()].copy()
        f["Year"] = f["Year"].astype(int)
    if years:
        years_int = [int(y) for y in years]
        f = f[f["Year"].isin(years_int)]
    if regions and "Region" in f.columns:
        f = f[f["Region"].isin(regions)]
    if countries and "Country" in f.columns:
        f = f[f["Country"].isin(countries)]
    if statuses and "Status" in f.columns:
        f = f[f["Status"].isin(statuses)]
    if sizes and "Size_Label" in f.columns:
        f = f[f["Size_Label"].isin(sizes)]
    if research and "Research_Label" in f.columns:
        f = f[f["Research_Label"].isin(research)]
    if rank_range and "Rank_Current"  in f.columns:
        f = f[f["Rank_Current"].between(rank_range[0], rank_range[1])]
    return f

# ─── REUSABLE COMPONENTS ─────────────────────────────────────────────────────

CARD_STYLE = {
    "background": f"linear-gradient(135deg, {DARK_CARD}, {MID_CARD})",
    "border": f"1px solid {BORDER}",
    "borderRadius": "14px",
    "boxShadow": "0 4px 24px rgba(0,0,0,0.35)",
    "padding": "18px 22px",
}

def kpi_card(icon, title, value_id, accent_color=ACCENT):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "22px", "marginRight": "8px"}),
            html.Span(title, style={"color": LIGHT_TXT, "fontSize": "12px",
                                    "fontWeight": "600", "letterSpacing": "0.07em",
                                    "textTransform": "uppercase"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div(id=value_id, style={
            "color": accent_color, "fontSize": "26px",
            "fontWeight": "800", "letterSpacing": "-0.5px"
        })
    ], style=CARD_STYLE)


def section_header(text):
    return html.Div([
        html.Div(style={
            "width": "4px", "height": "22px", "background": ACCENT,
            "borderRadius": "2px", "marginRight": "10px"
        }),
        html.H5(text, style={"color": WHITE, "margin": 0, "fontWeight": "700"})
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "18px"})


def hypothesis_panel(title, bullets):
    """Plain-English insight box shown above each tab's charts."""
    bullet_els = []
    for icon, text in bullets:
        bullet_els.append(
            html.Div([
                html.Span(icon, className="icon"),
                html.Span(text, className="text"),
            ], className="hyp-bullet")
        )
    return html.Div([
        html.H6(f"💡  {title}"),
        html.Div(bullet_els)
    ], className="hypothesis-panel")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

def dropdown_group(label, dropdown_id, placeholder):
    return html.Div([
        html.Label(label, style={
            "color": LIGHT_TXT, "fontSize": "11px", "fontWeight": "700",
            "letterSpacing": "0.08em", "textTransform": "uppercase", "marginBottom": "4px",
            "display": "block"
        }),
        dcc.Dropdown(
            id=dropdown_id, multi=True, placeholder=placeholder,
            style={"background": DARK_CARD},
            className="custom-dropdown"
        ),
    ], style={"marginBottom": "14px"})


sidebar = html.Div([
    html.Div([
        html.Div("⚙", style={"fontSize": "20px", "marginRight": "8px"}),
        html.Span("FILTERS", style={
            "color": ACCENT, "fontWeight": "900", "fontSize": "13px",
            "letterSpacing": "0.15em"
        })
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "20px",
              "paddingBottom": "12px", "borderBottom": f"1px solid {BORDER}"}),

    dropdown_group("Year", "year_filter", "All years"),
    dropdown_group("Region", "region_filter", "All regions"),
    dropdown_group("Country", "country_filter", "All countries"),
    dropdown_group("Institution Type", "status_filter", "All types"),
    dropdown_group("Size", "size_filter", "All sizes"),
    dropdown_group("Research Intensity", "research_filter", "All levels"),

    html.Div([
        html.Label("Rank Range", style={
            "color": LIGHT_TXT, "fontSize": "11px", "fontWeight": "700",
            "letterSpacing": "0.08em", "textTransform": "uppercase",
            "marginBottom": "4px", "display": "block"
        }),
        dcc.RangeSlider(
            id="rank_filter", min=1,
            max=int(df["Rank_Current"].max()) if "Rank_Current" in df.columns else 1500,
            value=[1, min(500, int(df["Rank_Current"].max()) if "Rank_Current" in df.columns else 500)],
            step=1,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ], style={"marginBottom": "18px"}),

    dbc.Button([
        html.Span("↺", style={"marginRight": "6px"}), "Reset Filters"
    ], id="reset_btn", color="light", outline=True,
        className="w-100",
        style={
            "borderColor": ACCENT, "color": ACCENT, "fontWeight": "700",
            "borderRadius": "8px", "fontSize": "12px", "letterSpacing": "0.06em"
        }
    )
], style={
    "padding": "20px 16px",
    "background": f"linear-gradient(180deg, {NAVY} 0%, #091030 100%)",
    "minHeight": "100vh",
    "borderRight": f"1px solid {BORDER}",
    "position": "sticky", "top": 0,
})

# ─── APP LAYOUT ──────────────────────────────────────────────────────────────

TAB_STYLE = {
    "background": "transparent",
    "color": MUTED_TXT,
    "border": "none",
    "borderBottom": "2px solid transparent",
    "padding": "10px 20px",
    "fontWeight": "600",
    "fontSize": "13px",
    "letterSpacing": "0.05em",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": ACCENT,
    "borderBottom": f"2px solid {ACCENT}",
    "background": "rgba(0,198,255,0.06)",
    "borderRadius": "4px 4px 0 0",
}
TABS_CONTAINER = {
    "background": "transparent",
    "borderBottom": f"1px solid {BORDER}",
    "marginBottom": "20px",
}

GLOBAL_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

body {{
    background: {NAVY} !important;
    font-family: 'DM Sans', sans-serif !important;
    color: {LIGHT_TXT} !important;
}}

/* ══════════════════════════════════════════════════
   DROPDOWN — covers BOTH Dash's old React-Select v1
   (.Select-*) AND new React-Select v5 (.react-select__*)
   class names, with and without .custom-dropdown prefix
   ══════════════════════════════════════════════════ */

/* ── Control box (the visible closed state) ── */
.custom-dropdown .Select-control,
.Select-control,
[class*="react-select__control"],
.custom-dropdown [class*="react-select__control"] {{
    background-color: #1a2a6c !important;
    border: 1px solid {ACCENT} !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    min-height: 34px !important;
}}

/* ── Input text inside the box ── */
.custom-dropdown .Select-input > input,
.Select-input > input,
[class*="react-select__input"] input,
[class*="react-select__single-value"],
[class*="react-select__placeholder"] {{
    color: {WHITE} !important;
    font-size: 13px !important;
}}

/* ── Placeholder text ── */
.custom-dropdown .Select-placeholder,
.Select-placeholder,
[class*="react-select__placeholder"] {{
    color: #8fadc8 !important;
    font-size: 13px !important;
}}

/* ── Selected value label (single select) ── */
.custom-dropdown .Select-value-label,
.Select-value-label,
[class*="react-select__single-value"] {{
    color: {WHITE} !important;
    font-size: 13px !important;
}}

/* ── Multi-select chip tags ── */
.custom-dropdown .Select-value,
.Select-value,
[class*="react-select__multi-value"] {{
    background-color: rgba(0,198,255,0.22) !important;
    border: 1px solid rgba(0,198,255,0.4) !important;
    border-radius: 4px !important;
}}
.custom-dropdown .Select-value-label,
[class*="react-select__multi-value__label"] {{
    color: {WHITE} !important;
    font-weight: 600 !important;
    font-size: 12px !important;
}}
.custom-dropdown .Select-value-icon,
[class*="react-select__multi-value__remove"] {{
    color: {ACCENT} !important;
    border-right: 1px solid rgba(0,198,255,0.3) !important;
    cursor: pointer !important;
}}
.custom-dropdown .Select-value-icon:hover,
[class*="react-select__multi-value__remove"]:hover {{
    background-color: rgba(231,76,60,0.35) !important;
    color: #ff6b6b !important;
}}

/* ── Dropdown arrow ── */
.custom-dropdown .Select-arrow,
.Select-arrow,
[class*="react-select__indicator"] svg,
[class*="react-select__dropdown-indicator"] svg {{
    color: {ACCENT} !important;
    fill: {ACCENT} !important;
    border-top-color: {ACCENT} !important;
}}
[class*="react-select__indicator-separator"] {{
    background-color: rgba(0,198,255,0.2) !important;
}}

/* ── Open menu outer container ── */
.custom-dropdown .Select-menu-outer,
.Select-menu-outer,
[class*="react-select__menu"] {{
    background-color: #0d1a45 !important;
    border: 1px solid {ACCENT} !important;
    border-radius: 8px !important;
    box-shadow: 0 12px 40px rgba(0,0,0,0.75) !important;
    z-index: 99999 !important;
    overflow: hidden !important;
}}

/* ── Menu list scroll container ── */
.custom-dropdown .Select-menu,
.Select-menu,
[class*="react-select__menu-list"] {{
    background-color: #0d1a45 !important;
    padding: 4px 0 !important;
    max-height: 220px !important;
}}

/* ── EVERY individual option row — THE KEY FIX ── */
.custom-dropdown .Select-option,
.Select-option,
[class*="react-select__option"],
div[class*="react-select__option"] {{
    background-color: #0d1a45 !important;
    color: #ffffff !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 9px 14px !important;
    cursor: pointer !important;
    line-height: 1.4 !important;
}}

/* ── Hovered option ── */
.custom-dropdown .Select-option:hover,
.Select-option.is-focused,
[class*="react-select__option"]:hover,
[class*="react-select__option--is-focused"] {{
    background-color: rgba(0,198,255,0.2) !important;
    color: #ffffff !important;
}}

/* ── Selected (checked) option ── */
.custom-dropdown .Select-option.is-selected,
[class*="react-select__option--is-selected"] {{
    background-color: rgba(0,198,255,0.35) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}}

/* ── VirtualizedSelect (used in older Dash versions for large lists) ── */
.VirtualizedSelectOption {{
    background-color: #0d1a45 !important;
    color: #ffffff !important;
    font-size: 13px !important;
    padding: 9px 14px !important;
}}
.VirtualizedSelectFocusedOption {{
    background-color: rgba(0,198,255,0.2) !important;
    color: #ffffff !important;
}}
.VirtualizedSelectSelectedOption {{
    background-color: rgba(0,198,255,0.35) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}}

/* ── DASH TABLE TOOLTIPS — dark bg, pure white text ── */
.dash-tooltip {{
    background-color: #050e2a !important;
    border: 1px solid {ACCENT} !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.7) !important;
    padding: 8px 14px !important;
    max-width: 280px !important;
}}
.dash-tooltip p,
.dash-tooltip span,
.dash-tooltip div,
.dash-tooltip * {{
    color: {WHITE} !important;
    font-size: 13px !important;
    font-family: 'DM Sans', sans-serif !important;
    line-height: 1.6 !important;
}}
/* Plotly hover tooltip override */
.hoverlayer .hovertext rect {{
    fill: #050e2a !important;
    stroke: {ACCENT} !important;
}}
.hoverlayer .hovertext text,
.hoverlayer .hovertext tspan {{
    fill: {WHITE} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {NAVY}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}

/* Range slider */
.rc-slider-rail {{ background: rgba(100,116,139,0.3) !important; }}
.rc-slider-track {{ background: {ACCENT} !important; }}
.rc-slider-handle {{ border-color: {ACCENT} !important; background: {ACCENT} !important; }}

/* Table */
.dash-spreadsheet-container .dash-spreadsheet-inner th {{
    background-color: #0B1437 !important;
    color: {ACCENT} !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid {BORDER} !important;
}}
.dash-spreadsheet-container .dash-spreadsheet-inner td {{
    background-color: {DARK_CARD} !important;
    color: {WHITE} !important;
    border-bottom: 1px solid rgba(100,116,139,0.15) !important;
}}
.dash-spreadsheet-container .dash-spreadsheet-inner td:hover {{
    background-color: rgba(0,198,255,0.08) !important;
}}

/* ── HYPOTHESIS PANEL ── */
.hypothesis-panel {{
    background: linear-gradient(135deg, #0a1640, #0f1d55) !important;
    border: 1px solid rgba(0,198,255,0.25) !important;
    border-left: 4px solid {GOLD} !important;
    border-radius: 12px !important;
    padding: 18px 22px !important;
    margin-bottom: 22px !important;
}}
.hypothesis-panel h6 {{
    color: {GOLD} !important;
    font-weight: 800 !important;
    font-size: 12px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-bottom: 10px !important;
}}
.hypothesis-panel p {{
    color: #dde6f5 !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    margin-bottom: 6px !important;
}}
.hypothesis-panel .hyp-bullet {{
    display: flex;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 6px;
}}
.hypothesis-panel .hyp-bullet span.icon {{
    color: {ACCENT};
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}}
.hypothesis-panel .hyp-bullet span.text {{
    color: #dde6f5;
    font-size: 13px;
    line-height: 1.65;
}}
"""

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
)
server = app.server

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>QS Rankings Dashboard</title>
        {{%favicon%}}
        {{%css%}}
        <style>{GLOBAL_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

app.layout = dbc.Container([
    dcc.Store(id="store_data", data=df.to_dict("records")),

    # ── HEADER ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("🎓", style={"fontSize": "32px", "marginRight": "14px"}),
            html.Div([
                html.H1("QS WORLD UNIVERSITY RANKINGS",
                        style={"color": WHITE, "fontWeight": "900", "fontSize": "22px",
                               "margin": 0, "letterSpacing": "0.08em"}),
                html.Div("Employer Reputation Intelligence Dashboard",
                         style={"color": ACCENT, "fontSize": "12px", "fontWeight": "600",
                                "letterSpacing": "0.12em", "textTransform": "uppercase"})
            ])
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("LIVE DATA", style={
                "background": f"rgba(46,204,113,0.15)", "color": GREEN,
                "border": f"1px solid {GREEN}", "borderRadius": "20px",
                "padding": "4px 14px", "fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "0.1em"
            }),
        ])
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "20px 28px",
        "background": f"linear-gradient(90deg, {DARK_CARD} 0%, {NAVY} 60%, rgba(0,198,255,0.05) 100%)",
        "borderBottom": f"1px solid {BORDER}",
        "marginBottom": "0",
    }),

    dbc.Row([
        # ── SIDEBAR ─────────────────────────────────────────────────────────
        dbc.Col(sidebar, width=2, style={"padding": 0}),

        # ── MAIN CONTENT ────────────────────────────────────────────────────
        dbc.Col([
            html.Div(style={"padding": "20px 24px"}, children=[

                # KPI ROW
                dbc.Row([
                    dbc.Col(kpi_card("🏛️", "Universities",   "kpi_unis",     ACCENT),  md=3),
                    dbc.Col(kpi_card("🌍", "Countries",      "kpi_countries", GOLD),    md=3),
                    dbc.Col(kpi_card("📊", "Avg Overall",    "kpi_overall",  ACCENT2), md=3),
                    dbc.Col(kpi_card("💼", "Avg ER Score",   "kpi_er",       GREEN),   md=3),
                ], className="g-3 mb-4"),

                # TABS
                dcc.Tabs(id="tabs", value="overview",
                         style=TABS_CONTAINER,
                         children=[
                    dcc.Tab(label="📊  Overview",            value="overview",
                            style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="💼  Employer Reputation",  value="er",
                            style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="📈  Trends & Time",        value="trends",
                            style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="🔬  Score Deep Dive",      value="deepdive",
                            style=TAB_STYLE, selected_style=TAB_SELECTED),
                    dcc.Tab(label="🗂️  Data Explorer",        value="explorer",
                            style=TAB_STYLE, selected_style=TAB_SELECTED),
                ]),

                html.Div(id="tab_content"),

                # EXECUTIVE SUMMARY
                html.Div([
                    html.Div(style={
                        "height": "1px", "background":
                        f"linear-gradient(90deg, {ACCENT}, transparent)",
                        "marginBottom": "18px", "marginTop": "28px"
                    }),
                    section_header("Executive Summary"),
                    html.Div(id="exec_summary")
                ])
            ])
        ], width=10, style={"padding": 0,
                            "background": f"linear-gradient(180deg, {NAVY} 0%, #091030 100%)"})
    ], className="g-0", style={"minHeight": "100vh"})
], fluid=True, style={"padding": 0, "background": NAVY})


# ─── CALLBACKS ───────────────────────────────────────────────────────────────

@app.callback(
    Output("year_filter",     "options"), Output("year_filter",     "value"),
    Output("region_filter",   "options"), Output("region_filter",   "value"),
    Output("country_filter",  "options"), Output("country_filter",  "value"),
    Output("status_filter",   "options"), Output("status_filter",   "value"),
    Output("size_filter",     "options"), Output("size_filter",     "value"),
    Output("research_filter", "options"), Output("research_filter", "value"),
    Output("rank_filter", "min"), Output("rank_filter", "max"), Output("rank_filter", "value"),
    Input("store_data", "data"),
    Input("reset_btn", "n_clicks"),
)
def init_filters(data, _reset_clicks):
    d = df_from_store(data)

    def opts(col):
        if col in d.columns:
            raw = d[col].dropna().unique().tolist()
            if col == "Year":
                vals = sorted([int(v) for v in raw])
            else:
                vals = sorted([str(v) for v in raw])
            return [{"label": str(x), "value": x} for x in vals], vals
        return [], []

    y_opt, y_val   = opts("Year")
    r_opt, r_val   = opts("Region")
    c_opt, c_val   = opts("Country")
    s_opt, s_val   = opts("Status")
    z_opt, z_val   = opts("Size_Label")
    re_opt, re_val = opts("Research_Label")
    mn = 1
    mx = int(d["Rank_Current"].max()) if "Rank_Current" in d.columns and d["Rank_Current"].notna().any() else 1500
    return (y_opt, y_val, r_opt, r_val, c_opt, c_val, s_opt, s_val,
            z_opt, z_val, re_opt, re_val, mn, mx, [mn, min(mx, 500)])


@app.callback(
    Output("kpi_unis",      "children"),
    Output("kpi_countries", "children"),
    Output("kpi_overall",   "children"),
    Output("kpi_er",        "children"),
    Output("tab_content",   "children"),
    Output("exec_summary",  "children"),
    Input("tabs",            "value"),
    Input("year_filter",     "value"),
    Input("region_filter",   "value"),
    Input("country_filter",  "value"),
    Input("status_filter",   "value"),
    Input("size_filter",     "value"),
    Input("research_filter", "value"),
    Input("rank_filter",     "value"),
    State("store_data",      "data")
)
def render(tab, years, regions, countries, statuses, sizes, research, rank_range, data):
    d = df_from_store(data)
    f = apply_filters(d, years, regions, countries, statuses, sizes, research, rank_range)

    uni     = f["Institution"].nunique()      if "Institution"   in f.columns else 0
    ctry    = f["Country"].nunique()           if "Country"       in f.columns else 0
    overall = f["Overall_Score"].mean()        if "Overall_Score" in f.columns and f["Overall_Score"].notna().any() else np.nan
    er      = f["ER_Score"].mean()             if "ER_Score"      in f.columns and f["ER_Score"].notna().any()      else np.nan

    # ── EXEC SUMMARY ────────────────────────────────────────────────────────
    years_present = sorted(f["Year"].dropna().unique().tolist()) if "Year" in f.columns else []
    summary = html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(f"{len(f):,}", style={"color": ACCENT, "fontSize": "20px", "fontWeight": "800"}),
                html.Div("Total Records", style={"color": MUTED_TXT, "fontSize": "11px"})
            ], style={**CARD_STYLE, "textAlign": "center"}), md=2),
            dbc.Col(html.Div([
                html.Div(", ".join(map(str, years_present)) or "—",
                         style={"color": GOLD, "fontSize": "14px", "fontWeight": "700"}),
                html.Div("Year(s) in View", style={"color": MUTED_TXT, "fontSize": "11px"})
            ], style={**CARD_STYLE, "textAlign": "center"}), md=3),
            dbc.Col(html.Div([
                html.Div(f"{overall:.2f}" if pd.notna(overall) else "—",
                         style={"color": ACCENT2, "fontSize": "20px", "fontWeight": "800"}),
                html.Div("Avg Overall Score", style={"color": MUTED_TXT, "fontSize": "11px"})
            ], style={**CARD_STYLE, "textAlign": "center"}), md=2),
            dbc.Col(html.Div([
                html.Div(f"{er:.2f}" if pd.notna(er) else "—",
                         style={"color": GREEN, "fontSize": "20px", "fontWeight": "800"}),
                html.Div("Avg ER Score", style={"color": MUTED_TXT, "fontSize": "11px"})
            ], style={**CARD_STYLE, "textAlign": "center"}), md=2),
            dbc.Col(html.Div([
                html.Div("Top ER Driver: Academic Reputation + Faculty/Student Ratio",
                         style={"color": LIGHT_TXT, "fontSize": "12px", "lineHeight": "1.6"}),
            ], style={**CARD_STYLE}), md=3),
        ], className="g-3")
    ])

    if f.empty:
        empty_msg = html.Div([
            html.Div("🔍", style={"fontSize": "48px", "marginBottom": "12px"}),
            html.Div("No data matches your current filters.",
                     style={"color": LIGHT_TXT, "fontSize": "16px"})
        ], style={"textAlign": "center", "padding": "60px"})
        kpi_dash = "—"
        return str(uni), str(ctry), kpi_dash, kpi_dash, empty_msg, summary

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    if tab == "overview":
        f_chart = f.copy()
        if "Year" in f_chart.columns:
            f_chart["Year"] = f_chart["Year"].astype(str)
        year_order = sorted(f_chart["Year"].unique()) if "Year" in f_chart.columns else None
        fig_hist = styled_fig(px.histogram(
            f_chart, x="Overall_Score", nbins=40,
            color="Year" if "Year" in f_chart.columns else None,
            title="Overall Score Distribution by Year",
            color_discrete_sequence=[ACCENT, GOLD, ACCENT2, GREEN, RED_SOFT],
            category_orders={"Year": year_order} if year_order else None,
        ))
        fig_hist.update_traces(opacity=0.8)

        fig_scatter = styled_fig(px.scatter(
            f, x="Rank_Current", y="Overall_Score",
            color="Region" if "Region" in f.columns else None,
            hover_name="Institution" if "Institution" in f.columns else None,
            size="ER_Score" if "ER_Score" in f.columns else None,
            size_max=16,
            title="Rank vs Overall Score (bubble = ER Score)",
            opacity=0.75,
            color_discrete_sequence=[ACCENT, GOLD, ACCENT2, GREEN, RED_SOFT, "#F39C12", "#9B59B6"]
        ))

        if "Country" in f.columns and "ER_Score" in f.columns:
            top_ctry = (f.groupby("Country")["ER_Score"].mean()
                        .nlargest(15).reset_index().sort_values("ER_Score"))
            fig_bar_ctry = styled_fig(px.bar(
                top_ctry, x="ER_Score", y="Country", orientation="h",
                title="Top 15 Countries — Avg Employer Reputation Score",
                color="ER_Score", color_continuous_scale=[[0, ACCENT2], [1, ACCENT]]
            ))
        else:
            fig_bar_ctry = go.Figure()
            styled_fig(fig_bar_ctry)

        if "Region" in f.columns and "Overall_Score" in f.columns:
            region_stats = f.groupby("Region")["Overall_Score"].mean().reset_index()
            fig_pie = styled_fig(px.bar(
                region_stats.sort_values("Overall_Score"), x="Overall_Score", y="Region",
                orientation="h", title="Avg Overall Score by Region",
                color="Overall_Score", color_continuous_scale=[[0, DARK_CARD], [1, ACCENT2]]
            ))
        else:
            fig_pie = go.Figure(); styled_fig(fig_pie)

        hyp_overview = hypothesis_panel("What am I looking at? — Overview", [
            ("📌", "This tab gives you a bird's-eye view of all universities in the dataset. "
                   "Each dot or bar represents one university."),
            ("📊", "The histogram on the top-left shows how scores are spread out — "
                   "a tall bar means many universities scored in that range."),
            ("🔵", "The bubble chart links rank position to overall score. "
                   "Universities closer to the top-left are ranked higher AND score higher — "
                   "the two should align, but outliers reveal interesting exceptions."),
            ("🌍", "The bottom charts show which countries and world regions consistently "
                   "produce highly regarded graduates in the eyes of employers."),
            ("💡", "Use the Year filter on the left to compare different ranking cycles, "
                   "including the latest 2024 data."),
        ])
        content = html.Div([
            hyp_overview,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_hist,    config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_scatter, config={"displayModeBar": False}), md=6),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_bar_ctry, config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_pie,      config={"displayModeBar": False}), md=6),
            ], className="g-3"),
        ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — EMPLOYER REPUTATION
    # ════════════════════════════════════════════════════════════════════════
    elif tab == "er":
        er_df = f.dropna(subset=["ER_Score"]).copy() if "ER_Score" in f.columns else pd.DataFrame()
        if er_df.empty:
            content = html.Div("No Employer Reputation data available.",
                               style={"color": LIGHT_TXT, "padding": "40px"})
        else:
            keep = [c for c in ["Institution", "Country", "Region", "ER_Score", "Rank_Current"] if c in er_df.columns]
            top    = er_df.nlargest(10, "ER_Score")[keep]
            bottom = er_df.nsmallest(10, "ER_Score")[keep]

            fig_top = styled_fig(px.bar(
                top.sort_values("ER_Score"), x="ER_Score", y="Institution",
                orientation="h", title="Top 10 — Highest Employer Reputation",
                color="ER_Score", color_continuous_scale=[[0, "#0052CC"], [0.5, ACCENT], [1, WHITE]],
                text="ER_Score"
            ))
            fig_top.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                  textfont_color=WHITE)

            fig_bot = styled_fig(px.bar(
                bottom.sort_values("ER_Score", ascending=False), x="ER_Score", y="Institution",
                orientation="h", title="Bottom 10 — Lowest Employer Reputation",
                color="ER_Score", color_continuous_scale=[[0, RED_SOFT], [1, GOLD]],
                text="ER_Score"
            ))
            fig_bot.update_traces(texttemplate="%{text:.1f}", textposition="outside",
                                  textfont_color=WHITE)

            # ML MODEL
            features = [c for c in score_cols if c != "ER_Score" and c in f.columns]
            ml_block = html.Div()
            if len(features) >= 2:
                model_df = f[["ER_Score"] + features].dropna(subset=["ER_Score"]).copy()
                imp = SimpleImputer(strategy="median")
                model_df[features] = imp.fit_transform(model_df[features])
                X = model_df[features]; y_ml = model_df["ER_Score"]
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_ml, test_size=0.2, random_state=42)
                mdl = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
                mdl.fit(X_train, y_train)
                preds = mdl.predict(X_test)

                mae  = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2   = r2_score(y_test, preds)
                imp_df = pd.DataFrame({
                    "Feature": [c.replace("_Score", "").replace("_", " ") for c in features],
                    "Importance": mdl.feature_importances_
                }).sort_values("Importance")

                residuals = y_test.values - preds
                fig_av = styled_fig(px.scatter(
                    x=y_test, y=preds,
                    title="Actual vs Predicted ER Score",
                    labels={"x": "Actual ER Score", "y": "Predicted ER Score"},
                    color_discrete_sequence=[ACCENT]
                ))
                fig_av.add_shape(type="line",
                    x0=y_test.min(), y0=y_test.min(),
                    x1=y_test.max(), y1=y_test.max(),
                    line=dict(color=GOLD, dash="dash", width=2))

                fig_imp = styled_fig(px.bar(
                    imp_df, x="Importance", y="Feature", orientation="h",
                    title="Feature Importance for ER Score Prediction",
                    color="Importance",
                    color_continuous_scale=[[0, DARK_CARD], [0.4, ACCENT2], [1, ACCENT]]
                ))
                fig_imp.update_traces(
                    text=imp_df["Importance"].apply(lambda v: f"{v:.3f}"),
                    textposition="outside", textfont_color=LIGHT_TXT
                )

                fig_res = styled_fig(px.histogram(
                    x=residuals, nbins=30, title="Residual Distribution",
                    labels={"x": "Residual"},
                    color_discrete_sequence=[ACCENT2]
                ))

                ml_block = html.Div([
                    html.Div(style={"height": "1px", "background":
                        f"linear-gradient(90deg, {ACCENT2}, transparent)",
                        "margin": "24px 0 20px"}),
                    section_header("🤖 Predictive Analytics — Random Forest Model"),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.Div(f"{r2:.4f}", style={"color": GREEN, "fontSize": "24px", "fontWeight": "800"}),
                            html.Div("R² Score", style={"color": MUTED_TXT, "fontSize": "12px"}),
                            html.Div("(1.0 = perfect)", style={"color": MUTED_TXT, "fontSize": "10px"})
                        ], style={**CARD_STYLE, "textAlign": "center"}), md=4),
                        dbc.Col(html.Div([
                            html.Div(f"{mae:.3f}", style={"color": GOLD, "fontSize": "24px", "fontWeight": "800"}),
                            html.Div("Mean Absolute Error", style={"color": MUTED_TXT, "fontSize": "12px"}),
                        ], style={**CARD_STYLE, "textAlign": "center"}), md=4),
                        dbc.Col(html.Div([
                            html.Div(f"{rmse:.3f}", style={"color": ACCENT2, "fontSize": "24px", "fontWeight": "800"}),
                            html.Div("RMSE", style={"color": MUTED_TXT, "fontSize": "12px"}),
                        ], style={**CARD_STYLE, "textAlign": "center"}), md=4),
                    ], className="g-3 mb-3"),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig_av,  config={"displayModeBar": False}), md=4),
                        dbc.Col(dcc.Graph(figure=fig_imp, config={"displayModeBar": False}), md=4),
                        dbc.Col(dcc.Graph(figure=fig_res, config={"displayModeBar": False}), md=4),
                    ], className="g-3")
                ])

            hyp_er = hypothesis_panel("What is Employer Reputation (ER Score)?", [
                ("🎓", "Employer Reputation measures how highly global employers rate a university's graduates. "
                       "It is based on a large worldwide survey of employers asking which universities produce "
                       "the most competent, innovative and effective graduates."),
                ("🏆", "A score of 100 means employers consistently choose graduates from that university above all others. "
                       "Scores above 80 are considered elite; below 40 is below average."),
                ("📉", "The bottom 10 chart shows universities with the weakest employer perception — "
                       "they may excel in research but their graduates are less recognised in the job market."),
                ("🤖", "The Predictive Analytics section below uses a machine-learning model to estimate "
                       "what drives ER Score. It finds which other scores (e.g. Academic Reputation, "
                       "Faculty-Student ratio) are the strongest predictors of employer perception."),
                ("💡", "Key finding: Academic Reputation and overall research strength tend to be "
                       "the biggest contributors to a strong Employer Reputation score."),
            ])
            content = html.Div([
                hyp_er,
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_top, config={"displayModeBar": False}), md=6),
                    dbc.Col(dcc.Graph(figure=fig_bot, config={"displayModeBar": False}), md=6),
                ], className="g-3"),
                ml_block
            ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — TRENDS
    # ════════════════════════════════════════════════════════════════════════
    elif tab == "trends":
        charts = []
        if "Year" in f.columns and "ER_Score" in f.columns:
            trend_er = f.groupby("Year", as_index=False)["ER_Score"].mean().sort_values("Year")
            trend_er["Year"] = trend_er["Year"].astype(str)
            fig_er_trend = styled_fig(px.area(
                trend_er, x="Year", y="ER_Score", markers=True,
                title="Average Employer Reputation Score — Year-on-Year",
                color_discrete_sequence=[ACCENT]
            ))
            fig_er_trend.update_traces(fill="tozeroy",
                fillcolor="rgba(0,198,255,0.12)", line_color=ACCENT)
            charts.append(dcc.Graph(figure=fig_er_trend, config={"displayModeBar": False}))

        if "Year" in f.columns and "Overall_Score" in f.columns:
            trend_ov = f.groupby("Year", as_index=False)["Overall_Score"].mean().sort_values("Year")
            trend_ov["Year"] = trend_ov["Year"].astype(str)
            fig_ov_trend = styled_fig(px.area(
                trend_ov, x="Year", y="Overall_Score", markers=True,
                title="Average Overall Score — Year-on-Year",
                color_discrete_sequence=[ACCENT2]
            ))
            fig_ov_trend.update_traces(fill="tozeroy",
                fillcolor="rgba(123,97,255,0.12)", line_color=ACCENT2)
            charts.append(dcc.Graph(figure=fig_ov_trend, config={"displayModeBar": False}))

        if "Year" in f.columns and "Region" in f.columns and "ER_Score" in f.columns:
            reg_trend = (f.groupby(["Year", "Region"], as_index=False)["ER_Score"]
                          .mean().sort_values("Year"))
            reg_trend["Year"] = reg_trend["Year"].astype(str)
            fig_reg = styled_fig(px.line(
                reg_trend, x="Year", y="ER_Score", color="Region", markers=True,
                title="Employer Reputation by Region — Yearly Trend",
                color_discrete_sequence=[ACCENT, GOLD, ACCENT2, GREEN, RED_SOFT, "#F39C12"]
            ), height=400)
            charts.append(dcc.Graph(figure=fig_reg, config={"displayModeBar": False}))

        if "Year" in f.columns and "Rank_Current" in f.columns:
            rank_cnt = f.groupby("Year")["Institution"].count().reset_index()
            rank_cnt.columns = ["Year", "Count"]
            rank_cnt["Year"] = rank_cnt["Year"].astype(str)
            fig_cnt = styled_fig(px.bar(
                rank_cnt, x="Year", y="Count",
                title="Number of Ranked Universities per Year",
                color="Count", color_continuous_scale=[[0, DARK_CARD], [1, GOLD]]
            ))
            charts.append(dcc.Graph(figure=fig_cnt, config={"displayModeBar": False}))

        rows = []
        for i in range(0, len(charts), 2):
            pair = charts[i:i+2]
            cols = [dbc.Col(c, md=6) for c in pair]
            rows.append(dbc.Row(cols, className="g-3 mb-3"))

        hyp_trends = hypothesis_panel("What do the trend charts tell me?", [
            ("📅", "These charts track how scores have changed year-by-year across the ranking cycles in your data. "
                   "An upward line means universities are improving on average; a dip signals a decline."),
            ("🌐", "The regional breakdown shows whether employer perception is rising faster in some "
                   "parts of the world (e.g. Asia vs Europe). This helps spot which regions are "
                   "becoming more competitive for graduate talent."),
            ("📈", "If both ER Score and Overall Score trend upward together, it suggests universities "
                   "are improving holistically — not just in research, but also in graduate employability."),
            ("💡", "Tip: select just one or two years in the Year filter to compare them directly."),
        ])
        content = html.Div([hyp_trends] + rows) if rows else html.Div(
            "Not enough time-series data.", style={"color": LIGHT_TXT, "padding": "40px"})

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — SCORE DEEP DIVE (NEW)
    # ════════════════════════════════════════════════════════════════════════
    elif tab == "deepdive":
        available_scores = [c for c in score_cols if c in f.columns]

        # Correlation heatmap
        if len(available_scores) >= 3:
            corr_data = f[available_scores].dropna()
            corr_matrix = corr_data.corr()
            nice_names = [c.replace("_Score", "").replace("_", " ") for c in corr_matrix.columns]
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=nice_names, y=nice_names,
                colorscale=[[0, DARK_CARD], [0.5, ACCENT2], [1, ACCENT]],
                text=corr_matrix.values.round(2),
                texttemplate="%{text}",
                textfont_color=WHITE,
                showscale=True
            ))
            styled_fig(fig_corr, height=420)
            fig_corr.update_layout(title="Score Correlation Matrix")
        else:
            fig_corr = go.Figure(); styled_fig(fig_corr)

        # Box plots per score
        if available_scores and "Region" in f.columns:
            long_df = f.melt(
                id_vars=["Region"],
                value_vars=available_scores,
                var_name="Metric", value_name="Score"
            )
            long_df["Metric"] = long_df["Metric"].str.replace("_Score", "").str.replace("_", " ")
            fig_box = styled_fig(px.box(
                long_df.dropna(subset=["Score"]),
                x="Metric", y="Score", color="Region",
                title="Score Distribution by Region",
                color_discrete_sequence=[ACCENT, GOLD, ACCENT2, GREEN, RED_SOFT, "#F39C12"]
            ), height=400)
        else:
            fig_box = go.Figure(); styled_fig(fig_box)

        # Radar: top 5 universities avg scores
        if len(available_scores) >= 3:
            top5 = (f.dropna(subset=["ER_Score"])
                     .nlargest(5, "ER_Score")[["Institution"] + available_scores])
            fig_radar = go.Figure()
            nice = [c.replace("_Score", "").replace("_", " ") for c in available_scores]
            colours_r = [ACCENT, GOLD, ACCENT2, GREEN, RED_SOFT]
            for i, (_, row) in enumerate(top5.iterrows()):
                vals = [row[c] for c in available_scores]
                vals += [vals[0]]  # close loop
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals, theta=nice + [nice[0]],
                    fill="toself",
                    fillcolor=f"rgba({int(colours_r[i%5][1:3],16)},"
                              f"{int(colours_r[i%5][3:5],16)},"
                              f"{int(colours_r[i%5][5:7],16)},0.15)",
                    line_color=colours_r[i % 5],
                    name=str(row["Institution"])[:30]
                ))
            styled_fig(fig_radar, height=420)
            fig_radar.update_layout(
                title="Score Profile — Top 5 Universities by ER Score",
                polar=dict(
                    bgcolor="rgba(11,20,55,0.8)",
                    radialaxis=dict(color=LIGHT_TXT, gridcolor=BORDER),
                    angularaxis=dict(color=LIGHT_TXT, gridcolor=BORDER)
                )
            )
        else:
            fig_radar = go.Figure(); styled_fig(fig_radar)

        # Score by institution type
        if available_scores and "Status" in f.columns:
            avg_status = f.groupby("Status")[available_scores].mean().reset_index()
            avg_long = avg_status.melt(
                id_vars=["Status"], value_vars=available_scores,
                var_name="Metric", value_name="Score"
            )
            avg_long["Metric"] = avg_long["Metric"].str.replace("_Score", "").str.replace("_", " ")
            fig_grouped = styled_fig(px.bar(
                avg_long, x="Metric", y="Score", color="Status", barmode="group",
                title="Avg Score by Institution Type",
                color_discrete_sequence=[ACCENT, GOLD, ACCENT2, GREEN]
            ), height=350)
        else:
            fig_grouped = go.Figure(); styled_fig(fig_grouped)

        hyp_deepdive = hypothesis_panel("Score Deep Dive — How do the metrics relate to each other?", [
            ("🔗", "The Correlation Matrix (top-left) shows how pairs of scores move together. "
                   "A value near +1.0 (bright) means two scores almost always rise or fall together. "
                   "Near 0 means they are unrelated."),
            ("🕸️", "The Radar Chart (top-right) shows the full score profile of the top 5 universities "
                   "ranked by Employer Reputation. A large, well-rounded shape means strength across all areas."),
            ("📦", "The Box Plot (bottom-left) shows the spread of scores by world region. "
                   "A tall box means universities in that region vary a lot; a short box means they are more consistent."),
            ("🏫", "The grouped bars (bottom-right) compare Public vs Private universities across every metric — "
                   "useful for understanding whether institution type influences specific scores."),
            ("💡", "No technical background needed — think of high correlation as: "
                   "'universities that do well in X almost always do well in Y too.'"),
        ])
        content = html.Div([
            hyp_deepdive,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_corr,   config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_radar,  config={"displayModeBar": False}), md=6),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_box,     config={"displayModeBar": False}), md=6),
                dbc.Col(dcc.Graph(figure=fig_grouped, config={"displayModeBar": False}), md=6),
            ], className="g-3"),
        ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — DATA EXPLORER
    # ════════════════════════════════════════════════════════════════════════
    else:
        display_cols = [c for c in [
            "Year", "Rank_Current", "Rank_Previous", "Rank_Change",
            "Institution", "Country", "Region", "Overall_Score",
            "ER_Score", "AR_Score", "FS_Score", "CPF_Score",
            "IFR_Score", "ISR_Score", "IRN_Score", "EO_Score", "SUS_Score",
            "Size_Label", "Research_Label", "Status"
        ] if c in f.columns]

        sorted_f = f.sort_values("ER_Score", ascending=False) if "ER_Score" in f.columns else f.copy()

        hyp_explorer = hypothesis_panel("How to use the Data Explorer", [
            ("🔍", "This is the raw data table — every row is one university in one ranking year. "
                   "You can scroll left and right to see all score columns."),
            ("↕️", "Click any column header to sort the table by that value — e.g. click 'ER Score' "
                   "to see the highest-rated universities at the top."),
            ("🔎", "Use the filter row (the blank row just below the headers) to search within a column. "
                   "For example, type 'United Kingdom' under Country to see only UK universities."),
            ("🔵", "Rows highlighted in blue have an ER Score above 90 — these are the most employer-recognised universities. "
                   "Rows in red have an ER Score below 30."),
            ("💡", "Hover over any cell to see the full value if it is cut off."),
        ])
        content = html.Div([
            hyp_explorer,
            html.Div([
                html.Span("📋", style={"marginRight": "6px"}),
                html.Span(f"Showing top 200 of {len(f):,} records · sorted by ER Score",
                          style={"color": MUTED_TXT, "fontSize": "12px"})
            ], style={"marginBottom": "12px"}),
            dash_table.DataTable(
                id="main_table",
                columns=[{"name": c.replace("_", " "), "id": c} for c in display_cols],
                data=sorted_f[display_cols].head(200).to_dict("records"),
                page_size=20,
                sort_action="native",
                filter_action="native",
                row_selectable="single",
                style_table={"overflowX": "auto", "borderRadius": "12px",
                             "border": f"1px solid {BORDER}"},
                style_cell={
                    "backgroundColor": DARK_CARD,
                    "color": WHITE,
                    "fontFamily": "'DM Sans', sans-serif",
                    "fontSize": "12px",
                    "padding": "10px 14px",
                    "border": f"1px solid rgba(100,116,139,0.1)",
                    "maxWidth": "180px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": NAVY,
                    "color": ACCENT,
                    "fontWeight": "700",
                    "fontSize": "11px",
                    "letterSpacing": "0.06em",
                    "textTransform": "uppercase",
                    "border": f"1px solid {BORDER}",
                },
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "rgba(15,27,76,0.5)"},
                    {"if": {"filter_query": "{ER_Score} > 90"},
                     "color": ACCENT, "fontWeight": "700"},
                    {"if": {"filter_query": "{ER_Score} < 30"},
                     "color": RED_SOFT},
                ],
                tooltip_data=[
                    {col: {"value": str(row[col]), "type": "markdown"}
                     for col in display_cols if col in row}
                    for row in sorted_f[display_cols].head(200).to_dict("records")
                ],
                tooltip_duration=None,
            ),
            html.Div("💡 High ER Score (>90) shown in blue · Low ER Score (<30) shown in red",
                     style={"color": MUTED_TXT, "fontSize": "11px", "marginTop": "10px"})
        ])

    kpi_overall_str = f"{overall:.1f}" if pd.notna(overall) else "—"
    kpi_er_str      = f"{er:.1f}"      if pd.notna(er)      else "—"
    return str(uni), str(ctry), kpi_overall_str, kpi_er_str, content, summary


# ─── RUN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)