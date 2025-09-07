import pandas as pd #importing pandas to read the csv file from pandas
import re
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import FuncFormatter
from dateutil import parser as dateparser
from pathlib import Path
# To integrate a Large Language Model
import google.generativeai as genai
import streamlit as st

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
# Display floats with two decimal places instead of scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)


# --- LLM Configuration ---
# To use a real LLM, you would uncomment the following lines and add your API key.
# You can get an API key from Google AI Studio: https://aistudio.google.com/
try:
    # Store your key in st.secrets for security
    genai.configure(api_key=st.secrets["google_api_key"])
    MODEL = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"Could not configure Generative AI. Is the API key set correctly? Error: {e}")
    MODEL = None


# Utility Function
CURRENCY_CLEAN_RE = re.compile(r"[,\s$]")
PARENS_NEG_RE = re.compile(r"^\((.*)\)$")

def to_number(x):
    """Convert Excel cell value to float; handles $, commas, parentheses negatives, blanks, and dashes."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "n/a", "-", "—", "–"}:
        return np.nan
    s = CURRENCY_CLEAN_RE.sub("", s)
    m = PARENS_NEG_RE.match(s)
    if m:
        try:
            return -float(m.group(1))
        except:
            pass
    try:
        return float(s)
    except:
        return np.nan

def coerce_date_from_header(h):
    """Try to parse a date from header text like 'as of march 31 2025 unaudited' or 'June 30, 2025'."""
    if h is None:
        return None
    s = str(h)
    s = s.replace("as of", "").replace("As of", "").replace("unaudited", "").replace("audited", "")
    s = s.replace("–", " ").replace("—", " ").replace("  ", " ").strip(" :\n\t")
    try:
        dt = dateparser.parse(s, dayfirst=False, fuzzy=True)
        return pd.to_datetime(dt.date())
    except Exception:
        return None

def format_currency(value, pos):
    """Formats a number into a currency string like $1.2M or -$500K."""
    sign = '-' if value < 0 else ''
    value = abs(value)
    if value >= 1_000_000:
        return f'{sign}${value / 1_000_000:.1f}M'
    if value >= 1_000:
        return f'{sign}${value / 1_000:.1f}K'
    return f'{sign}${value:,.0f}'

def first_nonempty_row(df):
    for i in range(min(10, len(df))):
        if df.iloc[i].notna().sum() >= 2:
            return i
    return 0

def read_any_excel(path):
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    raw = raw.dropna(axis=1, how="all")
    return raw

def find_label_column(df_body):
    """Heuristic to find the column with line item descriptions."""
    string_scores = {}
    # Analyze a sample of rows to find the most text-heavy column
    sample = df_body.head(20)
    for col_name in sample.columns:
        # A good indicator of a label column is having strings that aren't numbers
        is_text = sample[col_name].apply(
            lambda x: isinstance(x, str) and pd.isna(to_number(x))
        )
        string_scores[col_name] = is_text.sum()

    if not string_scores or max(string_scores.values()) == 0:
        # Fallback to the first column if no clear text column is found
        return df_body.columns[0]

    # The column with the highest score is likely the label column
    return max(string_scores, key=string_scores.get)

def melt_two_period_table(df):
    df = df.copy()
    start = first_nonempty_row(df)
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    df = df.iloc[start:, :].reset_index(drop=True)
    df = df.dropna(how="all")

    # --- Find header row and data start ---
    header_row_index = 0
    hdr = df.iloc[0]
    # Check first row for dates, tentatively assuming labels are not everywhere
    potential_dates = [coerce_date_from_header(h) for h in hdr]
    if sum(d is not None for d in potential_dates) < 1 and len(df) > 1:
        # If not enough dates, check second row
        header_row_index = 1
        hdr = df.iloc[header_row_index]

    data_start_row = header_row_index + 1
    df_body = df.iloc[data_start_row:]

    if df_body.empty:
        return pd.DataFrame(columns=["line_item", "date", "value"])

    # --- Dynamically find the label column and data columns ---
    label_col_name = find_label_column(df_body)
    data_col_names = [c for c in df.columns if c != label_col_name]

    # --- Parse dates from the identified header row for data columns ---
    dates = [coerce_date_from_header(hdr[c]) for c in data_col_names]

    # --- Melt the table ---
    long_rows = []
    for _, row in df_body.iterrows():
        item = str(row[label_col_name]).strip() if pd.notna(row[label_col_name]) else None
        if not item:
            continue
        for j, data_col in enumerate(data_col_names):
            val = to_number(row[data_col])
            long_rows.append({"line_item": item, "date": dates[j], "value": val})
    out = pd.DataFrame(long_rows)
    out = out.dropna(subset=["date", "line_item"]).reset_index(drop=True)
    return out

def parse_sectioned_table(df):
    df = df.copy()
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    start = first_nonempty_row(df)
    df = df.iloc[start:, :].reset_index(drop=True)
    df = df.dropna(how="all")

    # --- Find header row and data start ---
    hdr = df.iloc[0]
    data_start_row = 1
    # Check first row for dates
    potential_dates = [coerce_date_from_header(h) for h in hdr]
    if sum(d is not None for d in potential_dates) < 1:
        # No dates found, assume no header row with dates
        data_start_row = 0

    df_body = df.iloc[data_start_row:]
    if df_body.empty:
        return pd.DataFrame(columns=["section", "line_item", "date", "value"])

    # --- Dynamically find label column ---
    label_col_name = find_label_column(df_body)
    data_col_names = [c for c in df.columns if c != label_col_name]

    # --- Parse dates from the identified header row for data columns ---
    if data_start_row > 0:
        hdr = df.iloc[0]
        dates = [coerce_date_from_header(hdr[c]) for c in data_col_names]
    else:
        dates = [None] * len(data_col_names)

    # --- Parse sections and melt ---
    section = None
    rows = []
    for _, r in df_body.iterrows():
        label = r[label_col_name]
        if pd.isna(label):
            continue
        label_str = str(label).strip()
        if label_str.endswith(":"):
            section = label_str.rstrip(":").strip()
            continue
        for j, data_col in enumerate(data_col_names):
            dt = dates[j]
            val = to_number(r[data_col])
            rows.append({
                "section": section,
                "line_item": label_str,
                "date": dt,
                "value": val
            })
    out = pd.DataFrame(rows)
    out = out.dropna(subset=["line_item"])
    if "date" in out and out["date"].notna().any():
        out = out.dropna(subset=["date"])
    return out.reset_index(drop=True)

def extract_totals(df_long):
    if df_long.empty:
        return pd.DataFrame(columns=["date","line_item","value"])
    return df_long[df_long["line_item"].str.contains(r"(?i)total", na=False)].copy()

def brute_force_equity_parser(df):
    df = df.copy()
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    df = df.dropna(how="all")
    start = first_nonempty_row(df)
    df = df.iloc[start:, :]

    # Try to infer dates from header row
    hdr = df.iloc[0]
    dates = []
    for c in df.columns[1:]:
        dt = coerce_date_from_header(hdr[c])
        dates.append(dt if dt else pd.NaT)

    # If no dates were parsed, assign dummy sequential dates
    if all(pd.isna(d) for d in dates):
        dates = [pd.Timestamp(f"2025-01-{i+1:02d}") for i in range(len(df.columns) - 1)]
        df = df.iloc[1:, :]  # Skip header row
    else:
        df = df.iloc[1:, :]  # Skip header row

    rows = []
    for _, r in df.iterrows():
        label = str(r["c0"]).strip()
        if not label:
            continue
        for j, c in enumerate(df.columns[1:]):
            val = to_number(r[c])
            rows.append({
                "line_item": label,
                "date": dates[j],
                "value": val
            })

    out = pd.DataFrame(rows).dropna(subset=["line_item", "date"])
    out["statement"] = "equity"
    return out.reset_index(drop=True)

def parse_equity_custom(df):
    df = df.copy()
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    df = df.dropna(how="all")
    rows = []
    for _, r in df.iterrows():
        label = str(r.get("c4", "")).strip()  # e.g., "Balance at March 31, 2024"
        if not label or "Balance at" not in label:
            continue

        date = coerce_date_from_header(label)
        val_raw = r.get("c31", None)  # Total equity value
        val = to_number(val_raw)

        if date and not math.isnan(val):
            rows.append({
                "line_item": "Total Equity",
                "date": date,
                "value": val,
                "statement": "equity"
            })
            
    return pd.DataFrame(rows)
@st.cache_data
def load_and_process_data():
    """
    This function loads and processes all financial data from Excel files.
    The result is cached to improve performance.
    """
    # 🔧 Folder containing your Excel files
    DATA_DIR = Path("./10K_and_10Q_reportsCombined").expanduser().resolve()

    # --- 2) Flexible patterns (recursive, case-insensitive, handles .xls/.xlsx) ---
    RAW_PATTERNS = {
        "assets":       ["*asset*.xls*", "*balance*sheet*.xls*"],
        "liabilities":  ["*liab*.xls*", "*balance*sheet*.xls*"],
        "cashflows":    ["*cash*flow*.xls*"],
        "equity":       ["*equity*.xls*", "*stockholder*change*.xls*", "*shareholder*change*.xls*"],
        "intangibles":  ["*intang*.xls*"],
        "inventories":  ["*inventor*.xls*"],
        "loans":        ["*loan*.xls*", "*debt*.xls*"],
        "property":     ["*propert*.xls*", "*pp&e*.xls*", "*ppe*.xls*", "*plant*equipment*.xls*"]
    }

    # Helper: find first file that matches patterns
    def find_file(patterns):
        for pat in patterns:
            hits = list(DATA_DIR.glob(pat))
            if hits:
                return hits[0]
        return None

    files = {k: find_file(v) for k, v in RAW_PATTERNS.items()}

    assets_df = pd.DataFrame(columns=["line_item","date","value"])
    liab_df = pd.DataFrame(columns=["line_item","date","value"])
    cash_df = pd.DataFrame(columns=["section","line_item","date","value"])
    equity_df = pd.DataFrame(columns=["line_item","date","value"])
    intang_df = pd.DataFrame(columns=["line_item","date","value"])
    invent_df = pd.DataFrame(columns=["line_item","date","value"])
    loans_df = pd.DataFrame(columns=["line_item","date","value"])
    property_df = pd.DataFrame(columns=["line_item","date","value"])

    if files["assets"]:
        raw = read_any_excel(files["assets"])
        assets_df = melt_two_period_table(raw)
        assets_df["statement"] = "assets"

    if files["liabilities"]:
        raw = read_any_excel(files["liabilities"])
        liab_df = melt_two_period_table(raw)
        liab_df["statement"] = "liabilities"

    if files["cashflows"]:
        raw = read_any_excel(files["cashflows"])
        cash_df = parse_sectioned_table(raw)
        cash_df["statement"] = "cashflows"

    if files["equity"]:
        raw = read_any_excel(files["equity"])
        equity_df = parse_equity_custom(raw)

    if files["intangibles"]:
        raw = read_any_excel(files["intangibles"])
        intang_df = melt_two_period_table(raw)
        intang_df["statement"] = "intangibles"

    if files["inventories"]:
        raw = read_any_excel(files["inventories"])
        invent_df = melt_two_period_table(raw)
        invent_df["statement"] = "inventories"

    if files["loans"]:
        raw = read_any_excel(files["loans"])
        loans_df = melt_two_period_table(raw)
        loans_df["statement"] = "loans"

    if files["property"]:
        raw = read_any_excel(files["property"])
        property_df = melt_two_period_table(raw)
        property_df["statement"] = "property"

    # Normalise into a unified long table
    assets_like = pd.concat([df for df in [assets_df, liab_df, equity_df, intang_df, invent_df, loans_df, property_df] if not df.empty], ignore_index=True)
    assets_like = assets_like.rename(columns={"line_item":"line_item","date":"date","value":"value","statement":"statement"})
    assets_like["date"] = pd.to_datetime(assets_like["date"])

    cash_like = cash_df.rename(columns={"line_item":"line_item","date":"date","value":"value","statement":"statement","section":"section"})
    if not cash_like.empty:
        cash_like["date"] = pd.to_datetime(cash_like["date"])

    master = pd.concat([assets_like, cash_like], ignore_index=True, sort=False)
    master = master.dropna(subset=["date", "line_item"]).reset_index(drop=True)
    master["line_item_clean"] = master["line_item"].str.strip().str.replace(r"\s+", " ", regex=True)

    # Derivations: total & groupings
    totals = (master[master["line_item_clean"].str.contains("(?i)total", na=False)]
              .groupby(["statement","date"], as_index=False)["value"].sum()
              .rename(columns={"value":"total_value"}))

    def total_for(stmt, dt):
        val = totals.query("statement == @stmt and date == @dt")["total_value"]
        return float(val.iloc[0]) if len(val) else np.nan

    dates_sorted = sorted(master["date"].dropna().unique())
    latest_date = dates_sorted[-1] if dates_sorted else None

    summary_rows = []
    for dt in dates_sorted:
        a = total_for("assets", dt)
        l = total_for("liabilities", dt)
        e = total_for("equity", dt)
        summary_rows.append({"date": dt, "total_assets": a, "total_liabilities": l, "total_equity": e,
                             "balance_diff": a - (l + e) if (not math.isnan(a) and (not math.isnan(l) or not math.isnan(e))) else np.nan})
    balance_summary = pd.DataFrame(summary_rows)
    balance_summary[["total_assets", "total_liabilities", "total_equity"]] = balance_summary[["total_assets", "total_liabilities", "total_equity"]].fillna(0)

    # Key Ratios
    def find_value(df, pattern, dt, stmt=None):
        q = df[df["line_item_clean"].str.contains(pattern, case=False, na=False)]
        if stmt:
            q = q[q["statement"]==stmt]
        q = q[q["date"]==dt]
        return float(q["value"].sum()) if len(q) else np.nan

    ratio_rows = []
    for dt in dates_sorted:
        total_assets = total_for("assets", dt)
        total_liab = total_for("liabilities", dt)
        total_equity = total_for("equity", dt)
        debt_to_equity = (total_liab / total_equity) if (total_liab and total_equity and not math.isnan(total_liab) and not math.isnan(total_equity) and total_equity != 0) else np.nan

        current_assets = find_value(master, r"^current assets$", dt, "assets")
        current_liab = find_value(master, r"^current liabilities$", dt, "liabilities")
        inventories = find_value(master, r"^inventories?$", dt)
        cash = find_value(master, r"^cash( and cash equivalents)?$", dt)

        current_ratio = (current_assets / current_liab) if (current_assets and current_liab and current_liab != 0 and not math.isnan(current_assets) and not math.isnan(current_liab)) else np.nan
        quick_ratio = ((current_assets - inventories) / current_liab) if (current_assets and current_liab and not math.isnan(inventories) and current_liab != 0) else np.nan

        ratio_rows.append({
            "date": dt,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "cash": cash,
            "inventories": inventories,
            "total_assets": total_assets,
            "total_liabilities": total_liab,
            "total_equity": total_equity
        })
    ratios = pd.DataFrame(ratio_rows)

    # Cashflow breakdown
    if not cash_like.empty:
        cash_summary = (cash_like.groupby(["date","section"], as_index=False)["value"].sum()
                                  .pivot(index="date", columns="section", values="value")
                                  .reset_index().rename_axis(None, axis=1)).fillna(0)
    else:
        cash_summary = pd.DataFrame()

    return {
        "master": master,
        "balance_summary": balance_summary,
        "ratios": ratios,
        "cash_summary": cash_summary,
        "latest_date": latest_date,
    }

def generate_response_rule_based(prompt, data):
    """
    A simple rule-based response generator. This is kept as a fallback or for simple queries.
    """
    prompt = prompt.lower()
    balance_summary = data["balance_summary"]
    ratios = data["ratios"]

    def fmt_money(x):
        if x is None or pd.isna(x):
            return "n/a"
        return f"${x:,.0f}"

    def trend_sentence(series, label):
        series = [s for s in series if s is not None and not pd.isna(s)]
        if len(series) < 2:
            return f"{label} trend data insufficient."
        delta = series[-1] - series[0]
        sign = "increased" if delta > 0 else "decreased"
        return f"{label} {sign} by {fmt_money(abs(delta))} over the period."

    if "equity" in prompt:
        if balance_summary.empty:
            return "Equity data is not available."

        last = balance_summary.sort_values("date").iloc[-1]
        e = last["total_equity"]
        response = f"### Equity Snapshot ({pd.to_datetime(last['date']).date()})\n"
        response += f"- **Total Equity:** {fmt_money(e)}\n"

        if len(balance_summary) >= 2:
            first = balance_summary.sort_values("date").iloc[0]
            response += f"\n**Change from {pd.to_datetime(first['date']).date()} to {pd.to_datetime(last['date']).date()}:**\n"
            response += trend_sentence(balance_summary.sort_values("date")["total_equity"].tolist(), "Total equity")
        return response

    elif "asset" in prompt:
        if balance_summary.empty:
            return "Asset data is not available."

        last = balance_summary.sort_values("date").iloc[-1]
        a = last["total_assets"]
        response = f"### Assets Snapshot ({pd.to_datetime(last['date']).date()})\n"
        response += f"- **Total Assets:** {fmt_money(a)}\n"
        if len(balance_summary) >= 2:
            first = balance_summary.sort_values("date").iloc[0]
            response += f"\n**Change from {pd.to_datetime(first['date']).date()} to {pd.to_datetime(last['date']).date()}:**\n"
            response += trend_sentence(balance_summary.sort_values("date")["total_assets"].tolist(), "Total assets")

        latest_date = data["latest_date"]
        if latest_date:
            assets_latest = data["master"][(data["master"]["statement"]=="assets") & (data["master"]["date"]==latest_date)]
            comp = (assets_latest[~assets_latest["line_item_clean"].str.contains("(?i)total", na=False)]
                    .sort_values("value", ascending=False).head(5))
            if not comp.empty:
                bullets = "\n".join([f"- {r['line_item']}: {fmt_money(r['value'])}" for _, r in comp.iterrows()])
                response += f"\n\n### Top Asset Components — {latest_date.date()}\n{bullets}"
        return response

    elif "ratio" in prompt and ("debt" in prompt or "liabilities" in prompt) and "equity" in prompt:
        if ratios.empty:
            return "Ratio data is not available."

        rlast = ratios.sort_values("date").iloc[-1]
        response = f"### Key Ratios ({pd.to_datetime(rlast['date']).date()})\n"
        if not pd.isna(rlast["debt_to_equity"]):
            response += f"- **Debt-to-Equity:** {rlast['debt_to_equity']:.2f}"
        else:
            response += "- Debt-to-Equity ratio could not be calculated."
        return response

    else:
        return "I can provide information on assets, equity, and debt-to-equity ratios. What would you like to know?"

def format_data_for_llm(data):
    """Formats a subset of the financial data into a string for the LLM prompt."""
    if not data or data["master"].empty:
        return "No data available."

    latest_date = data.get('latest_date')
    if not latest_date:
        return "No date information available."

    # Select data for the latest date
    latest_balance = data['balance_summary'][data['balance_summary']['date'] == latest_date]
    latest_ratios = data['ratios'][data['ratios']['date'] == latest_date]
    latest_cash = data['cash_summary'][data['cash_summary']['date'] == latest_date] if not data['cash_summary'].empty else pd.DataFrame()

    # Build the context string
    context_parts = [f"Here is the financial data summary as of {latest_date.date()}:"]

    if not latest_balance.empty:
        context_parts.append("\nBalance Sheet Summary:\n" + latest_balance.to_string(index=False))
    if not latest_ratios.empty:
        context_parts.append("\nKey Ratios:\n" + latest_ratios.to_string(index=False))
    if not latest_cash.empty:
        context_parts.append("\nCash Flow Summary:\n" + latest_cash.to_string(index=False))

    context_parts.append("\nUse the data above to answer the user's question. Act as a helpful financial analyst assistant.")
    return "\n".join(context_parts)

def generate_response(prompt, data):
    """
    Generates a response using a Large Language Model for natural language understanding.
    Falls back to a rule-based system if the LLM is not available.
    """

    if MODEL is None:
        # Fallback to the simpler rule-based system if the LLM is not configured
        st.warning("LLM not configured. Falling back to rule-based responses. For advanced analysis, please configure an API key.", icon="⚠️")
        return generate_response_rule_based(prompt, data)

    try:
        # 1. Format the relevant data into a string for the LLM
        data_context = format_data_for_llm(data)

        # 2. Create a prompt for the LLM, combining the data and the user's question
        llm_prompt = f"""
{data_context}

User question: "{prompt}"

Answer:
"""
        # 3. Call the LLM
        response = MODEL.generate_content(llm_prompt)
        return response.text

    except Exception as e:
        st.error(f"An error occurred while communicating with the LLM: {e}")
        return generate_response_rule_based(prompt, data) # Fallback on error


st.set_page_config(page_title="GenAI Financial Chatbot", layout="wide")
st.title("GenAI Financial Analysis Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "How can I help you analyze the financial documents?",
        }
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load data
processed_data = load_and_process_data()

# Accept user input
if prompt := st.chat_input("Ask about assets, equity, or ratios..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            if not processed_data or processed_data["master"].empty:
                response = "I could not find any data to analyze. Please check that the Excel files are in the `10K_and_10Q_reportsCombined` folder and try again."
            else:
                response = generate_response(prompt, processed_data)
        st.markdown(response)

    # Add assistant's response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    