import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

from pathlib import Path
import pandas as pd

# Use relative path instead of hardcoding your Windows path
OUTPUT_ROOT = Path(__file__).parent  # current folder where appmain.py lives
COMPILED_CSV = OUTPUT_ROOT / "compiled_shortlong_allhorizons_20250830_095747.csv"

def _ensure_df(path=None) -> pd.DataFrame:
    p = Path(path) if path else COMPILED_CSV
    if not p.exists():
        raise FileNotFoundError(f"No compiled CSV found at {p}. Upload it to GitHub repo.")
    df = pd.read_csv(p)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(".NS","", regex=False)
    return df


def _pref_column(df, prefixes):
    pred_cols = [c for c in df.columns if c.startswith("pred_safe_")]
    for h in prefixes:
        c = f"pred_safe_{h}"
        if c in df.columns:
            return c, pred_cols
    if pred_cols:
        return pred_cols[0], pred_cols
    raise KeyError("No pred_safe_* columns found.")

def compute_budget(direct_amount=None, monthly_income=None, annual_income=None, risk_cap_pct=5.0, reserve_pct=10.0):
    if direct_amount is not None and direct_amount > 0:
        base = float(direct_amount)
    elif monthly_income is not None and monthly_income > 0:
        base = float(monthly_income) * (risk_cap_pct/100.0)
    elif annual_income is not None and annual_income > 0:
        base = (float(annual_income)/12.0) * (risk_cap_pct/100.0)
    else:
        return 0
    return int(base * (1 - reserve_pct/100.0))

def build_universe(compiled_df, preferred_horizons):
    df = compiled_df.copy()
    base_col, all_pred_cols = _pref_column(df, [str(h).lower() for h in preferred_horizons])
    df = df[df[base_col].notna() & df["closing_stock"].notna()].copy()
    df["price"] = pd.to_numeric(df["closing_stock"], errors="coerce")
    df["pred_value"] = pd.to_numeric(df[base_col], errors="coerce")
    df = df[df["price"] > 0]
    df["exp_ret"] = df["pred_value"]/df["price"] - 1.0
    keep_cols = ["symbol","price","pred_value","exp_ret","Sector","MarketCapCategory","VolatilityCategory","BaseDate","BaseHorizon"]
    uni = df[[c for c in keep_cols if c in df.columns]].drop_duplicates("symbol").reset_index(drop=True)
    return uni, base_col, all_pred_cols

def filter_positive(uni, min_exp_ret=0.0): 
    return uni[uni["exp_ret"] >= float(min_exp_ret)].copy()

def filter_affordable(uni, budget, per_stock_cap_pct):
    cap_rupees = budget*(per_stock_cap_pct/100.0)
    return uni[uni["price"] <= cap_rupees].copy()

def greedy_allocate(df, budget, max_positions=8, per_stock_cap_pct=20.0):
    d = df.sort_values("exp_ret", ascending=False).reset_index(drop=True)
    picks, cash = [], budget
    per_cap_rupees = int(budget*(per_stock_cap_pct/100.0))
    for _, r in d.iterrows():
        if len(picks) >= max_positions or cash < r["price"]: break
        alloc = min(per_cap_rupees, cash)
        sh = int(alloc//r["price"])
        if sh <= 0: continue
        cost = int(sh*r["price"]); cash -= cost
        picks.append({"symbol":r["symbol"], "price":float(r["price"]), "shares":int(sh),
                      "cost":cost, "pred_value":float(r["pred_value"]), "exp_ret":float(r["exp_ret"])})
    out = pd.DataFrame(picks)
    if not out.empty:
        out["weight_%"] = (out["cost"]/budget*100).round(1)
        out["leftover_cash"] = budget - out["cost"].sum()
    return out

def summarize_allocation(alloc_df, budget):
    if alloc_df.empty: 
        return {"invested":0,"leftover_cash":budget,"positions":0,"avg_price":np.nan}
    invested = int(alloc_df["cost"].sum())
    avgp = np.average(alloc_df["price"], weights=alloc_df["shares"])
    return {"invested":invested,"leftover_cash":budget-invested,
            "positions":len(alloc_df),"avg_price":avgp}

def evaluate_portfolio_value_from_compiled(alloc_df, compiled_df):
    if alloc_df.empty:
        return pd.DataFrame()

    sym = alloc_df["symbol"].tolist()
    shares = alloc_df.set_index("symbol")["shares"]
    cost_today = int(alloc_df["cost"].sum())

    horizon_map = {
        "d1": "Return after 1 Day (%)",
        "d2": "Return after 2 Days (%)",
        "d5": "Return after 5 Days (%)",
        "d10": "Return after 10 Days (%)",
        "m1": "Return after 1 Month (%)",
        "m3": "Return after 3 Months (%)",
        "m6": "Return after 6 Months (%)",
        "m12": "Return after 1 Year (%)",
        "m24": "Return after 2 Years (%)",
    }

    results = {}
    pred_cols = sorted([c for c in compiled_df.columns if c.startswith("pred_safe_")])

    for c in pred_cols:
        h = c.replace("pred_safe_", "")
        vals = compiled_df.set_index("symbol")[c].reindex(sym)
        fv = (shares * vals).fillna(0).sum()
        if cost_today > 0:
            ret_pct = (fv / cost_today - 1) * 100
            col_name = horizon_map.get(h, f"Return {h} (%)")

            # 👇 Format rule
            if round(ret_pct, 0) == 0:
                results[col_name] = f"{ret_pct:.2f}"
            else:
                results[col_name] = f"{ret_pct:.0f}"
        else:
            col_name = horizon_map.get(h, f"Return {h} (%)")
            results[col_name] = "NaN"

    return pd.DataFrame([results])

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Portfolio Allocator", layout="wide")

st.title("📊 Portfolio Allocator App")
st.caption("Built from your Jupyter outputs")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Budget mode", ["Direct","Monthly","Annual"])
direct = st.sidebar.number_input("Direct Amount ₹", value=10000, step=500)
monthly = st.sidebar.number_input("Monthly Income ₹", value=20000, step=1000)
annual = st.sidebar.number_input("Annual Income ₹", value=0, step=10000)
riskcap = st.sidebar.slider("Risk cap %", 0.0,50.0,5.0,0.5)
reserve = st.sidebar.slider("Reserve %",0.0,90.0,10.0,1.0)

st.sidebar.header("Allocator")
maxpos = st.sidebar.slider("Max positions",1,20,8)
percap = st.sidebar.slider("Per-stock cap %",5,100,20)
minret = st.sidebar.slider("Min expected return",0.0,1.0,0.0,0.01)
preferred_horizons = st.sidebar.multiselect("Preferred horizons",
    ["m24","m12","m6","m3","m1","d10","d5","d2","d1"], default=["m12","m6","m3"])

if st.button("▶ Run Allocation"):
    try:
        compiled = _ensure_df(COMPILED_CSV)
        uni, base_col, _ = build_universe(compiled, preferred_horizons)
        uni = filter_positive(uni,minret)

        B = compute_budget(direct_amount=direct if mode=="Direct" else None,
                           monthly_income=monthly if mode=="Monthly" else None,
                           annual_income=annual if mode=="Annual" else None,
                           risk_cap_pct=riskcap,reserve_pct=reserve)

        uni = filter_affordable(uni,B,percap)
        alloc = greedy_allocate(uni,B,maxpos,percap)
        summary = summarize_allocation(alloc,B)
        val = evaluate_portfolio_value_from_compiled(alloc,compiled)

        # --- Dashboard style layout ---
        st.subheader("📌 Portfolio Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Invested", f"₹{summary['invested']:,}")
        c2.metric("Leftover Cash", f"₹{summary['leftover_cash']:,}")
        c3.metric("Positions", summary['positions'])
        c4.metric("Avg Price", f"{summary['avg_price']:.2f}" if not np.isnan(summary['avg_price']) else "-")

        st.subheader("📈 Allocation")
        st.dataframe(alloc)

        st.subheader("🔮 Projected Portfolio Value")
        st.dataframe(val)

        st.subheader("📊 Universe (Filtered)")
        st.dataframe(uni)

    except Exception as e:
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())
