import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# === Paths ===
# === Paths ===
OUTPUT_ROOT = Path(r"C:\Users\Narasimhan\OneDrive - vf3wg\Stock_Prediction_Project\output")
COMPILED_CSV = r"C:\Users\Narasimhan\OneDrive - vf3wg\Stock_Prediction_Project\output\compiled_shortlong_allhorizons_20250830_095747.csv"


# === Utility functions ===
def _ensure_df(path=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("No compiled CSV found. Run compiler in Jupyter first.")
    df = pd.read_csv(p)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.upper().str.replace(".NS","", regex=False)
    return df

def compute_budget(mode, direct_amount, monthly_income, annual_income, risk_cap_pct, reserve_pct):
    if mode == "Direct":
        base = float(direct_amount)
    elif mode == "Monthly":
        base = float(monthly_income) * (risk_cap_pct/100.0)
    else:
        base = (float(annual_income)/12.0) * (risk_cap_pct/100.0)
    return int(base * (1 - reserve_pct/100.0))

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
        return pd.DataFrame([{"invested":0,"leftover_cash":budget,"positions":0,"avg_price":np.nan}])
    invested = int(alloc_df["cost"].sum())
    avgp = np.average(alloc_df["price"], weights=alloc_df["shares"])
    return pd.DataFrame([{"invested":invested,"leftover_cash":budget-invested,
                          "positions":len(alloc_df),"avg_price":avgp}])

# ================== STREAMLIT APP ==================
st.title("📱 Portfolio Allocator")

with st.container():
    st.subheader("Inputs")

    # Row 1 - Budget mode & Amount
    c1, c2 = st.columns([1,2])
    mode = c1.radio("Budget type", ["Direct","Monthly","Annual"])
    direct_amount = c2.number_input("Amount ₹", value=10000, step=500)

    # Row 2 - Monthly & Annual
    c3, c4 = st.columns(2)
    monthly = c3.number_input("Monthly Income ₹", value=20000, step=1000)
    annual = c4.number_input("Annual Income ₹", value=0, step=10000)

    # Row 3 - Risk & Reserve
    c5, c6 = st.columns(2)
    riskcap = c5.slider("Risk cap %", 0.0,50.0,5.0,0.5)
    reserve = c6.slider("Reserve %",0.0,90.0,10.0,1.0)

    # Row 4 - Max pos, per cap, min return
    c7, c8, c9 = st.columns(3)
    maxpos = c7.slider("Max positions",1,20,8)
    percap = c8.slider("Per-stock cap %",5,100,20)
    minret = c9.slider("Min expected return",0.0,1.0,0.0,0.01)

    horizons = st.multiselect("Base horizons", ["m24","m12","m6","m3","m1","d10","d5","d2","d1"], default=["m12","m6","m3"])

    run = st.button("▶ Run Allocation")

if run:
    try:
        compiled = _ensure_df(COMPILED_CSV)

        # Build universe
        uni = compiled.copy()
        uni = uni[["symbol","closing_stock","pred_safe_m12"]].rename(columns={"closing_stock":"price","pred_safe_m12":"pred_value"})
        uni["exp_ret"] = uni["pred_value"]/uni["price"] - 1.0
        uni = uni[uni["exp_ret"] >= minret]

        # Budget
        B = compute_budget(mode, direct_amount, monthly, annual, riskcap, reserve)

        # Allocation
        alloc = greedy_allocate(uni, B, maxpos, percap)
        summary = summarize_allocation(alloc, B)

        st.subheader("📌 Allocation Summary")
        st.table(summary)

        st.subheader("📌 Allocated Stocks")
        st.dataframe(alloc)

    except Exception as e:
        st.error(f"Error: {e}")
        st.text(traceback.format_exc())
