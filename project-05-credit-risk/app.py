"""
Streamlit dashboard for the credit-decision model.

The point of this dashboard is that the cost assumption is a control, not a
constant. Move the ratio in the sidebar and the threshold, the approval
rate, the money and the disparity all move with it -- which is the argument
of the project made operable rather than asserted.

Everything is recomputed from output/reports/test_scores.csv.gz, the actual
held-out predictions for all 7,500 test accounts. No model is loaded and
nothing is refitted, so the app runs from a clean clone with no training.

Author: Manuel Corona
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "output" / "reports"

st.set_page_config(page_title="Credit decisions", page_icon="=", layout="wide")

# Categorical slots 1-3 of the reference palette, validated all-pairs for
# both normal vision and CVD. Three is the cap for that guarantee, so no
# chart here carries more than three series; anything further is a facet or
# a table.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "rgba(82,81,78,0.18)"

MODEL_LABELS = {
    "base_rate": "Base rate (floor)",
    "delinquency_rule": "Delinquency rule (one column)",
    "logistic": "Logistic regression",
    "logistic_balanced": "Logistic, class_weight=balanced",
    "random_forest": "Random forest",
    "hist_gbm": "Gradient boosting",
    "hist_gbm_calibrated": "Gradient boosting, isotonic",
    "blinded": "Gradient boosting, demographics removed",
}
LADDER = ["base_rate", "delinquency_rule", "logistic", "logistic_balanced",
          "random_forest", "hist_gbm", "hist_gbm_calibrated"]

GROUP_LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate school", 2: "University", 3: "High school",
                  4: "Other/unknown"},
    "AGE_BAND": {b: b for b in ("21-29", "30-39", "40-49", "50+")},
}


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "test_scores.csv.gz")


@st.cache_data
def load_report(name: str):
    path = REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


# ---------------------------------------------------------------- policy --

def regret(y, p, t, ratio, weight=None):
    w = np.ones(len(y)) if weight is None else weight
    flag = p >= t
    return float(w[flag & (y == 0)].sum() + w[(~flag) & (y == 1)].sum() * ratio)


def rates(y, p, t):
    flag = p >= t
    pos = y == 1
    tp, fp = int((flag & pos).sum()), int((flag & ~pos).sum())
    fn, tn = int((~flag & pos).sum()), int((~flag & ~pos).sum())
    div = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "decline_rate": div(tp + fp, len(y)), "recall": div(tp, tp + fn),
            "fpr": div(fp, fp + tn), "precision": div(tp, tp + fp)}


# ------------------------------------------------------------------ tabs --

def tab_decision(df, ratio, threshold):
    y = df["default"].to_numpy()
    p = df["p_hist_gbm"].to_numpy()
    exposure = df["exposure"].to_numpy()
    n_good, n_bad = int((y == 0).sum()), int((y == 1).sum())

    r_policy = regret(y, p, threshold, ratio)
    r_half = regret(y, p, 0.5, ratio)
    decline_all, approve_all = float(n_good), float(n_bad * ratio)
    best_naive = min(decline_all, approve_all)

    st.markdown(
        f"A declined good customer costs **1 unit** of margin; a missed "
        f"default costs **{ratio:.1f}**. Expected value says approve while the "
        f"probability of default is below **{threshold:.3f}**.")

    c = st.columns(4)
    c[0].metric("Cost-optimal threshold", f"{threshold:.3f}")
    c[1].metric("Cost of this policy", f"{r_policy:,.0f}",
                f"{r_policy - best_naive:+,.0f} vs best no-model policy",
                delta_color="inverse")
    c[2].metric("Cost at a 0.5 cutoff", f"{r_half:,.0f}",
                f"{r_half / r_policy - 1:+.0%}", delta_color="inverse")
    c[3].metric("Applicants declined", f"{rates(y, p, threshold)['decline_rate']:.1%}")

    grid = np.linspace(0.01, 0.99, 197)
    curve = [regret(y, p, t, ratio) for t in grid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", name="Model policy",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="threshold %{x:.3f}<br>cost %{y:,.0f}<extra></extra>"))
    fig.add_hline(y=decline_all, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"decline everyone · {decline_all:,.0f}",
                  annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    fig.add_vline(x=threshold, line=dict(color=AQUA, width=2),
                  annotation_text=f"cost-optimal · {threshold:.3f}",
                  annotation_position="top right",
                  annotation_font=dict(color=MUTED, size=11))
    fig.add_vline(x=0.5, line=dict(color=ORANGE, width=2, dash="dash"),
                  annotation_text="the default 0.5",
                  annotation_position="top right",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_xaxes(title="Decline when probability of default is at least ...")
    fig.update_yaxes(title="Cost (units of margin)")
    st.plotly_chart(layout(fig, 420), use_container_width=True)

    st.markdown("##### The same model at three thresholds")
    rows = []
    for label, t in [("Cost-optimal", threshold), ("The default 0.5", 0.5),
                     ("Base rate", float(y.mean()))]:
        rr = rates(y, p, t)
        rows.append({
            "Policy": label, "Threshold": round(t, 3),
            "Declined": f"{rr['decline_rate']:.1%}",
            "Defaults caught": f"{rr['recall']:.1%}",
            "Good customers declined": f"{rr['fpr']:.1%}",
            "Cost": f"{regret(y, p, t, ratio):,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "Declining everyone costs "
        f"{decline_all:,.0f}; approving everyone costs {approve_all:,.0f}. "
        "A model is only worth deploying if it beats both — and at a 0.5 "
        "cutoff, at this cost ratio, it does not.")


def tab_ladder(df, ratio, threshold):
    y = df["default"].to_numpy()
    st.markdown(
        "Every model at the cheapest threshold this split allows — a best "
        "case, since it assumes the threshold was already known. Even so, "
        "**AUC orders these models differently than money does**, and two of "
        "them, with respectable AUCs, are worth nothing at all.")

    n_good = int((y == 0).sum())
    rows = []
    for key in LADDER:
        p = df[f"p_{key}"].to_numpy()
        best_t = min(np.linspace(0.01, 0.99, 197),
                     key=lambda t: regret(y, p, t, ratio))
        cost = regret(y, p, float(best_t), ratio)
        auc = _auc(y, p)
        rows.append({
            "Model": MODEL_LABELS[key], "AUC": round(auc, 4),
            "Threshold": round(float(best_t), 3),
            "Cost": round(cost, 1),
            "Saved vs no model": round(n_good - cost, 1),
            "Worth deploying": "yes" if cost < n_good - 1e-9 else "no",
        })
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)

    # Bars ordered by AUC, with the AUC on the axis label. A scatter of AUC
    # against money is the obvious form and it fails here: the three best
    # models sit within 0.005 AUC of each other, so their labels collide
    # into an unreadable stack. Ordering the axis by AUC keeps the same
    # comparison and cannot collide.
    ordered = table.sort_values("AUC")
    labels = [f"{m}   ·   AUC {a:.4f}"
              for m, a in zip(ordered["Model"], ordered["AUC"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ordered["Saved vs no model"], y=labels, orientation="h",
        marker=dict(color=BLUE, line=dict(color="#fcfcfb", width=2)),
        text=[f"{v:,.0f}" for v in ordered["Saved vs no model"]],
        textposition="outside", textfont=dict(color=MUTED, size=11),
        hovertemplate="%{y}<br>saved %{x:,.0f}<extra></extra>", showlegend=False))
    fig.update_xaxes(title="Money saved versus the best no-model policy",
                     range=[0, float(ordered["Saved vs no model"].max()) * 1.18])
    fig.update_yaxes(title=None, ticksuffix="  ")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(layout(fig, 420), use_container_width=True)
    st.caption(
        "Read down the axis and the AUC rises steadily; read the bars and "
        "the money does not. The bottom two are flat at zero — at this cost "
        "ratio their cheapest policy is to decline everyone, so they add "
        "nothing, and one of them has an AUC of 0.72. Between AUC 0.74 and "
        "0.79 the saving grows five-fold. Ranking quality and money are not "
        "the same axis, and one cannot be read off the other.")


def tab_capacity(df, ratio, threshold):
    y = df["default"].to_numpy()
    p = df["p_hist_gbm"].to_numpy()
    exposure = df["exposure"].to_numpy()
    total = float(exposure[y == 1].sum())

    st.markdown(
        "A different question: a review team can only look at **K** accounts. "
        "Spend that capacity on the most *likely* defaults, or on the most "
        "*expensive* ones?")
    k_pct = st.slider("Share of accounts reviewed", 1, 40, 5, 1,
                      format="%d%%") / 100
    k = max(1, int(round(len(y) * k_pct)))

    by_p = np.argsort(-p)[:k]
    by_ev = np.argsort(-(p * exposure))[:k]
    res = {}
    for name, sel in [("By probability", by_p), ("By expected loss", by_ev)]:
        hit = y[sel] == 1
        res[name] = {"caught": int(hit.sum()),
                     "prevented": float(exposure[sel][hit].sum())}

    c = st.columns(3)
    c[0].metric("Accounts reviewed", f"{k:,}")
    c[1].metric("Loss prevented — by probability",
                f"NT${res['By probability']['prevented']:,.0f}",
                f"{res['By probability']['caught']} defaults caught")
    c[2].metric("Loss prevented — by expected loss",
                f"NT${res['By expected loss']['prevented']:,.0f}",
                f"{res['By expected loss']['caught']} defaults caught")

    fig = go.Figure()
    names = list(res)
    fig.add_trace(go.Bar(
        x=names, y=[res[n]["prevented"] for n in names],
        marker=dict(color=[BLUE, ORANGE],
                    line=dict(color="#fcfcfb", width=2)),
        text=[f"NT${res[n]['prevented']:,.0f}" for n in names],
        textposition="outside", textfont=dict(color=MUTED, size=12),
        hovertemplate="%{x}<br>NT$%{y:,.0f}<extra></extra>", showlegend=False))
    fig.add_hline(y=total, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"all loss at risk · NT${total:,.0f}",
                  annotation_position="top left",
                  annotation_font=dict(color=MUTED, size=11))
    fig.update_yaxes(title="Loss prevented (NT$)")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(layout(fig, 380), use_container_width=True)

    delta = res["By expected loss"]["prevented"] / res["By probability"]["prevented"] - 1
    fewer = res["By probability"]["caught"] - res["By expected loss"]["caught"]
    st.caption(
        f"At this capacity, ranking by expected loss prevents **{delta:+.0%}** "
        f"more money while catching **{fewer} fewer** defaults. Recall, "
        "precision and AUC all prefer the other ordering. None of them knows "
        "that some balances are ten times the size of others.")


def tab_fairness(df, ratio, threshold):
    y = df["default"].to_numpy()
    p = df["p_hist_gbm"].to_numpy()
    attr = st.selectbox("Protected attribute", ["SEX", "AGE_BAND", "EDUCATION"],
                        format_func=lambda a: {"SEX": "Sex", "AGE_BAND": "Age band",
                                               "EDUCATION": "Education"}[a])
    groups = df[attr].to_numpy()
    labels = GROUP_LABELS[attr]

    rows = []
    for g in sorted(pd.unique(groups), key=str):
        m = groups == g
        n, pos = int(m.sum()), int(y[m].sum())
        rr = rates(y[m], p[m], threshold)
        rows.append({
            "Group": str(labels.get(g, g)), "n": n,
            "Default rate": y[m].mean(),
            "Declined": rr["decline_rate"], "Defaults caught": rr["recall"],
            "Good customers declined": rr["fpr"],
            "reliable": n >= 300 and pos >= 50,
        })
    table = pd.DataFrame(rows)
    shown = table[table["reliable"]]

    st.markdown(
        f"One threshold ({threshold:.3f}) applied to everyone. The model "
        "ranks about equally well inside every group — the error rates still "
        "come out different, because the groups default at different rates.")

    fig = go.Figure()
    for name, colour, col in [("Declined", BLUE, "Declined"),
                              ("Defaults caught", ORANGE, "Defaults caught"),
                              ("Good customers declined", AQUA,
                               "Good customers declined")]:
        fig.add_trace(go.Bar(
            x=shown["Group"], y=shown[col], name=name,
            marker=dict(color=colour, line=dict(color="#fcfcfb", width=2)),
            text=[f"{v:.1%}" for v in shown[col]], textposition="outside",
            textfont=dict(color=MUTED, size=11),
            hovertemplate="%{x}<br>" + name + " %{y:.1%}<extra></extra>"))
    fig.update_yaxes(title="Rate", tickformat=".0%", range=[0, 1.15])
    fig.update_layout(barmode="group", hovermode="closest")
    st.plotly_chart(layout(fig, 400), use_container_width=True)

    display = shown.drop(columns=["reliable"]).copy()
    for col in ("Default rate", "Declined", "Defaults caught",
                "Good customers declined"):
        display[col] = display[col].map("{:.1%}".format)
    st.dataframe(display, hide_index=True, use_container_width=True)
    dropped = table[~table["reliable"]]["Group"].tolist()
    if dropped:
        st.caption(f"Excluded as too small to report on: {', '.join(dropped)}.")

    audit = load_report("fairness.json")
    if audit:
        a = audit["audit"].get(attr, {})
        eo = a.get("cost_of_equal_opportunity", {})
        dp = a.get("cost_of_demographic_parity", {})
        st.markdown("##### What it costs to close the gap")
        st.markdown(
            f"Measured once at the threshold frozen on validation "
            f"({audit['threshold']:.3f}), not at the slider's: giving every "
            f"group the same "
            f"**recall** costs **{eo.get('extra_regret_pct', float('nan')):+.2f}%** "
            f"of the model's value; giving every group the same **decline "
            f"rate** costs **{dp.get('extra_regret_pct', float('nan')):+.2f}%**. "
            "Removing sex, age, education and marital status from the model "
            "entirely costs more than either, and closes less of the gap.")


def tab_accounts(df, ratio, threshold):
    y = df["default"].to_numpy()
    p = df["p_hist_gbm"].to_numpy()
    st.markdown(
        "The held-out accounts themselves, with the decision the current "
        "policy would make and what it cost.")
    c1, c2 = st.columns(2)
    outcome = c1.selectbox("Show", ["All", "Missed defaults (expensive)",
                                    "Declined good customers", "Correct decisions"])
    order = c2.selectbox("Sort by", ["Expected loss", "Probability of default",
                                     "Balance at risk"])

    view = df.copy()
    view["Decision"] = np.where(p >= threshold, "Decline", "Approve")
    view["Outcome"] = np.where(y == 1, "Defaulted", "Repaid")
    view["Expected loss"] = p * view["exposure"]
    correct = (p >= threshold) == (y == 1)
    if outcome == "Missed defaults (expensive)":
        view = view[(p < threshold) & (y == 1)]
    elif outcome == "Declined good customers":
        view = view[(p >= threshold) & (y == 0)]
    elif outcome == "Correct decisions":
        view = view[correct]
    key = {"Expected loss": "Expected loss", "Probability of default": "p_hist_gbm",
           "Balance at risk": "exposure"}[order]
    view = view.sort_values(key, ascending=False).head(200)

    out = pd.DataFrame({
        "Probability of default": view["p_hist_gbm"].map("{:.3f}".format),
        "Balance at risk": view["exposure"].map("NT${:,.0f}".format),
        "Expected loss": view["Expected loss"].map("NT${:,.0f}".format),
        "Decision": view["Decision"], "Outcome": view["Outcome"],
        "Sex": view["SEX"].map(GROUP_LABELS["SEX"]),
        "Age": view["AGE_BAND"],
    })
    st.dataframe(out, hide_index=True, use_container_width=True, height=460)
    st.caption(f"Showing up to 200 of {len(df):,} held-out accounts.")


def _auc(y, p):
    """Rank-based AUC, so the app needs no scikit-learn at serving time."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ------------------------------------------------------------------ main --

st.title("Credit decisions, priced")
st.caption(
    "30,000 Taiwanese credit-card accounts, 2005. 7,500 held out. The model "
    "is fixed; the cost assumption is yours to set.")

df = load_scores()

with st.sidebar:
    st.header("Cost assumption")
    ratio = st.slider(
        "A missed default costs this many declined good customers", 1.0, 20.0,
        7.5, 0.5)
    threshold = 1.0 / (1.0 + ratio)
    st.metric("Cost-optimal threshold", f"{threshold:.3f}")
    st.caption(
        "Expected value approves while p(default) < 1/(1+R). At R = 1 that is "
        "0.5 — the familiar cutoff is the claim that turning away a paying "
        "customer hurts exactly as much as writing off a balance.")
    st.divider()
    st.caption(
        f"Test split: {len(df):,} accounts, {int(df['default'].sum()):,} "
        f"defaults ({df['default'].mean():.2%}), "
        f"NT${df['exposure'].sum():,.0f} at risk.")

tabs = st.tabs(["The decision", "The ladder", "Review capacity", "Who pays",
                "Accounts"])
with tabs[0]:
    tab_decision(df, ratio, threshold)
with tabs[1]:
    tab_ladder(df, ratio, threshold)
with tabs[2]:
    tab_capacity(df, ratio, threshold)
with tabs[3]:
    tab_fairness(df, ratio, threshold)
with tabs[4]:
    tab_accounts(df, ratio, threshold)
