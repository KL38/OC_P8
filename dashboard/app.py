"""Credit Scoring Monitoring Dashboard.

Three tabs:
- Operational: volume, latency p50/p95, error rate, score distribution
- Drift: embedded Evidently HTML + summary
- Business: GRANTED vs REFUSED, top-driver features

Reads from Supabase (predictions_log) — never touches the test table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from queries import fetch_recent, fetch_summary, fetch_volume_by_hour

DRIFT_REPORT_PATH = Path(__file__).parent / "static" / "drift_report.html"

st.set_page_config(
    page_title="OC P8 Monitoring",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Credit Scoring — Monitoring")
st.caption("Prêt à Dépenser · prod observability + data drift")

with st.sidebar:
    st.header("Filtres")
    days = st.slider("Fenêtre (jours)", min_value=1, max_value=90, value=7)
    st.markdown("---")
    st.markdown(
        "**Sources**\n\n"
        "- Logs : Supabase `predictions_log`\n"
        "- Drift : `static/drift_report.html`\n"
        "- Régénérer le rapport : `uv run python scripts/generate_drift_report.py`"
    )


tab_ops, tab_drift, tab_business = st.tabs(
    ["⚙️ Opérationnel", "🌊 Data Drift", "💼 Business"]
)


# -------------------------------------------------------------------- Ops --
with tab_ops:
    try:
        summary = fetch_summary(days)
    except Exception as exc:
        st.error(f"Impossible de joindre Supabase : {exc}")
        st.stop()

    if not summary["total"]:
        st.warning(f"Aucune prédiction enregistrée sur les {days} derniers jours.")
        st.stop()

    cols = st.columns(5)
    cols[0].metric("Total requêtes", f"{summary['total']:,}")
    cols[1].metric(
        "Erreurs",
        f"{summary['errors']:,}",
        delta=f"{(summary['errors'] / summary['total']) * 100:.1f} %",
        delta_color="inverse",
    )
    cols[2].metric("Latence p50", f"{int(summary['p50'] or 0)} ms")
    cols[3].metric("Latence p95", f"{int(summary['p95'] or 0)} ms")
    cols[4].metric(
        "% REFUSED",
        f"{(summary['refused'] / max(summary['total'], 1)) * 100:.1f} %",
    )

    st.subheader("Volume & latence par heure")
    hourly = fetch_volume_by_hour(days)
    if not hourly.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                px.bar(hourly, x="hour", y="total", title="Requêtes / heure"),
                use_container_width=True,
            )
        with c2:
            fig = px.line(
                hourly.melt(id_vars="hour", value_vars=["p50", "p95"]),
                x="hour",
                y="value",
                color="variable",
                title="Latence (ms)",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution des probabilités")
    recent = fetch_recent(days)
    ok = recent[recent["status_code"] == 200]
    if not ok.empty:
        st.plotly_chart(
            px.histogram(
                ok,
                x="probability_default",
                nbins=40,
                color="decision",
                title="probability_default — split par décision",
            ),
            use_container_width=True,
        )


# ------------------------------------------------------------------ Drift --
with tab_drift:
    st.subheader("Rapport Data Drift (Evidently)")
    if DRIFT_REPORT_PATH.exists():
        st.caption(f"Source : {DRIFT_REPORT_PATH.name}")
        html = DRIFT_REPORT_PATH.read_text(encoding="utf-8")
        st.components.v1.html(html, height=900, scrolling=True)
    else:
        st.info(
            "Aucun rapport Evidently disponible. Génère-le avec :\n\n"
            "`uv run python scripts/generate_drift_report.py --days 30`\n\n"
            "Puis redéploie le Space ou copie le HTML dans `dashboard/static/`."
        )


# --------------------------------------------------------------- Business --
with tab_business:
    recent = fetch_recent(days)
    if recent.empty:
        st.warning("Pas de données pour la période.")
    else:
        ok = recent[recent["status_code"] == 200]
        c1, c2 = st.columns(2)
        with c1:
            decision_counts = ok["decision"].value_counts().reset_index()
            decision_counts.columns = ["decision", "count"]
            st.plotly_chart(
                px.pie(decision_counts, names="decision", values="count", title="Décisions"),
                use_container_width=True,
            )
        with c2:
            known = ok["client_known"].value_counts().rename({True: "Connu", False: "Inconnu"})
            st.plotly_chart(
                px.pie(
                    pd.DataFrame({"type": known.index, "count": known.values}),
                    names="type",
                    values="count",
                    title="Clients connus vs inconnus",
                ),
                use_container_width=True,
            )

        st.subheader("Derniers appels")
        st.dataframe(
            ok[["timestamp", "sk_id_curr", "client_known", "probability_default",
                "decision", "latency_ms", "model_version"]].head(50),
            use_container_width=True,
            hide_index=True,
        )
