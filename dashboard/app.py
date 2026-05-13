"""Credit Scoring Monitoring Dashboard.

Four tabs:
- Operational: volume, latency p50/p95, error rate, score distribution
- Drift: embedded Evidently HTML + summary
- Business: GRANTED vs REFUSED, top-driver features
- Advanced: output drift, critical features, weighted drift score

Reads from Supabase (predictions_log) — never touches the test table.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

from queries import (
    fetch_latency_breakdown,
    fetch_proba_distribution,
    fetch_recent,
    fetch_summary,
    fetch_volume_by_hour,
    load_drift_report_json,
    load_feature_importance,
    load_proba_reference,
    parse_drift_results,
)

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


tab_ops, tab_drift, tab_business, tab_advanced = st.tabs(
    ["⚙️ Opérationnel", "🌊 Data Drift", "💼 Business", "🧠 Indicateurs avancés"]
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

    cols = st.columns(6)
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
    cols[5].metric(
        "% Nouveaux clients",
        f"{(summary['unknowns'] / max(summary['total'], 1)) * 100:.1f} %",
        help="Part de clients sans entrée dans le feature store (no_history_template).",
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

    st.subheader("Décomposition de la latence")
    st.caption(
        "Le `latency_ms` total est mesuré en bout-en-bout (handler FastAPI). "
        "`feature_assembly_ms` couvre lookup feature store + transforms + ratios + reindex. "
        "`inference_ms` est le wall-clock de `model.predict_proba`. "
        "`inference_cpu_ms` est le CPU time consommé pendant l'inférence (peut être 0 sur paths très rapides). "
        "Le delta `latency_ms - assembly - inference` mesure l'overhead restant (DB log, sérialisation, parsing)."
    )
    cols_perf = st.columns(4)
    cols_perf[0].metric(
        "Total p50 / p95",
        f"{int(summary['p50'] or 0)} / {int(summary['p95'] or 0)} ms",
    )
    cols_perf[1].metric(
        "Feature assembly p50 / p95",
        f"{(summary['asm_p50'] or 0):.1f} / {(summary['asm_p95'] or 0):.1f} ms",
    )
    cols_perf[2].metric(
        "Inference wall p50 / p95",
        f"{(summary['inf_p50'] or 0):.2f} / {(summary['inf_p95'] or 0):.2f} ms",
    )
    cols_perf[3].metric(
        "Inference CPU p50 / p95",
        f"{(summary['inf_cpu_p50'] or 0):.2f} / {(summary['inf_cpu_p95'] or 0):.2f} ms",
    )

    breakdown = fetch_latency_breakdown(days)
    if not breakdown.empty:
        breakdown = breakdown.copy()
        breakdown["overhead_p50"] = (
            breakdown["total_p50"].fillna(0)
            - breakdown["feature_assembly_p50"].fillna(0)
            - breakdown["inference_p50"].fillna(0)
        ).clip(lower=0)
        long_df = breakdown.melt(
            id_vars="hour",
            value_vars=["feature_assembly_p50", "inference_p50", "overhead_p50"],
            var_name="composant",
            value_name="ms",
        )
        long_df["composant"] = long_df["composant"].map({
            "feature_assembly_p50": "Feature assembly",
            "inference_p50": "Model inference",
            "overhead_p50": "Overhead (DB log + sérialisation)",
        })
        fig_breakdown = px.area(
            long_df,
            x="hour",
            y="ms",
            color="composant",
            title="Décomposition p50 par heure (stacked)",
        )
        fig_breakdown.update_layout(yaxis_title="latence p50 (ms)")
        st.plotly_chart(fig_breakdown, use_container_width=True)
    else:
        st.info(
            "Pas encore de données instrumentées sur la fenêtre. "
            "Lance du trafic via `scripts/seed_traffic.py` après le deploy de l'API étape 4."
        )

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


# --------------------------------------------------------- Advanced KPIs --
with tab_advanced:
    st.caption(
        "Indicateurs avancés au-delà du drift par feature : drift de la sortie "
        "modèle, suivi des features critiques, et score de drift pondéré par "
        "importance SHAP."
    )

    proba_ref = load_proba_reference()
    importance = load_feature_importance()
    drift_json = load_drift_report_json()
    drift_results = parse_drift_results(drift_json)

    # ---------------------------------------------------- Output drift --
    st.subheader("1. Output drift — distribution de probability_default")
    if proba_ref is None:
        st.info(
            "`dashboard/static/proba_reference.json` introuvable. "
            "Génère-le avec `uv run python scripts/build_monitoring_artefacts.py`."
        )
    else:
        try:
            current_proba = fetch_proba_distribution(limit=500)
        except Exception as exc:
            st.error(f"Impossible de récupérer les probas prod : {exc}")
            current_proba = []

        if not current_proba:
            st.warning("Pas de prédiction logguée pour calculer la distribution prod.")
        else:
            ref_values = np.array(proba_ref.get("values", []))
            cur_values = np.array(current_proba)

            # K-S test on raw samples — robust comparison of distributions.
            # scipy returns a KstestResult NamedTuple (statistic, pvalue); the
            # type stubs are weak, hence the ignore comment.
            ks_result = scipy_stats.ks_2samp(ref_values, cur_values)
            ks_p = float(ks_result.pvalue)  # type: ignore[attr-defined]
            detected = ks_p < 0.05

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Reference mean", f"{ref_values.mean():.3f}")
            c2.metric(
                "Current mean",
                f"{cur_values.mean():.3f}",
                delta=f"{(cur_values.mean() - ref_values.mean()):+.3f}",
            )
            c3.metric("K-S p-value", f"{ks_p:.2e}")
            c4.metric(
                "Output drift",
                "✓ détecté" if detected else "✗ stable",
                delta_color="inverse" if detected else "normal",
            )

            # Overlay histogram.
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=ref_values, name="Reference (training)",
                    opacity=0.55, nbinsx=40, histnorm="probability",
                    marker_color="#888",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=cur_values, name=f"Current (last {len(cur_values)})",
                    opacity=0.7, nbinsx=40, histnorm="probability",
                    marker_color="#e74c3c",
                )
            )
            fig.update_layout(
                barmode="overlay",
                xaxis_title="probability_default",
                yaxis_title="density",
                title="Distribution de la proba de défaut — reference vs current",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Le K-S test compare les deux échantillons sur leur forme de "
                "distribution. Un drift de la sortie modèle est l'indicateur le "
                "plus direct d'un comportement modèle altéré en prod — il "
                "agrège l'effet de tous les drifts d'inputs simultanément."
            )

    # ------------------------------------------------- Critical features --
    st.subheader("2. Features critiques (top 10 SHAP)")
    if importance is None:
        st.info(
            "`dashboard/static/feature_importance.json` introuvable. "
            "Génère-le avec `uv run python scripts/build_monitoring_artefacts.py`."
        )
    elif not drift_results:
        st.info(
            "`dashboard/static/drift_report.json` introuvable. "
            "Régénère le drift report avec `uv run python scripts/generate_drift_report.py`."
        )
    else:
        top_n = 10
        rows = []
        for entry in importance["top"][:top_n]:
            feat = entry["feature"]
            imp = entry["importance"]
            result = drift_results.get(feat, {})
            detected = result.get("detected")
            score = result.get("score")
            stattest = result.get("stattest") or "—"
            rows.append({
                "Rank": entry["rank"],
                "Feature": feat,
                "SHAP importance": round(imp, 4),
                "Drift": "🔴 Détecté" if detected else ("🟢 Stable" if detected is False else "—"),
                "Drift score": (f"{score:.4f}" if score is not None else "—"),
                "Stat test": stattest,
            })
        df_critical = pd.DataFrame(rows)

        n_drifted = sum(1 for r in rows if "Détecté" in r["Drift"])
        c1, c2 = st.columns([1, 3])
        c1.metric(
            f"Drifted parmi top {top_n}",
            f"{n_drifted}/{top_n}",
            delta_color="inverse",
        )
        c2.caption(
            f"Méthode : {importance['method']} sur {importance['sample_size']} "
            "lignes de reference. Le nombre de features critiques qui ont drifté "
            "est l'indicateur le plus actionnable — un drift sur un top-feature "
            "demande un retraining prioritaire."
        )

        st.dataframe(df_critical, use_container_width=True, hide_index=True)

    # -------------------------------------------------- Weighted drift --
    st.subheader("3. Score de drift pondéré par importance")
    if importance is None or not drift_results:
        st.info(
            "Indicateur indisponible tant que `feature_importance.json` et "
            "`drift_report.json` ne sont pas tous les deux présents."
        )
    else:
        total_importance = 0.0
        drifted_importance = 0.0
        n_features_seen = 0
        for entry in importance["top"]:
            feat = entry["feature"]
            imp = float(entry["importance"])
            total_importance += imp
            result = drift_results.get(feat)
            if result is None:
                continue
            n_features_seen += 1
            if result.get("detected"):
                drifted_importance += imp

        weighted_ratio = (drifted_importance / total_importance) if total_importance > 0 else 0.0
        threshold = 0.30

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Drift pondéré",
            f"{weighted_ratio:.1%}",
            delta=f"seuil {threshold:.0%}",
            delta_color="inverse" if weighted_ratio >= threshold else "normal",
        )
        c2.metric(
            "Importance couverte",
            f"{n_features_seen} / {len(importance['top'])} features",
        )
        c3.metric(
            "Verdict",
            "🔴 Alerte" if weighted_ratio >= threshold else "🟢 OK",
        )

        st.caption(
            "**Formule** : Σ(importance × drift_detected) / Σ(importance) sur les "
            f"top-{len(importance['top'])} features SHAP. Pondère le verdict "
            "binaire d'Evidently par l'impact réel de chaque feature sur le "
            "modèle. Seuil : 30% de l'importance totale qui drift → alerte. "
            "Indicateur plus fin que le ratio brut affiché par Evidently dans "
            "l'onglet Data Drift."
        )
