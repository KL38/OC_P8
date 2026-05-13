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
    hours = st.slider(
        "Fenêtre (heures)",
        min_value=1,
        max_value=168,
        value=24,
        help="Plage temporelle pour toutes les métriques. 24h = 1 jour, 168h = 7 jours.",
    )
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

# Fetched once and reused across the Operational and Business tabs. The
# @st.cache_data decorator on fetch_recent already deduplicates the DB
# round-trip, but computing the boolean mask twice would still cost two
# DataFrame allocations.
try:
    _recent_df = fetch_recent(hours)
    _ok_df = _recent_df[_recent_df["status_code"] == 200]
except Exception:
    # If Supabase is unreachable, the tab_ops error path below already shows
    # the message; just keep these empty so downstream blocks degrade gracefully.
    _recent_df = pd.DataFrame()
    _ok_df = pd.DataFrame()


# -------------------------------------------------------------------- Ops --
with tab_ops:
    try:
        summary = fetch_summary(hours)
    except Exception as exc:
        st.error(f"Impossible de joindre Supabase : {exc}")
        st.stop()

    if not summary["total"]:
        st.warning(f"Aucune prédiction enregistrée sur les {hours} dernières heures.")
        st.stop()

    # Headline: total server-side wall-clock = handler + DB log. The detail
    # decomposition lives in the dedicated section below.
    _total_p50_top = int(round(float(summary["p50"] or 0))) + int(
        round(float(summary["db_log_p50"] or 0))
    )
    _total_p95_top = int(round(float(summary["p95"] or 0))) + int(
        round(float(summary["db_log_p95"] or 0))
    )

    cols = st.columns(6)
    cols[0].metric("Total requêtes", f"{summary['total']:,}")
    cols[1].metric(
        "Erreurs",
        f"{summary['errors']:,}",
        delta=f"{(summary['errors'] / summary['total']) * 100:.1f} %",
        delta_color="inverse",
    )
    cols[2].metric(
        "Total p50",
        f"{_total_p50_top} ms",
        help="Wall-clock serveur complet = handler (`latency_ms`) + DB log (`db_log_ms`). Détail dans la section *Décomposition* plus bas.",
    )
    cols[3].metric(
        "Total p95",
        f"{_total_p95_top} ms",
        help="Wall-clock serveur p95 = handler p95 + DB log p95.",
    )
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
    hourly = fetch_volume_by_hour(hours)
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

    def _ms(v) -> int:
        """Format helper — round to int ms, default 0 when SQL returns NULL."""
        return 0 if v is None else int(round(float(v)))

    handler_p50 = _ms(summary["p50"])
    handler_p95 = _ms(summary["p95"])
    asm_p50 = _ms(summary["asm_p50"])
    asm_p95 = _ms(summary["asm_p95"])
    inf_p50 = _ms(summary["inf_p50"])
    inf_p95 = _ms(summary["inf_p95"])
    inf_cpu_p50 = _ms(summary["inf_cpu_p50"])
    inf_cpu_p95 = _ms(summary["inf_cpu_p95"])
    db_log_p50 = _ms(summary["db_log_p50"])
    db_log_p95 = _ms(summary["db_log_p95"])
    plumb_p50 = _ms(summary["plumbing_p50"])
    plumb_p95 = _ms(summary["plumbing_p95"])
    total_p50 = handler_p50 + db_log_p50
    total_p95 = handler_p95 + db_log_p95

    st.caption(
        f"**Latence client perçue ≈ handler ({handler_p50} ms p50).** "
        f"Le **DB log** ({db_log_p50} ms p50) s'exécute en `BackgroundTask` "
        "après l'envoi de la réponse — il n'impacte plus le client (étape 4).  \n"
        "Le **handler** (`latency_ms`) couvre l'assembly + l'inférence + la construction "
        "de la réponse. Le **DB log** (`db_log_ms`) est mesuré séparément dans `api/logger.py` "
        "autour de l'INSERT Supabase, et reste affiché comme métrique de santé serveur. "
        "Le **plumbing Δ** = `latency_ms - assembly - inference` isole le résidu Python "
        "entre les sous-mesures (inits de variables, return statement, entrée dans le "
        "`finally`) — typiquement < 1 ms."
    )

    cols_perf = st.columns(7)
    cols_perf[0].metric(
        "Total p50 / p95",
        f"{total_p50} / {total_p95} ms",
        help="Wall-clock serveur complet = `latency_ms` (handler) + `db_log_ms` (INSERT). C'est le temps réel passé côté serveur sur une requête.",
    )
    cols_perf[1].metric(
        "Handler p50 / p95",
        f"{handler_p50} / {handler_p95} ms",
        help="`latency_ms` = assembly + inference + plumbing. **N'inclut pas** le DB log.",
    )
    cols_perf[2].metric(
        "Feature assembly p50 / p95",
        f"{asm_p50} / {asm_p95} ms",
        help="Lookup feature store + transforms + ratios + reindex.",
    )
    cols_perf[3].metric(
        "Inference wall p50 / p95",
        f"{inf_p50} / {inf_p95} ms",
        help="`model.predict_proba` (wall-clock).",
    )
    cols_perf[4].metric(
        "Inference CPU p50 / p95",
        f"{inf_cpu_p50} / {inf_cpu_p95} ms",
        help="CPU time consommé pendant l'inférence (peut lire 0 sur paths très rapides — résolution de `time.process_time`).",
    )
    cols_perf[5].metric(
        "DB log p50 / p95",
        f"{db_log_p50} / {db_log_p95} ms",
        help="INSERT Supabase mesuré autour de `conn.execute(insert(...))` dans `api/logger.py`. Domine généralement l'overhead total.",
    )
    cols_perf[6].metric(
        "Plumbing Δ p50 / p95",
        f"{plumb_p50} / {plumb_p95} ms",
        help="`latency_ms - feature_assembly_ms - inference_ms`. Résidu Python entre les sous-mesures (typiquement < 1 ms).",
    )

    breakdown = fetch_latency_breakdown(hours)
    if not breakdown.empty:
        breakdown = breakdown.copy()
        # Plumbing per hour = handler - assembly - inference, clamped at 0 to
        # absorb sub-ms rounding artefacts. We then stack 4 components whose
        # total equals handler + db_log = full server wall-clock.
        breakdown["plumbing_p50"] = (
            breakdown["total_p50"].fillna(0)
            - breakdown["feature_assembly_p50"].fillna(0)
            - breakdown["inference_p50"].fillna(0)
        ).clip(lower=0)
        long_df = breakdown.melt(
            id_vars="hour",
            value_vars=[
                "feature_assembly_p50",
                "inference_p50",
                "plumbing_p50",
                "db_log_p50",
            ],
            var_name="composant",
            value_name="ms",
        )
        long_df["composant"] = long_df["composant"].map({
            "feature_assembly_p50": "Feature assembly",
            "inference_p50": "Model inference",
            "plumbing_p50": "Plumbing Python (résidu)",
            "db_log_p50": "DB log (INSERT Supabase)",
        })
        fig_breakdown = px.area(
            long_df,
            x="hour",
            y="ms",
            color="composant",
            title="Décomposition p50 par heure (stacked = wall-clock serveur)",
        )
        fig_breakdown.update_layout(yaxis_title="latence p50 (ms)")
        st.plotly_chart(fig_breakdown, use_container_width=True)
    else:
        st.info(
            "Pas encore de données instrumentées sur la fenêtre. "
            "Lance du trafic via `scripts/seed_traffic.py` après le deploy de l'API étape 4."
        )

    st.subheader("Distribution des probabilités")
    if not _ok_df.empty:
        st.plotly_chart(
            px.histogram(
                _ok_df,
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
    if _recent_df.empty:
        st.warning("Pas de données pour la période.")
    else:
        ok = _ok_df
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
