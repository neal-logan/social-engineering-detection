"""
End-to-end self-test against a synthetic dataset that mimics the
detection metadata + conversations schema.

Run with `python -m analysis.selftest` (from the parent directory) or
directly with `python selftest.py`. Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Make the parent of `analysis/` importable when run directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import figures, loading, preliminaries, query, roc, schema as S


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

def _mk_conversation_record(
    *, request_id: str, n_turns: int, ctype: str, rng: random.Random
) -> dict:
    """Build one conversation-record envelope as the generation pipeline would."""
    conv = {}
    for i in range(n_turns):
        if i == 0:
            conv[f"turn_{i}"] = {"passage": f"rep greeting {rng.random():.3f}"}
        elif S.speaker_for_turn(i) == S.SPEAKER_CALLER:
            d = {"passage": f"caller turn {i} text"}
            for col in S.CIALDINI_FLAG_COLUMNS:
                # threat conversations have higher cialdini-flag rate
                p = 0.3 if ctype == "threat" else 0.05
                d[col] = rng.random() < p
            conv[f"turn_{i}"] = d
        else:
            d = {"passage": f"rep response {i}"}
            for col in S.POLICY_VIOLATION_TYPES:
                # threats have a small chance of violation per turn,
                # benign almost zero
                p = 0.04 if ctype == "threat" else 0.005
                d[col] = rng.random() < p
            conv[f"turn_{i}"] = d
    return {
        "request_id": request_id,
        "selection": {},
        "replicate_index": 0,
        "flavor": "fake",
        "conversation": conv,
        "usage": {"input_tokens": 100, "output_tokens": 200,
                  "total_tokens": 300},
        "generated_at_utc": "2026-04-26T00:00:00Z",
    }


def _detector_predictions_for(
    record: dict, ctype: str, rng: random.Random
) -> tuple[dict, dict]:
    """Generate per-turn predictions for one conversation, one row of metadata."""
    n_turns = len(record["conversation"])
    detections = {}
    for key in S.DETECTION_KEYS:
        objective, stance = key.split("__", 1)
        eligible = set(S.detector_eligible_turns(n_turns, objective))
        # Per-stance base detection rates
        if ctype == "threat":
            p_fire = {"high_recall": 0.55, "balanced": 0.35,
                      "high_precision": 0.20}[stance]
        else:
            p_fire = {"high_recall": 0.10, "balanced": 0.04,
                      "high_precision": 0.01}[stance]
        preds: list = []
        p_dets: list = []
        p_nds: list = []
        lats: list = []
        ins: list = []
        outs: list = []
        for t in range(n_turns):
            if t in eligible:
                fire = rng.random() < p_fire
                preds.append("Y" if fire else "N")
                p = rng.uniform(0.55, 0.95) if fire else rng.uniform(0.05, 0.45)
                p_dets.append(round(p, 3))
                p_nds.append(round(1 - p, 3))
                lats.append(rng.randint(80, 800))
                ins.append(rng.randint(50, 400))
                outs.append(1)
            else:
                preds.append(None)
                p_dets.append(None)
                p_nds.append(None)
                lats.append(None)
                ins.append(None)
                outs.append(None)
        detections[key] = {
            "prediction": preds,
            "p_detected": p_dets,
            "p_not_detected": p_nds,
            "latency_ms": lats,
            "input_tokens": ins,
            "output_tokens": outs,
        }
    # Build the metadata row dict (per-turn lists JSON-serialised)
    meta_row = {}
    total_lat = 0
    total_in = 0
    total_out = 0
    for key, d in detections.items():
        for field, lst in d.items():
            meta_row[f"{key}__{field}"] = json.dumps(lst)
        total_lat += sum(v or 0 for v in d["latency_ms"])
        total_in += sum(v or 0 for v in d["input_tokens"])
        total_out += sum(v or 0 for v in d["output_tokens"])
    meta_row["detection_total_latency_ms"] = total_lat
    meta_row["detection_total_input_tokens"] = total_in
    meta_row["detection_total_output_tokens"] = total_out
    return detections, meta_row


def _build_synthetic(tmp: Path, n_threat: int = 30, n_benign: int = 30,
                     seed: int = 7) -> tuple[Path, Path, Path, Path]:
    """Write threat + benign metadata.xlsx and conversations.json into tmp."""
    rng = random.Random(seed)

    threat_meta_rows = []
    threat_convs = {"conversations": {}}
    benign_meta_rows = []
    benign_convs = {"conversations": {}}

    for kind, n_rows, meta_rows, conv_store in [
        ("threat", n_threat, threat_meta_rows, threat_convs),
        ("benign", n_benign, benign_meta_rows, benign_convs),
    ]:
        for i in range(n_rows):
            n_turns = rng.choice([10, 20])
            rid = f"{kind}_{i:04d}"
            rec = _mk_conversation_record(
                request_id=rid, n_turns=n_turns, ctype=kind, rng=rng
            )
            conv_store["conversations"][rid] = rec
            _, meta_extra = _detector_predictions_for(rec, kind, rng)
            row = {
                "request_id": rid,
                "replicate_index": 0,
                "prompt_template_key": kind,
                "scenario": rng.choice(["credit_union", "brokerage", "health_ins"]),
                "representative": rng.choice(["by_book", "tired", "helpful", "distracted"]),
                "caller": rng.choice(["a", "b", "c"]),
                "benign_context": rng.choice(["minimal", "moderate", "heavy"]),
                "cialdini_emphasis": rng.choice(list(S.CIALDINI_PRINCIPLES)),
                "turn_count_value": n_turns,
                "flavor": "x",
                "generation_model": "gpt-5.4",
                "generation_temperature": 1.0,
                "generation_status": "success",
                "detection_model": "gpt-5.4-nano",
                "detection_status": S.DETECTION_STATUS_SUCCESS,
                "detection_started_at_utc": "2026-04-26T00:00:00Z",
                "detection_finished_at_utc": "2026-04-26T00:01:00Z",
                # Always-null in success rows; pandas will read it as
                # all-NaN object column. Tests that make_context handles
                # this correctly (DuckDB can't infer a type from all-null).
                "detection_last_error": None,
            }
            row.update(meta_extra)
            meta_rows.append(row)

    # Sprinkle a few non-success rows in threat to test filtering
    threat_meta_rows.append({
        **threat_meta_rows[0], "request_id": "threat_dropme",
        "detection_status": S.DETECTION_STATUS_PARTIAL,
    })

    threat_meta_path = tmp / "threat_metadata.xlsx"
    benign_meta_path = tmp / "benign_metadata.xlsx"
    threat_conv_path = tmp / "threat_conversations.json"
    benign_conv_path = tmp / "benign_conversations.json"

    pd.DataFrame(threat_meta_rows).to_excel(threat_meta_path, index=False)
    pd.DataFrame(benign_meta_rows).to_excel(benign_meta_path, index=False)
    with open(threat_conv_path, "w") as f:
        json.dump(threat_convs, f)
    with open(benign_conv_path, "w") as f:
        json.dump(benign_convs, f)

    return threat_meta_path, threat_conv_path, benign_meta_path, benign_conv_path


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def main() -> int:
    print("Building synthetic dataset…")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        figures.set_output_dir(tmp / "out")

        tmp_files = _build_synthetic(tmp)
        threat_meta_path, threat_conv_path, benign_meta_path, benign_conv_path = tmp_files

        # ---- Loading + validation
        print("Loading threat dataset…")
        threat = loading.load_dataset(threat_meta_path, threat_conv_path)
        assert threat.n_dropped_non_success == 1, (
            f"expected 1 non-success dropped, got {threat.n_dropped_non_success}"
        )
        print(f"  {threat.n_conversations} threat conversations retained")

        print("Loading benign dataset…")
        benign = loading.load_dataset(benign_meta_path, benign_conv_path)
        print(f"  {benign.n_conversations} benign conversations retained")

        # ---- Preliminaries: combined conversation table
        threat_conv = preliminaries.build_conversation_table(threat)
        benign_conv = preliminaries.build_conversation_table(benign)
        combined = pd.concat([threat_conv, benign_conv], ignore_index=True)
        print(f"  conversation table: {len(combined)} rows, "
              f"{len(combined.columns)} cols")

        threat_turns = preliminaries.build_turn_table(threat)
        benign_turns = preliminaries.build_turn_table(benign)
        combined_turns = pd.concat([threat_turns, benign_turns], ignore_index=True)
        print(f"  turn table: {len(combined_turns)} rows, "
              f"{len(combined_turns.columns)} cols")

        # ---- Sanity checks on derived columns
        # running_total_* lists must be length n_turns and monotone non-decreasing
        for _, row in combined.iterrows():
            for c in S.CIALDINI_FLAG_COLUMNS + S.POLICY_VIOLATION_TYPES:
                lst = row[f"running_total_{c}"]
                assert len(lst) == row["n_turns"], f"len mismatch {c}"
                assert all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1)), \
                    f"{c} not monotone for {row['request_id']}"

        # first_prediction_turn must be NaN iff no turn fires
        for key in S.DETECTION_KEYS:
            col = f"first_prediction_turn__{key}"
            for _, row in combined.iterrows():
                preds_list = row[f"{key}__prediction"]
                fired = any(p == 1 for p in preds_list if p is not None)
                if fired:
                    assert not np.isnan(row[col])
                else:
                    assert np.isnan(row[col])

        # ---- Ibis context + queries
        print("Building ibis context…")
        ctx = query.make_context(combined, combined_turns)

        print("Latency percentiles…")
        lat_tbl = query.latency_percentiles_table(ctx)
        figures.save_table("latency_percentiles", lat_tbl, index=False)
        assert {"detection_key", "conversation_type"}.issubset(lat_tbl.columns)
        print(lat_tbl.head(3).to_string(index=False))

        print("Confusion matrices…")
        cm1 = query.confusion_threat_vs_se_ever_predicted(ctx)
        cm2 = query.confusion_violation_vs_pv_ever_predicted(ctx)
        cm3 = query.confusion_turn_violation_vs_pv_prediction(ctx)
        figures.save_table("confusion_threat_vs_se_ever", cm1)
        figures.save_table("confusion_violation_vs_pv_ever", cm2)
        figures.save_table("confusion_turn_violation_vs_pv", cm3)

        print("Coverage…")
        cov = query.detection_coverage_summary(ctx)
        figures.save_table("detection_coverage", cov, index=False)
        # Since we restricted to success, coverage must be 1.0
        assert (cov["coverage"].dropna() == 1.0).all()

        print("Aggregation-rule recall…")
        agg = query.conversation_recall_under_aggregation_rules(ctx)
        figures.save_table("conv_recall_aggregation_rules", agg, index=False)
        print(agg.to_string(index=False))

        # ---- Figures
        print("Histograms…")
        figures.hist_first_violation_turn_by_type(combined, conversation_filter="threat")
        figures.hist_first_violation_turn_by_type(combined, conversation_filter="benign")
        figures.hist_first_violation_turn_by_type(combined)

        figures.hist_first_prediction_turn_by_stance(
            combined, objective="social_engineering", conversation_filter="threat")
        figures.hist_first_prediction_turn_by_stance(
            combined, objective="social_engineering", conversation_filter="benign")
        figures.hist_first_prediction_turn_by_stance(
            combined, objective="policy_violation", conversation_filter="threat")
        figures.hist_first_prediction_turn_by_stance(
            combined, objective="policy_violation", conversation_filter="benign")

        figures.hist_pred_minus_violation_diff(
            combined, objective="social_engineering", conversation_filter="threat")
        figures.hist_pred_minus_violation_diff(
            combined, objective="policy_violation", conversation_filter="threat")

        figures.hist_violations_pre_at_post(
            combined, objective="social_engineering", conversation_filter="threat")
        figures.hist_violations_pre_at_post(
            combined, objective="policy_violation", conversation_filter="threat")

        # New: violations by type × representative type
        figures.hist_violations_by_type_x_representative(
            combined, conversation_filter="threat")
        figures.hist_violations_by_type_x_representative(
            combined, conversation_filter="benign")

        # Heatmap: prompted Cialdini emphasis x Cialdini signals actually present
        threat_only = combined[combined["conversation_type"] == "threat"]
        emphasis_label = (
            threat_only["cialdini_emphasis"]
            .apply(S.cialdini_principle_label)
        )
        actual = threat_only.groupby(emphasis_label)[
            [f"total_cialdini_{p}" for p in S.CIALDINI_PRINCIPLES]
        ].mean()
        actual.columns = [c.replace("total_cialdini_", "") for c in actual.columns]
        actual.index.name = "prompted_emphasis"
        figures.heatmap_from_dataframe(
            actual,
            title="Mean signals by principle (rows = prompted emphasis), threat only",
            name="heatmap_prompted_x_actual_cialdini",
            fmt="{:.2f}",
        )

        # Confusion matrices via the scenario-aligned helper
        for scen in roc.SCENARIOS:
            cm = query.confusion_for_scenario(ctx, scen, stance="balanced")
            figures.save_table(f"confusion_for_scenario__{scen}__balanced", cm)
        # Heatmap example: cialdini emphasis x violation type, threat conversations
        threat_only = combined[combined["conversation_type"] == "threat"]
        heat_df = (
            threat_only.groupby("cialdini_emphasis")[
                [f"total_{v}" for v in S.POLICY_VIOLATION_TYPES]
            ].mean()
        )
        heat_df.columns = [c.replace("total_", "") for c in heat_df.columns]
        figures.heatmap_from_dataframe(
            heat_df,
            title="Mean violations per conversation, threat only",
            name="heatmap_emphasis_x_violation",
        )

        # Sankey
        try:
            figures.sankey_threat_to_outcomes(combined)
        except RuntimeError as e:
            print(f"  (skipping sankey: {e})")

        # ---- Combined first-prediction column sanity
        for stance in S.STANCES:
            col_combined = f"first_prediction_turn__combined__{stance}"
            assert col_combined in combined.columns, (
                f"missing combined column {col_combined}"
            )
            # Combined turn must equal min(SE, PV) where defined
            for _, row in combined.iterrows():
                se = row[f"first_prediction_turn__social_engineering__{stance}"]
                pv = row[f"first_prediction_turn__policy_violation__{stance}"]
                comb = row[col_combined]
                cands = [v for v in (se, pv) if not np.isnan(v)]
                if not cands:
                    assert np.isnan(comb), (
                        f"combined should be NaN: rid={row['request_id']} "
                        f"stance={stance}"
                    )
                else:
                    assert comb == min(cands), (
                        f"combined != min: rid={row['request_id']} stance={stance} "
                        f"se={se} pv={pv} combined={comb}"
                    )

        # ---- ROC end-to-end (4 scenarios x 1 figure each)
        print("ROC scenarios…")
        for scen in roc.SCENARIOS:
            r_overall = roc.compute_roc(combined, combined_turns, scen,
                                        slice_label="overall")
            assert 0.0 <= r_overall.auc <= 1.0, (
                f"AUC out of [0,1]: {scen}={r_overall.auc}"
            )
            print(f"  {scen:14s}  overall AUC = {r_overall.auc:.3f}  "
                  f"n+={r_overall.n_pos}  n-={r_overall.n_neg}")

            # By Cialdini and benign context
            r_by_cialdini = roc.compute_roc_by_slice(
                combined, combined_turns, scen, "cialdini_emphasis"
            )
            r_by_context = roc.compute_roc_by_slice(
                combined, combined_turns, scen, "benign_context"
            )

            figures.plot_roc_curves(
                [r_overall], title=f"{scen} ROC (overall)",
                name=f"roc__{scen}__overall",
            )
            figures.plot_roc_curves(
                r_by_cialdini, title=f"{scen} ROC by Cialdini emphasis",
                name=f"roc__{scen}__by_cialdini",
            )
            figures.plot_roc_curves(
                r_by_context, title=f"{scen} ROC by benign context",
                name=f"roc__{scen}__by_benign_context",
            )

            # Heatmap: benign_context x cialdini_emphasis -> AUC
            heat = roc.auc_heatmap_table(
                combined, combined_turns, scen,
                row_column="benign_context",
                col_column="cialdini_emphasis",
            )
            figures.heatmap_from_dataframe(
                heat,
                title=f"{scen} AUC: benign_context × cialdini_emphasis",
                name=f"heatmap_auc__{scen}",
            )

        # Operating-points + AUC-summary CSVs
        all_overall = [
            roc.compute_roc(combined, combined_turns, s, slice_label="overall")
            for s in roc.SCENARIOS
        ]
        figures.save_table(
            "roc_operating_points_overall",
            roc.operating_points_to_dataframe(all_overall), index=False,
        )
        figures.save_table(
            "roc_auc_summary_overall",
            roc.auc_summary_dataframe(all_overall), index=False,
        )

        # ---- File listing
        out = list((tmp / "out").glob("*"))
        print(f"\nWrote {len(out)} artifacts to {tmp/'out'}:")
        for p in sorted(out)[:25]:
            print(f"   {p.name}")
        if len(out) > 25:
            print(f"   …and {len(out) - 25} more")

        print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
