"""Scoring primitives and the per-drug resistance validation, shared across eval files."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SRC"))

from feature_engineering import DATA, drug_map, flat
from rule_engine import RuleEngine


# SCORING


def safe_ratio(num, den):
    return round(num / den, 3) if den else 0.0


def binary_rates(actual, predicted):
    """Sensitivity, specificity, and precision from boolean actual and predicted arrays."""
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    return {'sensitivity': safe_ratio(tp, tp + fn), 'specificity': safe_ratio(tn, tn + fp),
            'precision': safe_ratio(tp, tp + fp), 'n': tp + fn}


def class_rates(truth, prediction, labels):
    """Per-class sensitivity, specificity, and precision for a multiclass label set."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    return {label: binary_rates(truth == label, prediction == label) for label in labels}


def balanced_accuracy(rates):
    """Mean per-class sensitivity over classes that appear in the truth."""
    present = [r['sensitivity'] for r in rates.values() if r['n']]
    return round(sum(present) / len(present), 3) if present else 0.0


def macro_f1(rates):
    """Unweighted mean F1 over classes that appear in the truth."""
    scores = []
    for r in rates.values():
        if not r['n']:
            continue
        denom = r['sensitivity'] + r['precision']
        scores.append(2 * r['sensitivity'] * r['precision'] / denom if denom else 0.0)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def mcnemar(b, c):
    """Continuity-corrected McNemar test for two paired classifiers."""
    n = b + c
    if not n:
        return {'chi2': 0.0, 'p_value': 1.0, 'discordant': 0}
    from scipy.stats import chi2
    stat = (abs(b - c) - 1) ** 2 / n
    return {'chi2': round(stat, 2), 'p_value': float(chi2.sf(stat, 1)), 'discordant': n}


def brier(predictions):
    """Mean squared error of the predicted probability against the outcome."""
    if not predictions:
        return 0.0
    probs = np.array([p for p, _ in predictions], dtype=float)
    labels = np.array([1.0 if y else 0.0 for _, y in predictions])
    return round(float(np.mean((probs - labels) ** 2)), 4)


# PER-DRUG VALIDATION

def dst_truth(drugs):
    """Per-isolate measured R/S call per drug, NaN where the drug was not tested."""
    df = flat(pd.read_parquet(DATA / "DST_MEASUREMENTS.parquet",
                              columns=["UNIQUEID", "DRUG", "PHENOTYPE"]), "UNIQUEID")
    df = df[df["PHENOTYPE"].astype(str).isin(["R", "S"])].copy()
    df["drug"] = df["DRUG"].astype(str).map(drugs)
    df = df.dropna(subset=["drug"])
    df["call"] = (df["PHENOTYPE"].astype(str) == "R").astype(float)
    return df.pivot_table(index="UNIQUEID", columns="drug", values="call", aggfunc="max")


def catalog_truth(drugs):
    """Per-isolate genotypic resistance call per drug from the catalog effects."""
    eff = flat(pd.read_parquet(DATA / "EFFECTS.parquet",
                               columns=["UNIQUEID", "DRUG", "PREDICTION"]), "UNIQUEID")
    r = eff[eff["PREDICTION"].astype(str) == "R"].copy()
    r["drug"] = r["DRUG"].astype(str).map(drugs)
    r = r.dropna(subset=["drug"])
    r["call"] = 1.0
    return r.pivot_table(index="UNIQUEID", columns="drug", values="call", aggfunc="max")


def exclusion_set(engine, isolate):
    """Drugs the rule engine flags for one isolate, direct and class cross-resistance."""
    recs = engine.evaluate_strain(isolate)["recommendations"]
    return {e["drug"] for e in recs["exclusions"] if e["drug"]}


def engine_call_sets(drugs, isolates):
    """Per-isolate set of drugs the rule engine flags as resistant."""
    from validation import IsolateOntology, RuleEngineEvaluator
    evaluator = RuleEngineEvaluator(DATA / "EFFECTS.parquet", drugs)
    by_isolate = evaluator.mutations(isolates)
    engine = RuleEngine(IsolateOntology(by_isolate))
    engine.build_rules()
    return {i: exclusion_set(engine, i) for i in by_isolate}


def calls_frame(call_sets, drugs):
    """Boolean per-drug call table from a mapping of isolate to flagged-drug set."""
    index = list(call_sets)
    data = {d: np.fromiter((d in call_sets[i] for i in index), dtype=float, count=len(index))
            for d in drugs}
    return pd.DataFrame(data, index=index)


def drug_scores(truth, call, drugs):
    """Sensitivity, specificity, and precision per drug over isolates tested for it."""
    aligned = call.reindex(index=truth.index, columns=drugs).fillna(0.0)
    scores = {}
    for drug in drugs:
        measured = truth[drug].notna()
        actual = (truth.loc[measured, drug] == 1.0).to_numpy()
        predicted = (aligned.loc[measured, drug] == 1.0).to_numpy()
        scores[drug] = binary_rates(actual, predicted)
    return scores


def per_drug_scores():
    """Per-drug sensitivity and specificity for the engine and catalog against DST."""
    drugs = drug_map(DATA / "DRUG_CODES.csv")
    dst = dst_truth(drugs)
    catalog = catalog_truth(drugs)
    targets = sorted(set(dst.columns) & set(catalog.columns))
    engine = calls_frame(engine_call_sets(drugs, list(dst.index)), targets)
    return {
        "eval_isolates": int(dst.index.size),
        "drugs": targets,
        "rule_engine": drug_scores(dst, engine, targets),
        "who_catalog": drug_scores(dst, catalog, targets),
    }


def print_per_drug(summary):
    print(f"\nPer-Drug Classification ({summary['eval_isolates']:,} isolates)")
    for name in ("rule_engine", "who_catalog"):
        print(f"\n{name}")
        for drug in summary["drugs"]:
            r = summary[name][drug]
            print(f"  {drug:14s}: sens {r['sensitivity']:.1%}  spec {r['specificity']:.1%}  "
                  f"ppv {r['precision']:.1%}  (R={r['n']})")


def main():
    summary = per_drug_scores()
    with open("per_drug_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print_per_drug(summary)
    print("\nResults saved to per_drug_results.json")


if __name__ == "__main__":
    main()