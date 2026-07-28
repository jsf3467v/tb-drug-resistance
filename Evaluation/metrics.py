"""Scoring primitives and the per-drug resistance validation, shared across eval files."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "SRC"))

from feature_engineering import DATA, drug_map, flat
from rule_engine import RuleEngine

RESULTS = EVAL_DIR / "per_drug_results.json"


def safe_ratio(num, den):
    """Ratio, or zero when nothing was counted. Deliberately unrounded, because
    balanced accuracy and macro F1 average these and rounding here moved their
    last reported digit on roughly one run in seven."""
    return num / den if den else 0.0


def binary_rates(actual, predicted):
    """Sensitivity, specificity, precision, and their harmonic mean from boolean
    actual and predicted arrays. F1 travels with the pair it combines, so the
    per-drug and per-tier arms report it from one definition rather than two."""
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())
    tn = int((~actual & ~predicted).sum())
    sensitivity = safe_ratio(tp, tp + fn)
    precision = safe_ratio(tp, tp + fp)
    return {'sensitivity': sensitivity, 'specificity': safe_ratio(tn, tn + fp),
            'precision': precision,
            'f1': safe_ratio(2 * sensitivity * precision, sensitivity + precision),
            'n': tp + fn, 'evaluated': tp + fp + fn + tn}


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
    scores = [r['f1'] for r in rates.values() if r['n']]
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def wilson_interval(hits, n, confidence=0.95):
    """Score interval for a proportion, which holds where the normal
    approximation would run past zero or one."""
    if not n:
        return 0.0, 0.0
    z = float(norm.ppf(0.5 + confidence / 2))
    p, shift = hits / n, z * z / n
    center = (p + shift / 2) / (1 + shift)
    spread = z * math.sqrt(p * (1 - p) / n + shift / (4 * n)) / (1 + shift)
    return round(center - spread, 3), round(center + spread, 3)


def mcnemar(b, c):
    """Continuity-corrected McNemar test for two paired classifiers. The
    correction is floored at zero, since without it two classifiers that
    disagree equally often return a positive statistic instead of none."""
    n = b + c
    if not n:
        return {'chi2': 0.0, 'p_value': 1.0, 'discordant': 0}
    stat = max(0.0, abs(b - c) - 1.0) ** 2 / n
    return {'chi2': round(stat, 2), 'p_value': float(chi2.sf(stat, 1)), 'discordant': n}


def prediction_arrays(predictions):
    """Probability and outcome arrays from (probability, outcome) pairs."""
    n = len(predictions)
    probs = np.fromiter((p for p, _ in predictions), dtype=float, count=n)
    labels = np.fromiter((bool(y) for _, y in predictions), dtype=bool, count=n)
    return probs, labels


def brier(predictions):
    """Mean squared error of the predicted probability against the outcome."""
    if not predictions:
        return 0.0
    probs, labels = prediction_arrays(predictions)
    return round(float(np.mean((probs - labels) ** 2)), 4)


def brier_constant(predictions):
    """Brier score of a constant prediction at the observed base rate, which is
    p(1-p). The floor a probability has to beat to carry any information."""
    if not predictions:
        return 0.0
    _, labels = prediction_arrays(predictions)
    base = float(labels.mean())
    return round(base * (1.0 - base), 4)


def auc(predictions):
    """Area under the ROC curve, by rank sum so ties are handled. Zero when
    either class is absent, since ranking is undefined against nothing."""
    if not predictions:
        return 0.0
    probs, labels = prediction_arrays(predictions)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if not positives or not negatives:
        return 0.0
    ranks = rankdata(probs)
    hits = ranks[labels].sum() - positives * (positives + 1) / 2
    return round(float(hits / (positives * negatives)), 4)


def dst_truth(drugs):
    """Per-isolate measured R/S call per drug, NaN where the drug was not tested.
    Replicates aggregate by max, so an isolate called both R and S for one drug
    counts as R."""
    df = flat(pd.read_parquet(DATA / "DST_MEASUREMENTS.parquet",
                              columns=["UNIQUEID", "DRUG", "PHENOTYPE"]), "UNIQUEID")
    df = df[df["PHENOTYPE"].astype(str).isin(["R", "S"])].copy()
    df["drug"] = df["DRUG"].astype(str).map(drugs)
    df = df.dropna(subset=["drug"])
    df["call"] = (df["PHENOTYPE"].astype(str) == "R").astype(float)
    return df.pivot_table(index="UNIQUEID", columns="drug", values="call", aggfunc="max")


def catalog_calls(graded):
    """Per-isolate genotypic resistance call per drug, from rows already graded
    and held by the evaluator, so EFFECTS is read once for both arms."""
    return graded[["UNIQUEID", "drug"]].assign(call=1.0).pivot_table(
        index="UNIQUEID", columns="drug", values="call", aggfunc="max")


class IsolateOntology:
    """Feeds per-isolate mutations to the rule engine without a database."""

    def __init__(self, mutations):
        self.mutations = mutations

    def patient_strain_mapping(self, strain_id):
        """The engine reads an id beginning with P as a patient. These are
        isolates, so the id maps to itself and the branch changes nothing."""
        return [{'strain': strain_id}]

    def strain_mutations_detailed(self, strain_id):
        return self.mutations.get(strain_id, [])


class RuleEngineEvaluator:
    name = 'rule_engine'

    def __init__(self, effects_path, drugs):
        self.effects_path = effects_path
        self.drugs = drugs
        self.resistant = None

    def predictions(self, isolates):
        by_isolate = self.mutations(isolates)
        engine = RuleEngine(IsolateOntology(by_isolate))
        calls = {isolate: self.tier(engine, isolate) for isolate in by_isolate}
        return pd.Series(calls).reindex(isolates).fillna('below-MDR')

    def graded(self):
        """Resistance-graded rows from EFFECTS, read once. Both the tier arm and
        the per-isolate diagnosis ask for these, and the file is large."""
        if self.resistant is None:
            eff = flat(pd.read_parquet(self.effects_path,
                       columns=['UNIQUEID', 'GENE', 'MUTATION', 'DRUG', 'PREDICTION']),
                       'UNIQUEID')
            r = eff[eff['PREDICTION'].astype(str) == 'R'].copy()
            r['drug'] = r['DRUG'].astype(str).map(self.drugs)
            r['gene'] = r['GENE'].astype('string').fillna('NA')
            r['mutation'] = r['MUTATION'].astype('string').fillna('NA')
            self.resistant = r.dropna(subset=['drug'])
        return self.resistant

    def mutations(self, isolates):
        """Per-isolate mutation records. Grouping with a DataFrame per isolate
        rebuilt one frame per group, which dominated the run at CRyPTIC scale,
        so the rows are bucketed in a single pass over the columns instead."""
        rows = self.graded()
        rows = rows[rows['UNIQUEID'].isin(isolates)]
        by_isolate = {}
        for uid, gene, drug, mutation in zip(rows['UNIQUEID'].to_numpy(),
                                             rows['gene'].to_numpy(),
                                             rows['drug'].to_numpy(),
                                             rows['mutation'].to_numpy(), strict=True):
            by_isolate.setdefault(uid, []).append(
                {'gene': gene, 'drug': drug, 'mutation': mutation})
        return by_isolate

    @staticmethod
    def tier(engine, isolate):
        classes = engine.strain_recommendations(isolate)['classifications']
        return classes[0]['type'] if classes else 'below-MDR'


def exclusion_set(engine, isolate):
    """Drugs the rule engine flags for one isolate, direct and class cross-resistance."""
    recs = engine.strain_recommendations(isolate)
    return {e["drug"] for e in recs["exclusions"] if e["drug"]}


def engine_call_sets(evaluator, isolates):
    """Per-isolate set of drugs the rule engine flags as resistant."""
    by_isolate = evaluator.mutations(isolates)
    engine = RuleEngine(IsolateOntology(by_isolate))
    return {i: exclusion_set(engine, i) for i in by_isolate}


def calls_frame(call_sets, drugs):
    """Boolean per-drug call table from a mapping of isolate to flagged-drug set."""
    index = list(call_sets)
    data = {d: np.fromiter((d in call_sets[i] for i in index), dtype=float, count=len(index))
            for d in drugs}
    return pd.DataFrame(data, index=index)


def aligned_calls(call, truth, drugs):
    """Boolean call table on the truth index. An isolate the arm never saw
    carries no resistance call, which reads as susceptible."""
    return call.reindex(index=truth.index, columns=drugs).fillna(0.0) == 1.0


def drug_scores(truth, call, drugs):
    """Sensitivity, specificity, and precision per drug over isolates tested for it."""
    scores = {}
    for drug in drugs:
        measured = truth[drug].notna()
        scores[drug] = binary_rates(truth.loc[measured, drug] == 1.0,
                                    call.loc[measured, drug])
    return scores


def paired_scores(truth, first, second, drugs):
    """McNemar over the isolates tested for each drug. The two arms read the same
    rows, so the per-drug gap needs a paired test rather than two intervals."""
    scores = {}
    for drug in drugs:
        measured = truth[drug].notna()
        actual = (truth.loc[measured, drug] == 1.0).to_numpy()
        a = first.loc[measured, drug].to_numpy()
        b = second.loc[measured, drug].to_numpy()
        scores[drug] = mcnemar(int(((a != actual) & (b == actual)).sum()),
                               int(((a == actual) & (b != actual)).sum()))
    return scores


def per_drug_scores():
    """Per-drug sensitivity and specificity for the engine and catalog against DST.
    DST is measured phenotype and independent of both. The two predictions are
    not independent of each other, which the scheme note records."""
    drugs = drug_map(DATA / "DRUG_CODES.csv")
    dst = dst_truth(drugs)
    evaluator = RuleEngineEvaluator(DATA / "EFFECTS.parquet", drugs)
    catalog_raw = catalog_calls(evaluator.graded())
    targets = sorted(set(dst.columns) & set(catalog_raw.columns))
    engine = aligned_calls(calls_frame(engine_call_sets(evaluator, list(dst.index)), targets),
                           dst, targets)
    catalog = aligned_calls(catalog_raw, dst, targets)
    arms = {"rule_engine": drug_scores(dst, engine, targets),
            "who_catalog": drug_scores(dst, catalog, targets)}
    return {
        "eval_isolates": int(dst.index.size),
        "drugs": targets,
        "macro_f1": {name: macro_f1(scores) for name, scores in arms.items()},
        "scheme": "truth is measured DST, with disagreeing replicates resolved to R. Both "
                  "predictions derive from EFFECTS, so the engine calls are the catalog "
                  "calls plus class cross-resistance, the two columns are not independent, "
                  "and the paired tests across the three injectables share driving mutations "
                  "rather than being three separate confirmations. Drugs are the DST and "
                  "catalog columns intersected, so a drug the catalog never grades is left "
                  "out of both arms rather than scored at zero sensitivity in each",
        "paired": paired_scores(dst, engine, catalog, targets),
        **arms,
    }


def print_per_drug(summary):
    # Each drug is scored on the isolates carrying its own DST result, so the
    # denominator is reported per drug rather than taken from the header.
    print(f"\nPer-Drug Classification ({summary['eval_isolates']:,} isolates with any DST)")
    print(f"  {summary['scheme']}")
    for name in ("rule_engine", "who_catalog"):
        print(f"\n{name}  macro-F1 {summary['macro_f1'][name]:.3f}")
        for drug in summary["drugs"]:
            r = summary[name][drug]
            print(f"  {drug:14s}: sens {r['sensitivity']:.1%}  spec {r['specificity']:.1%}  "
                  f"ppv {r['precision']:.1%}  F1 {r['f1']:.3f}  "
                  f"(R={r['n']:,} of {r['evaluated']:,})")

    print("\nengine vs catalog, McNemar over the drugs the two arms differ on")
    for drug, m in summary["paired"].items():
        if m["discordant"]:
            print(f"  {drug:14s}: chi2 {m['chi2']:.2f}  p {m['p_value']:.2e}  "
                  f"{m['discordant']:,} discordant")


def main():
    summary = per_drug_scores()
    RESULTS.write_text(json.dumps(summary, indent=2))
    print_per_drug(summary)
    print(f"\nSaved {RESULTS.name} in {RESULTS.parent.name}")


if __name__ == "__main__":
    main()