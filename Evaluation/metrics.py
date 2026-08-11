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

from feature_engineering import DATA, drug_map, flat, named_calls
from rule_engine import RuleEngine

RESULTS = EVAL_DIR / "per_drug_results.json"
N_BINS = 10


def safe_ratio(num, den):
    """Ratio, or zero when nothing was counted. Unrounded, since the macro
    averages are taken over these."""
    return num / den if den else 0.0


def binary_rates(actual, predicted):
    """Sensitivity, specificity, precision, and F1 from boolean arrays. One
    definition, read by both the per-drug and per-tier arms."""
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
    """Wilson score interval, which stays inside zero and one at small n."""
    if not n:
        return 0.0, 0.0
    z = float(norm.ppf(0.5 + confidence / 2))
    p, shift = hits / n, z * z / n
    center = (p + shift / 2) / (1 + shift)
    spread = z * math.sqrt(p * (1 - p) / n + shift / (4 * n)) / (1 + shift)
    return round(center - spread, 3), round(center + spread, 3)


def mcnemar(b, c):
    """Continuity-corrected McNemar for paired classifiers. The correction is
    floored at zero, so equal disagreement returns no statistic."""
    n = b + c
    if not n:
        return {'chi2': 0.0, 'p_value': 1.0, 'discordant': 0}
    stat = max(0.0, abs(b - c) - 1.0) ** 2 / n
    return {'chi2': round(stat, 2), 'p_value': float(chi2.sf(stat, 1)), 'discordant': n}


def confidence_bins(predictions, n_bins=N_BINS):
    """Per-bin count, summed confidence, and summed hits."""
    conf = np.fromiter((c for c, _ in predictions), dtype=float, count=len(predictions))
    hit = np.fromiter((bool(y) for _, y in predictions), dtype=float, count=len(predictions))
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return (np.bincount(idx, minlength=n_bins),
            np.bincount(idx, weights=conf, minlength=n_bins),
            np.bincount(idx, weights=hit, minlength=n_bins))


def expected_calibration_error(predictions, n_bins=N_BINS):
    """Count-weighted gap between confidence and accuracy across the bins."""
    if not predictions:
        return 0.0
    count, conf_sum, hit_sum = confidence_bins(predictions, n_bins)
    return round(float(np.abs(hit_sum - conf_sum).sum() / count.sum()), 4)


def reliability_diagram(predictions, n_bins=N_BINS):
    """Per-bin confidence and accuracy. An empty bin reports null, not zero."""
    count, conf_sum, hit_sum = confidence_bins(predictions, n_bins)
    width = 1.0 / n_bins
    edges = np.arange(n_bins) * width
    safe = np.maximum(count, 1)
    filled = count > 0
    conf = np.where(filled, conf_sum / safe, edges + width / 2)
    acc = hit_sum / safe

    return [{'bin': f"{edges[i]:.1f}-{edges[i] + width:.1f}",
             'confidence': round(float(conf[i]), 3),
             'accuracy': round(float(acc[i]), 3) if filled[i] else None,
             'count': int(count[i])} for i in range(n_bins)]


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
    """Brier score of a constant prediction at the base rate, p(1-p). The floor
    a probability must beat to carry information."""
    if not predictions:
        return 0.0
    _, labels = prediction_arrays(predictions)
    base = float(labels.mean())
    return round(base * (1.0 - base), 4)


def auc(predictions):
    """Area under the ROC curve by rank sum, so ties average. Zero when either
    class is absent."""
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
    """Measured R/S call per isolate and drug, NaN where untested. Replicates
    aggregate by max, and the rows are the ones the tier label reads."""
    _, calls = named_calls(DATA / "DST_MEASUREMENTS.parquet", "PHENOTYPE", drugs)
    return calls.assign(call=calls["resistant"].astype(float)).groupby(
        ["UNIQUEID", "drug"], observed=True)["call"].max().unstack()


def catalog_calls(graded):
    """Genotypic call per isolate and drug, from rows the evaluator already holds."""
    return graded[["UNIQUEID", "drug"]].assign(call=1.0).groupby(
        ["UNIQUEID", "drug"], observed=True)["call"].max().unstack()


class IsolateOntology:
    """Feeds per-isolate mutations to the rule engine without a database."""

    def __init__(self, mutations):
        self.mutations = mutations

    def patient_strain_mapping(self, strain_id):
        """Isolate ids never begin with P, so the id maps to itself."""
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
        """Resistance-graded rows from EFFECTS, read once and held."""
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
        """Per-isolate mutation records. Bucketed in one pass rather than grouped,
        since a frame per isolate dominated the run at this scale."""
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
    """Boolean call table on the truth index. An unseen isolate reads susceptible."""
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
    """McNemar per drug over the isolates tested for it. The arms read the same
    rows, so the gap needs a paired test."""
    scores = {}
    for drug in drugs:
        measured = truth[drug].notna()
        actual = (truth.loc[measured, drug] == 1.0).to_numpy()
        a = first.loc[measured, drug].to_numpy()
        b = second.loc[measured, drug].to_numpy()
        scores[drug] = mcnemar(int(((a != actual) & (b == actual)).sum()),
                               int(((a == actual) & (b != actual)).sum()))
    return scores


def superset(engine, catalog):
    """True when every catalog call is also an engine call. Measured, not asserted,
    since it is what makes the arms dependent."""
    return not (catalog & ~engine).to_numpy().any()


def per_drug_scores():
    """Per-drug rates for the engine and catalog against measured DST."""
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
        "macro_f1": {name: macro_f1(scores) for name, scores in arms.items()},
        "scheme": {
            "truth_table": "DST_MEASUREMENTS",
            "prediction_table": "EFFECTS",
            "replicates": "max",
            "drug_selection": "intersection",
            "engine_extends_catalog": superset(engine, catalog),
        },
        "paired": paired_scores(dst, engine, catalog, targets),
        **arms,
    }


def print_per_drug(summary):
    """One row per drug, engine rates beside the catalog F1 and the paired gap.
    Each drug carries its own denominator, since it is scored on the isolates
    tested for it."""
    macro = summary["macro_f1"]
    print(f"\nPer-drug, {summary['eval_isolates']:,} isolates with any DST"
          f"   macro-F1  engine {macro['rule_engine']:.3f}  catalog {macro['who_catalog']:.3f}")
    print(f"\n  {'drug':14s}{'R':>8s}{'tested':>9s}{'sens':>8s}{'spec':>7s}"
          f"{'ppv':>7s}{'F1':>8s}{'catalog F1':>12s}{'discordant':>12s}")

    for drug, r in summary["rule_engine"].items():
        gap = summary["paired"][drug]["discordant"]
        print(f"  {drug:14s}{r['n']:>8,}{r['evaluated']:>9,}{r['sensitivity']:>8.1%}"
              f"{r['specificity']:>7.1%}{r['precision']:>7.1%}{r['f1']:>8.3f}"
              f"{summary['who_catalog'][drug]['f1']:>12.3f}"
              f"{(f'{gap:,}' if gap else '-'):>12s}")


def main():
    summary = per_drug_scores()
    RESULTS.write_text(json.dumps(summary, indent=2))
    print_per_drug(summary)
    print(f"\nSaved {RESULTS.name} in {RESULTS.parent.name}")


if __name__ == "__main__":
    main()