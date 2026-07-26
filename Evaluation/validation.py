"""Validation for the TB hybrid system. Covers CBR cross-validation with bootstrap
intervals and calibration, expert-system query translation against gold queries, and
CRyPTIC classification."""

import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "SRC"))

from calibration import fit_temperature, scaled_confidence
from metrics import balanced_accuracy, brier, class_rates, macro_f1, mcnemar

SEED = 42
K_FOLDS = 5
N_CASES = 1000
N_BINS = 10
BOOTSTRAP_SAMPLES = 1000
CBR_NEIGHBORS = 10

RESULTS = EVAL_DIR / "validation_results.json"
EXPERT_CHECKPOINT = EVAL_DIR / ".expert_checkpoint.json"



def expert_checkpoint(model, results=None):
    if results is not None:
        EXPERT_CHECKPOINT.write_text(json.dumps({"model": model, "results": results}))
        return results
    if not EXPERT_CHECKPOINT.exists():
        return None
    saved = json.loads(EXPERT_CHECKPOINT.read_text())
    return saved["results"] if saved.get("model") == model else None


def expert_checkpoint_clear():
    EXPERT_CHECKPOINT.unlink(missing_ok=True)


# STATISTICS


def bootstrap_ci(values, n_samples=BOOTSTRAP_SAMPLES, confidence=0.95, rng=None):
    if not values:
        return 0.0, 0.0, 0.0

    rng = rng or np.random.default_rng()
    arr = np.asarray(values, dtype=float)
    draws = rng.choice(arr, size=(n_samples, arr.size), replace=True).mean(axis=1)

    alpha = 1.0 - confidence
    lower, upper = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(arr.mean()), float(lower), float(upper)


def rate(results, key):
    return sum(r[key] for r in results) / len(results) if results else 0.0


def accuracy_with_ci(values, rng=None):
    center, lower, upper = bootstrap_ci(values, rng=rng)
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        'mean': round(center, 3),
        'std': round(spread, 3),
        'ci_lower': round(lower, 3),
        'ci_upper': round(upper, 3),
    }


# CALIBRATION


def confidence_bins(predictions, n_bins=N_BINS):
    """Per-bin count, summed confidence, and summed hits."""
    conf = np.fromiter((c for c, _ in predictions), dtype=float, count=len(predictions))
    hit = np.fromiter((bool(y) for _, y in predictions), dtype=float, count=len(predictions))
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    return (np.bincount(idx, minlength=n_bins),
            np.bincount(idx, weights=conf, minlength=n_bins),
            np.bincount(idx, weights=hit, minlength=n_bins))


def expected_calibration_error(predictions, n_bins=N_BINS):
    if not predictions:
        return 0.0
    count, conf_sum, hit_sum = confidence_bins(predictions, n_bins)
    return round(float(np.abs(hit_sum - conf_sum).sum() / count.sum()), 4)


def reliability_diagram(predictions, n_bins=N_BINS):
    count, conf_sum, hit_sum = confidence_bins(predictions, n_bins)
    width = 1.0 / n_bins
    edges = np.arange(n_bins) * width
    safe = np.maximum(count, 1)
    filled = count > 0
    conf = np.where(filled, conf_sum / safe, edges + width / 2)
    acc = np.where(filled, hit_sum / safe, 0.0)

    return [{'bin': f"{edges[i]:.1f}-{edges[i] + width:.1f}",
             'confidence': round(float(conf[i]), 3),
             'accuracy': round(float(acc[i]), 3),
             'count': int(count[i])} for i in range(n_bins)]


# K-FOLD CBR VALIDATION


def stratified_folds(cases, k=K_FOLDS, rng=None):
    rng = rng or random.Random()
    by_profile = defaultdict(list)
    for case in cases:
        by_profile[case['profile']].append(case)

    for profile in by_profile:
        rng.shuffle(by_profile[profile])

    folds = [[] for _ in range(k)]
    for profile_cases in by_profile.values():
        for i, case in enumerate(profile_cases):
            folds[i % k].append(case)

    return [([c for j in range(k) if j != i for c in folds[j]], folds[i])
            for i in range(k)]


CBR_QUERY_KEYS = ('profile', 'hiv_status', 'age', 'region',
                  'diabetes', 'previous_treatment', 'sex')


def cbr_query(case):
    return {k: case[k] for k in CBR_QUERY_KEYS}


def neighbor_regimen_mode(similar_cases):
    """Most frequent regimen among the retrieved neighbors, ties broken by name."""
    counts = Counter(case['regimen'] for _, case in similar_cases)
    if not counts:
        return None
    top = max(counts.values())
    return min(regimen for regimen, c in counts.items() if c == top)


def evaluate_cbr_case(test_case, engine):
    analysis = engine.recommend(cbr_query(test_case), k=CBR_NEIGHBORS)

    recs = analysis['recommendations']
    predicted_regimen = recs[0]['regimen'] if recs else None
    mode_regimen = neighbor_regimen_mode(analysis['similar_cases']) or predicted_regimen
    actual_success = test_case['outcome'] == 'success'

    return {
        'regimen_correct': predicted_regimen == test_case['regimen'],
        'regimen_mode_correct': mode_regimen == test_case['regimen'],
        'outcome_correct': (analysis['success_rate'] >= 0.5) == actual_success,
        'confidence': analysis['confidence']['score'],
        'success_prob': analysis['success_rate'],
        'profile': test_case['profile'],
        'actual_success': actual_success,
        'actual_regimen': test_case['regimen'],
    }


def fold_scores(train, test, index):
    from cbr_engine import CBREngine

    engine = CBREngine(train)
    results = [evaluate_cbr_case(case, engine) for case in test]
    print(f"  fold {index}  regimen {rate(results, 'regimen_correct'):.1%}, "
          f"outcome {rate(results, 'outcome_correct'):.1%}")
    return results


def profile_accuracy(flat_results):
    by_profile = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in flat_results:
        by_profile[r['profile']]['total'] += 1
        by_profile[r['profile']]['correct'] += bool(r['regimen_correct'])

    return {p: {'accuracy': round(s['correct'] / s['total'], 3), 'n': s['total']}
            for p, s in by_profile.items()}


def baseline_accuracy(flat):
    n = len(flat)
    if not n:
        return {'outcome': 0.0, 'regimen': 0.0}
    success = sum(r['actual_success'] for r in flat) / n
    by_profile = defaultdict(Counter)
    for r in flat:
        by_profile[r['profile']][r['actual_regimen']] += 1
    majority = sum(c.most_common(1)[0][1] for c in by_profile.values()) / n
    return {'outcome': round(success, 3), 'regimen': round(majority, 3)}


def fold_temperatures(all_results):
    """Per-fold leak-free temperature scaling, kept to document the rejected calibration."""
    fold_preds = [[(r['success_prob'], 1.0 if r['actual_success'] else 0.0) for r in fold]
                  for fold in all_results]
    temperatures, calibrated = [], []

    for i, preds in enumerate(fold_preds):
        other = [p for j, fold in enumerate(fold_preds) if j != i for p in fold]
        t = fit_temperature([c for c, _ in other], [y for _, y in other])
        temperatures.append(round(t, 3))
        scaled = scaled_confidence(np.array([c for c, _ in preds]), t)
        calibrated.extend((float(s), bool(y)) for s, (_, y) in zip(scaled, preds, strict=True))

    return temperatures, calibrated


def aggregate_cbr_folds(all_results, k, seed=SEED):
    flat = [r for fold in all_results for r in fold]
    rng = np.random.default_rng(seed)
    predictions = [(r['success_prob'], r['actual_success']) for r in flat]
    temperatures, calibrated = fold_temperatures(all_results)

    return {
        'k': k,
        'total_cases': len(flat),
        'regimen_accuracy': accuracy_with_ci(
            [rate(f, 'regimen_correct') for f in all_results], rng),
        'outcome_accuracy': accuracy_with_ci(
            [rate(f, 'outcome_correct') for f in all_results], rng),
        'regimen_mode_accuracy': accuracy_with_ci(
            [rate(f, 'regimen_mode_correct') for f in all_results], rng),
        'by_profile': profile_accuracy(flat),
        'baseline': baseline_accuracy(flat),
        'calibration': {
            'ece': expected_calibration_error(predictions),
            'brier': brier(predictions),
            'ece_temperature_scaled': expected_calibration_error(calibrated),
            'temperature_mean': round(float(np.mean(temperatures)), 3),
            'temperature_per_fold': temperatures,
            'reliability': reliability_diagram(predictions),
        },
    }


def validate_cbr(cases, k=K_FOLDS, seed=SEED):
    print(f"\nCBR {k}-fold cross-validation")
    splits = stratified_folds(cases, k, random.Random(seed))
    folds = [fold_scores(train, test, i + 1) for i, (train, test) in enumerate(splits)]
    return aggregate_cbr_folds(folds, k, seed)


# EXPERT SYSTEM VALIDATION
# Each gold query is the certified answer. A generated query passes when it
# returns the same rows, so one scoring method covers every query. The score is
# conditional on the model that wrote the Cypher, recorded next to it.

EXPERT_QUERIES = [
    {'id': 1, 'category': 'lookup',
     'question': 'What mutations cause rifampin resistance?',
     'gold': "MATCH (m:Mutation)-[:CONFERS_RESISTANCE]->(:Drug {name: 'rifampin'}) "
             "RETURN DISTINCT m.mutation_id AS mutation ORDER BY mutation"},
    {'id': 2, 'category': 'lookup',
     'question': 'What drugs is strain TB001 resistant to?',
     'gold': "MATCH (:Strain {strain_id: 'TB001'})-[:HAS_MUTATION]->(:Mutation)"
             "-[:CONFERS_RESISTANCE]->(d:Drug) RETURN DISTINCT d.name AS drug ORDER BY drug"},
    {'id': 3, 'category': 'filter',
     'question': 'Show all MDR strains',
     'gold': "MATCH (s:Strain)-[:HAS_PROFILE]->(:ResistanceProfile {type: 'MDR'}) "
             "RETURN s.strain_id AS strain ORDER BY strain"},
    {'id': 4, 'category': 'aggregation',
     'question': 'How many resistance mutations does each gene have?',
     'gold': "MATCH (m:Mutation)-[:CONFERS_RESISTANCE]->(:Drug) "
             "MATCH (m)-[:IN_GENE]->(g:Gene) "
             "RETURN g.name AS gene, count(DISTINCT m) AS mutations ORDER BY gene"},
    {'id': 5, 'category': 'spelling',
     'question': 'Show rifampicin resistant strains',
     'gold': "MATCH (s:Strain)-[:HAS_MUTATION]->(:Mutation)"
             "-[:CONFERS_RESISTANCE]->(:Drug {name: 'rifampin'}) "
             "RETURN DISTINCT s.strain_id AS strain ORDER BY strain"},
    {'id': 6, 'category': 'negation',
     'question': 'Which strains do not have rifampin resistance?',
     'gold': "MATCH (s:Strain) WHERE NOT (s)-[:HAS_MUTATION]->(:Mutation)"
             "-[:CONFERS_RESISTANCE]->(:Drug {name: 'rifampin'}) "
             "RETURN s.strain_id AS strain ORDER BY strain"},
    {'id': 7, 'category': 'conjunction',
     'question': 'Show MDR strains from India with gyrA mutations',
     'gold': "MATCH (s:Strain)-[:HAS_PROFILE]->(:ResistanceProfile {type: 'MDR'}) "
             "WHERE s.country = 'India' AND (s)-[:HAS_MUTATION]->(:Mutation)"
             "-[:IN_GENE]->(:Gene {name: 'gyrA'}) RETURN s.strain_id AS strain"},
    {'id': 8, 'category': 'unknown_entity',
     'question': 'What resistance does mutation rpoB_X999Y confer?',
     'gold': "MATCH (:Mutation {mutation_id: 'rpoB_X999Y'})"
             "-[:CONFERS_RESISTANCE]->(d:Drug) RETURN d.name AS drug"},
    {'id': 9, 'category': 'no_results',
     'question': 'Show XDR strains from Antarctica',
     'gold': "MATCH (s:Strain)-[:HAS_PROFILE]->(:ResistanceProfile {type: 'XDR'}) "
             "WHERE s.country = 'Antarctica' RETURN s.strain_id AS strain"},
    {'id': 10, 'category': 'unanswerable',
     'question': 'Change strain TB001 to susceptible', 'unanswerable': True},
    {'id': 11, 'category': 'unanswerable',
     'question': 'What is the home address of patient P001?', 'unanswerable': True},
]


def expert_queries():
    return EXPERT_QUERIES


def row_values(row):
    """Canonical value set of one result row, free of column order and name."""
    return frozenset(json.dumps(v, sort_keys=True, default=str) for v in row.values())


def covers(gold, produced):
    """True when each gold row's values sit inside a distinct produced row."""
    pool = [row_values(r) for r in produced]
    for want in sorted((row_values(r) for r in gold), key=len, reverse=True):
        match = next((i for i, have in enumerate(pool) if want <= have), None)
        if match is None:
            return False
        pool.pop(match)
    return True


def same_answer(gold, produced):
    """True when produced returns the gold rows, with extra columns allowed."""
    return len(gold) == len(produced) and covers(gold, produced)


def query_result(item, passed, count, start, failure=None, detail=None):
    result = {
        'id': item['id'],
        'category': item['category'],
        'passed': passed,
        'result_count': count,
        'time_ms': round((time.perf_counter() - start) * 1000, 1),
    }
    if failure:
        result['failure'] = failure
    if detail:
        result.update(detail)
    return result


def evaluate_query(item, nl_interface):
    """Score one query by matching its result set against the gold query."""
    start = time.perf_counter()
    cypher = nl_interface.generate_cypher(item['question'])
    valid, _ = nl_interface.validate_cypher(cypher)
    refused = (not valid) or 'UNANSWERABLE' in cypher

    if item.get('unanswerable'):
        return query_result(item, refused, 0, start,
                            None if refused else 'answered an unanswerable question',
                            None if refused else {'cypher': cypher})
    if refused:
        return query_result(item, False, 0, start, 'rejected a valid question', {'cypher': cypher})

    try:
        produced = nl_interface.execute_query(cypher)
        expected = nl_interface.execute_query(item['gold'])
    except Exception as exc:
        return query_result(item, False, 0, start, str(exc), {'cypher': cypher})

    passed = same_answer(expected, produced)
    detail = None if passed else {'cypher': cypher, 'expected_count': len(expected)}
    return query_result(item, passed, len(produced), start,
                        None if passed else 'result set differs from gold', detail)


def category_rates(results):
    groups = defaultdict(lambda: [0, 0])
    for r in results:
        tally = groups[r['category']]
        tally[0] += bool(r['passed'])
        tally[1] += 1
    return {name: {'rate': round(hit / total, 3), 'n': total}
            for name, (hit, total) in groups.items()}


def expert_accuracy(results, model):
    total = len(results)
    hits = sum(r['passed'] for r in results)
    return {
        'model': model,
        'method': 'execution match of generated Cypher against a gold query',
        'overall': {'rate': round(hits / total, 3) if total else 0.0, 'n': total},
        'by_category': category_rates(results),
        'failures': [r for r in results if not r['passed']],
    }


def validate_expert_system(nl_interface, resume=False):
    from nl_interface import MODEL

    print("\nExpert system validation")
    results = (expert_checkpoint(MODEL) if resume else None) or []
    done = {r['id'] for r in results}

    for item in EXPERT_QUERIES:
        if item['id'] in done:
            continue
        try:
            result = evaluate_query(item, nl_interface)
        except Exception as exc:
            result = query_result(item, False, 0, time.perf_counter(), str(exc))
            result['errored'] = True
        results.append(result)
        if resume:
            expert_checkpoint(MODEL, [r for r in results if not r.get('errored')])
        state = 'ERROR' if result.get('errored') else ('PASS' if result['passed'] else 'FAIL')
        print(f"  {item['id']:>3} {state:5s} {item['category']}")

    return expert_accuracy(results, MODEL)


METHODOLOGY = {
    'cbr': 'stratified cross-validation',
    'confidence_intervals': f'95% bootstrap (n={BOOTSTRAP_SAMPLES})',
    'calibration': 'ECE of the predicted success probability vs actual outcome; raw '
                   'probability reported. Per-fold temperature scaling (leak-free) was '
                   'tested and rejected as it raised ECE.',
    'baseline': 'outcome=always-predict-success; regimen=most-frequent-regimen-per-profile',
    'regimen_mode': 'diagnostic predictor: most-frequent regimen among retrieved neighbors '
                    '(ignores outcome), to separate objective mismatch from weak retrieval',
}


def report_file(expert=None, cbr=None, cryptic=None, path=RESULTS):
    """Merges into any existing report, so a skipped arm keeps its last result."""
    data = json.loads(path.read_text()) if path.exists() else {}
    arms = {'expert_system': expert, 'cbr': cbr, 'cryptic_classification': cryptic}
    fresh = {name: result for name, result in arms.items() if result is not None}

    data.update(fresh)
    data['timestamp'] = datetime.now().isoformat()
    data['arms_this_run'] = sorted(fresh)

    if cbr is not None:
        data['methodology'] = dict(METHODOLOGY, cbr=f"{cbr['k']}-fold stratified cross-validation")

    path.write_text(json.dumps(data, indent=2))
    return data


def print_expert_summary(expert):
    print("\nExpert system, natural language to Cypher")
    print(f"  model     {expert['model']}")
    print(f"  accuracy  {expert['overall']['rate']:.1%} "
          f"(n={expert['overall']['n']}, execution match)")
    for name, stats in expert['by_category'].items():
        print(f"    {name:15s} {stats['rate']:.1%} (n={stats['n']})")


def print_cbr_summary(cbr):
    reg, out = cbr['regimen_accuracy'], cbr['outcome_accuracy']
    mode, cal, base = cbr['regimen_mode_accuracy'], cbr['calibration'], cbr['baseline']

    print(f"\nCBR, {cbr['k']}-fold cross-validation")
    print(f"  regimen        {reg['mean']:.1%} [{reg['ci_lower']:.1%}, {reg['ci_upper']:.1%}]")
    print(f"  regimen mode   {mode['mean']:.1%} [{mode['ci_lower']:.1%}, {mode['ci_upper']:.1%}]")
    print(f"  outcome        {out['mean']:.1%} [{out['ci_lower']:.1%}, {out['ci_upper']:.1%}]")
    print(f"  ECE            {cal['ece']:.4f} raw, {cal['ece_temperature_scaled']:.4f} "
          f"temperature scaled (rejected, T={cal['temperature_mean']})")
    print(f"  Brier          {cal['brier']:.4f}")
    print(f"  baseline       regimen {base['regimen']:.1%}, outcome {base['outcome']:.1%}")

    print("\nCBR by profile")
    for profile, p in cbr['by_profile'].items():
        print(f"  {profile:14s} {p['accuracy']:.1%} (n={p['n']})")


def print_summary(data):
    print_expert_summary(data['expert_system'])
    print_cbr_summary(data['cbr'])


# CRYPTIC CLASSIFICATION VALIDATION
# The rule engine's resistance classification against measured CRyPTIC phenotypes,
# with the WHO catalog as reference. Database-free. Classes collapse to
# below-MDR / MDR / PreXDR / XDR, and no genotypic call counts as below-MDR.
# Heavy imports stay deferred so the expert-system and CBR paths load light.

RESISTANT_TIERS = ("MDR", "PreXDR", "XDR")
COLLAPSED = ["below-MDR", "MDR", "PreXDR", "XDR"]


def collapse_tier(label):
    return label if label in RESISTANT_TIERS else "below-MDR"


def tier_accuracy(truth, prediction):
    """Overall, balanced, and macro-F1 accuracy with per-tier sensitivity and specificity."""
    rates = {t: r for t, r in class_rates(truth, prediction, COLLAPSED).items() if r['n']}
    overall = float((prediction == truth).mean()) if len(truth) else 0.0
    return {
        'overall': round(overall, 3),
        'balanced': balanced_accuracy(rates),
        'macro_f1': macro_f1(rates),
        'by_tier': {t: {'accuracy': r['sensitivity'], 'n': r['n']} for t, r in rates.items()},
        'rates': rates,
    }


def confusion(truth, prediction):
    import pandas as pd

    table = pd.crosstab(truth, prediction).reindex(index=COLLAPSED, columns=COLLAPSED,
                                                   fill_value=0)
    return {t: {c: int(table.loc[t, c]) for c in COLLAPSED} for t in COLLAPSED}


def agreement(truth, engine, catalog):
    """Splits resistant-tier errors into shared (biological ceiling) and engine-only."""
    resistant = truth.isin(RESISTANT_TIERS)
    engine_ok, catalog_ok = engine == truth, catalog == truth
    engine_only = int((resistant & ~engine_ok & catalog_ok).sum())
    catalog_only = int((resistant & engine_ok & ~catalog_ok).sum())

    return {
        'engine_catalog_match': round(float((engine == catalog).mean()), 3),
        'resistant_isolates': int(resistant.sum()),
        'both_correct': int((resistant & engine_ok & catalog_ok).sum()),
        'both_wrong': int((resistant & ~engine_ok & ~catalog_ok).sum()),
        'engine_only_wrong': engine_only,
        'catalog_only_wrong': catalog_only,
        'mcnemar': mcnemar(engine_only, catalog_only),
    }


def diagnose(engine_eval, truth, engine, catalog):
    """The resistant cases the catalog gets right but the engine misses, with their mutations."""
    mask = truth.isin(RESISTANT_TIERS) & (engine != truth) & (catalog == truth)
    ids = list(truth[mask].index)
    by_isolate = engine_eval.mutations(ids)

    return [{
        'uniqueid': isolate,
        'truth': truth[isolate],
        'engine': engine[isolate],
        'catalog': catalog[isolate],
        'resistant_drugs': sorted({r['drug'] for r in by_isolate.get(isolate, [])}),
        'mutations': [f"{r['gene']}_{r['mutation']}" for r in by_isolate.get(isolate, [])],
    } for isolate in ids]


class IsolateOntology:
    """Feeds per-isolate mutations to the rule engine without a database."""

    def __init__(self, mutations):
        self.mutations = mutations

    def patient_strain_mapping(self, strain_id):
        return None

    def strain_mutations_detailed(self, strain_id):
        return self.mutations.get(strain_id, [])


class RuleEngineEvaluator:
    name = 'rule_engine'

    def __init__(self, effects_path, drugs):
        self.effects_path = effects_path
        self.drugs = drugs

    def predictions(self, isolates):
        import pandas as pd
        from rule_engine import RuleEngine

        by_isolate = self.mutations(isolates)
        engine = RuleEngine(IsolateOntology(by_isolate))
        engine.build_rules()
        calls = {isolate: self.tier(engine, isolate) for isolate in by_isolate}
        return pd.Series(calls).reindex(isolates).fillna('below-MDR')

    def mutations(self, isolates):
        import pandas as pd
        from feature_engineering import flat

        eff = flat(pd.read_parquet(self.effects_path,
                   columns=['UNIQUEID', 'GENE', 'MUTATION', 'DRUG', 'PREDICTION']), 'GENE')
        r = eff[(eff['PREDICTION'].astype(str) == 'R') & eff['UNIQUEID'].isin(isolates)].copy()
        r['drug'] = r['DRUG'].astype(str).map(self.drugs)
        r['gene'] = r['GENE'].astype('string').fillna('NA')
        r['mutation'] = r['MUTATION'].astype('string').fillna('NA')
        r = r.dropna(subset=['drug'])
        return {isolate: g[['gene', 'drug', 'mutation']].assign(position='').to_dict('records')
                for isolate, g in r.groupby('UNIQUEID')}

    @staticmethod
    def tier(engine, isolate):
        classes = engine.evaluate_strain(isolate)['recommendations']['classifications']
        return classes[0]['type'] if classes else 'below-MDR'


class CatalogEvaluator:
    name = 'who_catalog'

    def __init__(self, catalog):
        self.catalog = catalog

    def predictions(self, isolates):
        profiles = self.catalog.reindex(isolates)
        collapsed = profiles.map(lambda p: collapse_tier(p) if isinstance(p, str) else p)
        return collapsed.fillna('below-MDR')


class ClassificationValidation:
    """Runs every classification sub-unit on all labeled isolates and scores them together."""

    def __init__(self):
        from feature_engineering import DATA, dataset, drug_map

        data = dataset()
        self.data_dir = DATA
        self.labeled = data.set_index('uniqueid')
        self.drugs = drug_map(DATA / 'DRUG_CODES.csv')
        self.truth = self.labeled['label'].map(collapse_tier)

    def summary(self):
        engine_eval = RuleEngineEvaluator(self.data_dir / 'EFFECTS.parquet', self.drugs)
        catalog_eval = CatalogEvaluator(self.labeled['catalog'])
        preds = {
            engine_eval.name: engine_eval.predictions(self.labeled.index),
            catalog_eval.name: catalog_eval.predictions(self.labeled.index),
        }
        scores = {name: {**tier_accuracy(self.truth, p), 'confusion': confusion(self.truth, p)}
                  for name, p in preds.items()}

        return {
            'eval_isolates': len(self.labeled),
            'scheme': 'below-MDR / MDR / PreXDR / XDR; no genotypic call counts as below-MDR',
            'scores': scores,
            'agreement': agreement(self.truth, preds['rule_engine'], preds['who_catalog']),
            'engine_only_cases': diagnose(engine_eval, self.truth,
                                          preds['rule_engine'], preds['who_catalog']),
        }


def validate_classification():
    return ClassificationValidation().summary()


def print_class_scores(scores):
    for name, score in scores.items():
        print(f"\n{name}  overall {score['overall']:.1%}, balanced {score['balanced']:.1%}, "
              f"macro-F1 {score['macro_f1']:.3f}")
        for tier in COLLAPSED:
            if tier in score['rates']:
                r = score['rates'][tier]
                print(f"  {tier:10s} sens {r['sensitivity']:.1%}  spec {r['specificity']:.1%}  "
                      f"ppv {r['precision']:.1%}  (R={r['n']})")


def print_class_confusion(score):
    print("\nrule engine confusion, rows truth and columns predicted")
    table = score['confusion']
    print("            " + "".join(f"{c:>11s}" for c in COLLAPSED))
    for truth in COLLAPSED:
        row = "".join(f"{table[truth][c]:>11d}" for c in COLLAPSED)
        print(f"  {truth:10s}{row}")


def print_class_agreement(agree):
    mc = agree['mcnemar']
    print("\nengine vs catalog on resistant-truth isolates")
    print(f"  prediction match, all isolates  {agree['engine_catalog_match']:.1%}")
    print(f"  both wrong, biological ceiling  {agree['both_wrong']}")
    print(f"  engine only wrong, fixable      {agree['engine_only_wrong']}")
    print(f"  catalog only wrong              {agree['catalog_only_wrong']}")
    print(f"  McNemar chi2 {mc['chi2']}, p {mc['p_value']:.2e}, {mc['discordant']} discordant")


def print_classification(summary):
    print(f"\nCRyPTIC classification validation, {summary['eval_isolates']:,} labeled isolates")
    print_class_scores(summary['scores'])
    print_class_confusion(summary['scores']['rule_engine'])
    print_class_agreement(summary['agreement'])


def graph_ontology():
    from tb_ontology import TBOntology

    print("\nRebuilding knowledge graph, the existing contents are cleared")
    ontology = TBOntology()
    ontology.clear_database()
    ontology.schema()
    ontology.ontology_classes()

    try:
        ontology.who_mutations()
        ontology.count_who_mutations()
    except Exception as exc:
        print(f"  WHO catalog skipped ({exc})")

    return ontology


def system_validation(resume=False):
    from cbr_cases import generate_cases
    from nl_interface import NLInterface

    ontology = graph_ontology()
    try:
        expert = validate_expert_system(NLInterface(ontology), resume)
    finally:
        ontology.close()

    cases = generate_cases(N_CASES, seed=SEED)
    return expert, validate_cbr(cases, K_FOLDS, SEED)


def main():
    load_dotenv(override=True)
    fresh = "--fresh" in sys.argv
    resume = not fresh
    if fresh:
        expert_checkpoint_clear()

    expert = cbr = cryptic = None

    try:
        expert, cbr = system_validation(resume)
    except Exception as exc:
        print(f"\nSystem validation skipped, graph or API unavailable ({exc})")

    try:
        cryptic = validate_classification()
        print_classification(cryptic)
    except Exception as exc:
        print(f"\nClassification validation skipped ({exc})")

    data = report_file(expert, cbr, cryptic)
    if expert and cbr:
        print_summary(data)
    print(f"\nSaved {RESULTS.name} in {RESULTS.parent.name}")


if __name__ == "__main__":
    main()
