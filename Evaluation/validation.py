"""Validation for the TB hybrid system, covering CBR cross-validation with bootstrap
intervals and calibration, expert-system query translation against gold queries, and
CRyPTIC classification."""

import json
import time
import random
import math
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from dotenv import load_dotenv

# This file lives in Evaluation/ and imports core modules from the
# sibling SRC/ folder.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "SRC"))

from calibration import fit_temperature, scaled_confidence
from metrics import balanced_accuracy, brier, class_rates, macro_f1, mcnemar


# STATISTICS


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def bootstrap_ci(values, n_samples=1000, confidence=0.95, rng=None):
    if not values:
        return 0.0, 0.0, 0.0

    if rng is None:
        rng = np.random.default_rng()

    arr = np.asarray(values, dtype=float)
    draws = rng.choice(arr, size=(n_samples, arr.size), replace=True).mean(axis=1)

    alpha = 1.0 - confidence
    lower, upper = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])

    return float(arr.mean()), float(lower), float(upper)


# CALIBRATION

def expected_calibration_error(predictions, n_bins=10):
    if not predictions:
        return 0.0

    bins = defaultdict(list)
    for conf, correct in predictions:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, 1.0 if correct else 0.0))

    ece = 0.0
    total = len(predictions)

    for bin_data in bins.values():
        if not bin_data:
            continue
        n = len(bin_data)
        avg_conf = sum(c for c, _ in bin_data) / n
        avg_acc = sum(a for _, a in bin_data) / n
        ece += (n / total) * abs(avg_acc - avg_conf)

    return round(ece, 4)


def reliability_diagram(predictions, n_bins=10):
    bins = defaultdict(list)
    for conf, correct in predictions:
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, 1.0 if correct else 0.0))

    data = []
    for i in range(n_bins):
        bin_data = bins[i]
        if bin_data:
            avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
            avg_acc = sum(a for _, a in bin_data) / len(bin_data)
            count = len(bin_data)
        else:
            avg_conf, avg_acc, count = (i + 0.5) / n_bins, 0.0, 0

        data.append({
            'bin': f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}",
            'confidence': round(avg_conf, 3),
            'accuracy': round(avg_acc, 3),
            'count': count
        })

    return data


# K-FOLD CBR VALIDATION


def stratified_folds(cases, k=5, rng=None):
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

    splits = []
    for i in range(k):
        test = folds[i]
        train = [c for j in range(k) if j != i for c in folds[j]]
        splits.append((train, test))

    return splits


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
    analysis = engine.recommend(cbr_query(test_case), k=10)

    recs = analysis['recommendations']
    predicted_regimen = recs[0]['regimen'] if recs else None
    mode_regimen = neighbor_regimen_mode(analysis['similar_cases']) or predicted_regimen

    predicted_success = analysis['success_rate'] >= 0.5
    actual_success = test_case['outcome'] == 'success'

    return {
        'regimen_correct': predicted_regimen == test_case['regimen'],
        'regimen_mode_correct': mode_regimen == test_case['regimen'],
        'outcome_correct': predicted_success == actual_success,
        'confidence': analysis['confidence']['score'],
        'success_prob': analysis['success_rate'],
        'profile': test_case['profile'],
        'actual_success': actual_success,
        'actual_regimen': test_case['regimen']
    }


def run_fold(train, test, fold_idx):
    from cbr_engine import CBREngine

    engine = CBREngine(train)
    results = [evaluate_cbr_case(case, engine) for case in test]

    regimen_acc = sum(r['regimen_correct'] for r in results) / len(results)
    outcome_acc = sum(r['outcome_correct'] for r in results) / len(results)

    print(f"  Fold {fold_idx}: Regimen {regimen_acc:.1%}, Outcome {outcome_acc:.1%}")

    return results


def profile_accuracy(flat_results):
    by_profile = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in flat_results:
        by_profile[r['profile']]['total'] += 1
        if r['regimen_correct']:
            by_profile[r['profile']]['correct'] += 1

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


def accuracy_with_ci(values, rng=None):
    m, lo, hi = bootstrap_ci(values, rng=rng)
    return {
        'mean': round(m, 3),
        'std': round(std(values), 3),
        'ci_lower': round(lo, 3),
        'ci_upper': round(hi, 3)
    }


def fold_temperatures(all_results):
    """Per-fold leak-free temperature scaling, kept to document the rejected calibration that raises ECE."""
    fold_preds = [[(r['success_prob'], 1.0 if r['actual_success'] else 0.0) for r in fold]
                  for fold in all_results]
    temperatures = []
    calibrated = []
    for i, preds in enumerate(fold_preds):
        other = [p for j, fold in enumerate(fold_preds) if j != i for p in fold]
        t = fit_temperature([c for c, _ in other], [y for _, y in other])
        temperatures.append(round(t, 3))
        scaled = scaled_confidence(np.array([c for c, _ in preds]), t)
        calibrated.extend((float(s), bool(y)) for s, (_, y) in zip(scaled, preds))
    return temperatures, calibrated


def aggregate_cbr_folds(all_results, k, seed=42):
    flat = [r for fold in all_results for r in fold]
    rng = np.random.default_rng(seed)

    regimen_accs = [sum(r['regimen_correct'] for r in fold) / len(fold)
                    for fold in all_results]
    outcome_accs = [sum(r['outcome_correct'] for r in fold) / len(fold)
                    for fold in all_results]
    regimen_mode_accs = [sum(r['regimen_mode_correct'] for r in fold) / len(fold)
                         for fold in all_results]

    predictions = [(r['success_prob'], r['actual_success']) for r in flat]
    temperatures, calibrated = fold_temperatures(all_results)

    return {
        'k': k,
        'total_cases': len(flat),
        'regimen_accuracy': accuracy_with_ci(regimen_accs, rng),
        'outcome_accuracy': accuracy_with_ci(outcome_accs, rng),
        'regimen_mode_accuracy': accuracy_with_ci(regimen_mode_accs, rng),
        'by_profile': profile_accuracy(flat),
        'baseline': baseline_accuracy(flat),
        'calibration': {
            'ece': expected_calibration_error(predictions),
            'brier': brier(predictions),
            'ece_temperature_scaled': expected_calibration_error(calibrated),
            'temperature_mean': round(float(np.mean(temperatures)), 3),
            'temperature_per_fold': temperatures,
            'reliability': reliability_diagram(predictions)
        }
    }


def validate_cbr(cases, k=5, seed=42):
    print(f"\nCBR {k}-Fold Cross-Validation")

    splits = stratified_folds(cases, k, random.Random(seed))
    all_results = []

    for i, (train, test) in enumerate(splits):
        results = run_fold(train, test, i + 1)
        all_results.append(results)

    return aggregate_cbr_folds(all_results, k, seed)


# EXPERT SYSTEM VALIDATION

# The natural-language layer turns a question into Cypher. Each gold query is the
# certified answer, and a generated query passes when it returns the same rows.
# One scoring method covers every query, so they stay comparable, and the score
# is conditional on the model that wrote the Cypher, recorded next to it.

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


def expert_accuracy(results):
    """Overall and per-category accuracy, tagged with the model that produced it."""
    from nl_interface import MODEL
    total = len(results)
    hits = sum(r['passed'] for r in results)
    return {
        'model': MODEL,
        'method': 'execution match of generated Cypher against a gold query',
        'overall': {'rate': round(hits / total, 3) if total else 0.0, 'n': total},
        'by_category': category_rates(results),
        'failures': [r for r in results if not r['passed']],
    }


def validate_expert_system(nl_interface):
    print("\nExpert System Validation")
    results = []
    for item in EXPERT_QUERIES:
        try:
            result = evaluate_query(item, nl_interface)
        except Exception as exc:
            result = query_result(item, False, 0, time.perf_counter(), str(exc))
        results.append(result)
        print(f"  {item['id']:>3} {'PASS' if result['passed'] else 'FAIL':4s} {item['category']}")
    return expert_accuracy(results)


def report_file(kg_results=None, cbr_results=None, cryptic_results=None,
                filename='validation_results.json'):
    data = {'timestamp': datetime.now().isoformat()}

    if kg_results is not None:
        data['expert_system'] = kg_results

    if cbr_results is not None:
        data['cbr'] = cbr_results
        data['methodology'] = {
            'cbr': f"{cbr_results['k']}-fold stratified cross-validation",
            'confidence_intervals': '95% bootstrap (n=1000)',
            'calibration': 'ECE of the predicted success probability vs actual outcome; raw probability reported. Per-fold temperature scaling (leak-free) was tested and rejected as it raised ECE.',
            'baseline': 'outcome=always-predict-success; regimen=most-frequent-regimen-per-profile',
            'regimen_mode': 'diagnostic predictor: most-frequent regimen among retrieved neighbors (ignores outcome), to separate objective mismatch from weak retrieval'
        }

    if cryptic_results is not None:
        data['cryptic_classification'] = cryptic_results

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    return data


def print_expert_summary(expert):
    print("\nExpert System (NL to Cypher)")
    print(f"  Model:    {expert['model']}")
    print(f"  Accuracy: {expert['overall']['rate']:.1%} (n={expert['overall']['n']}, execution match)")
    print("  By category:")
    for name, stats in expert['by_category'].items():
        print(f"    {name:15s}: {stats['rate']:.1%} (n={stats['n']})")


def print_cbr_summary(cbr):
    reg = cbr['regimen_accuracy']
    out = cbr['outcome_accuracy']
    mode = cbr['regimen_mode_accuracy']

    print(f"\nCBR ({cbr['k']}-fold CV)")
    print(f"  Regimen:       {reg['mean']:.1%} [{reg['ci_lower']:.1%}, {reg['ci_upper']:.1%}]")
    print(f"  Regimen (mode): {mode['mean']:.1%} [{mode['ci_lower']:.1%}, {mode['ci_upper']:.1%}]")
    print(f"  Outcome:       {out['mean']:.1%} [{out['ci_lower']:.1%}, {out['ci_upper']:.1%}]")
    cal = cbr['calibration']
    print(f"  ECE:   {cal['ece']:.4f} raw, {cal['ece_temperature_scaled']:.4f} after temperature "
          f"scaling (rejected, T={cal['temperature_mean']})")
    print(f"  Brier: {cal['brier']:.4f}")
    base = cbr['baseline']
    print(f"  Baseline: regimen {base['regimen']:.1%}, outcome {base['outcome']:.1%}")

    print("\nCBR by Profile")
    for profile in ['Susceptible', 'MonoResistant', 'PolyResistant', 'MDR', 'PreXDR', 'XDR']:
        if profile in cbr['by_profile']:
            p = cbr['by_profile'][profile]
            print(f"  {profile:12s}: {p['accuracy']:.1%} (n={p['n']})")


def print_summary(data):
    print_expert_summary(data['expert_system'])
    print_cbr_summary(data['cbr'])


# CRYPTIC CLASSIFICATION VALIDATION

# Validates the rule engine's resistance classification against measured CRyPTIC
# phenotypes, with the WHO catalog as the reference. Database-free, the engine is
# the deployed code, fed catalog-graded mutations through an in-memory ontology.
# Classes collapse to below-MDR / MDR / PreXDR / XDR; a no genotypic call counts
# as below-MDR, so every sub-unit is scored on all isolates. Heavy imports are
# deferred so importing this module for the expert-system and CBR tests stays light.

RESISTANT_TIERS = ("MDR", "PreXDR", "XDR")
COLLAPSED = ["below-MDR", "MDR", "PreXDR", "XDR"]


def _flat(df, key):
    return df if key in df.columns else df.reset_index()


def collapse_tier(label):
    return label if label in RESISTANT_TIERS else "below-MDR"


def tier_accuracy(truth, prediction):
    """Overall, balanced, and macro-F1 accuracy with per-tier sensitivity and specificity."""
    rates = {t: r for t, r in class_rates(truth, prediction, COLLAPSED).items() if r['n']}
    by_tier = {t: {'accuracy': r['sensitivity'], 'n': r['n']} for t, r in rates.items()}
    return {
        'overall': round(float((prediction == truth).mean()), 3),
        'balanced': balanced_accuracy(rates),
        'macro_f1': macro_f1(rates),
        'by_tier': by_tier,
        'rates': rates,
    }


def confusion(truth, prediction):
    import pandas as pd
    table = pd.crosstab(truth, prediction).reindex(index=COLLAPSED, columns=COLLAPSED, fill_value=0)
    return {t: {c: int(table.loc[t, c]) for c in COLLAPSED} for t in COLLAPSED}


def agreement(truth, engine, catalog):
    """Splits resistant-tier errors into shared (biological ceiling) and engine-only."""
    resistant = truth.isin(RESISTANT_TIERS)
    engine_ok = engine == truth
    catalog_ok = catalog == truth
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
    cases = []
    for isolate in ids:
        records = by_isolate.get(isolate, [])
        cases.append({
            'uniqueid': isolate,
            'truth': truth[isolate],
            'engine': engine[isolate],
            'catalog': catalog[isolate],
            'resistant_drugs': sorted({r['drug'] for r in records}),
            'mutations': [f"{r['gene']}_{r['mutation']}" for r in records],
        })
    return cases


class IsolateOntology:
    """Feeds per-isolate mutations to the rule engine without a database."""

    def __init__(self, mutations):
        self.mutations = mutations

    def patient_strain_mapping(self, strain_id):
        return None

    def strain_mutations_detailed(self, strain_id):
        return self.mutations.get(strain_id, [])


class Evaluator:
    """A validated sub-unit that maps isolates to a collapsed resistance class."""

    name = 'evaluator'

    def predictions(self, isolates):
        raise NotImplementedError


class RuleEngineEvaluator(Evaluator):
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
        eff = _flat(pd.read_parquet(self.effects_path,
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


class CatalogEvaluator(Evaluator):
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
        from feature_engineering import dataset, drug_map, DATA
        data = dataset()
        self.data_dir = DATA
        self.labeled = data.set_index('uniqueid')
        self.drugs = drug_map(DATA / 'DRUG_CODES.csv')
        self.truth = self.labeled['label'].map(collapse_tier)

    def run(self):
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
    return ClassificationValidation().run()


def print_class_scores(scores):
    for name, score in scores.items():
        print(f"\n{name}: overall {score['overall']:.1%}, balanced {score['balanced']:.1%}, "
              f"macro-F1 {score['macro_f1']:.3f}")
        for tier in COLLAPSED:
            if tier in score['rates']:
                r = score['rates'][tier]
                print(f"  {tier:10s}: sens {r['sensitivity']:.1%}  spec {r['specificity']:.1%}  "
                      f"ppv {r['precision']:.1%}  (R={r['n']})")


def print_class_confusion(score):
    print("\nrule_engine confusion (rows truth, cols predicted):")
    table = score['confusion']
    print("            " + "".join(f"{c:>11s}" for c in COLLAPSED))
    for truth in COLLAPSED:
        row = "".join(f"{table[truth][c]:>11d}" for c in COLLAPSED)
        print(f"  {truth:10s}{row}")


def print_class_agreement(agree):
    print("\nengine vs catalog (resistant-truth isolates):")
    print(f"  prediction match (all isolates): {agree['engine_catalog_match']:.1%}")
    print(f"  both wrong (biological ceiling): {agree['both_wrong']}")
    print(f"  engine-only wrong (fixable)    : {agree['engine_only_wrong']}")
    print(f"  catalog-only wrong             : {agree['catalog_only_wrong']}")
    mc = agree['mcnemar']
    print(f"  McNemar: chi2 {mc['chi2']}, p {mc['p_value']:.2e} ({mc['discordant']} discordant)")


def print_classification(summary):
    print("\nCRyPTIC Classification Validation")
    print(f"labeled isolates: {summary['eval_isolates']:,}")
    print_class_scores(summary['scores'])
    print_class_confusion(summary['scores']['rule_engine'])
    print_class_agreement(summary['agreement'])


def setup_ontology():
    from tb_ontology import TBOntology

    print("\nInitializing knowledge graph")
    ontology = TBOntology()
    ontology.clear_database()
    ontology.schema()
    ontology.ontology_classes()

    try:
        ontology.who_mutations()
        ontology.count_who_mutations()
    except Exception as e:
        print(f"WHO data skipped ({e})")

    return ontology


def run_system_validation():
    from nl_interface import NLInterface
    from cbr_cases import generate_cases

    ontology = setup_ontology()
    nl_interface = NLInterface(ontology)
    kg_results = validate_expert_system(nl_interface)
    ontology.close()

    print("\nGenerating 1000 synthetic cases")
    cases = generate_cases(1000, seed=42)
    cbr_results = validate_cbr(cases, k=5)
    return kg_results, cbr_results


def main():
    print("Validation Summary")
    load_dotenv()

    expert = cbr = cryptic = None

    try:
        expert, cbr = run_system_validation()
    except Exception as e:
        print(f"\nSystem validation skipped, graph or API unavailable ({e})")

    try:
        cryptic = validate_classification()
        print_classification(cryptic)
    except Exception as e:
        print(f"\nClassification validation skipped ({e})")

    data = report_file(expert, cbr, cryptic)
    if expert and cbr:
        print_summary(data)
    print("\nResults saved to validation_results.json")


if __name__ == "__main__":
    main()