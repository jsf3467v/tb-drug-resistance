"""
Enhanced Validation Framework for TB Hybrid AI System

Features:
- K-fold cross-validation for CBR
- Bootstrap confidence intervals
- Expected Calibration Error (ECE)
- Edge case testing for expert system
"""

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

load_dotenv()


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
    """Most frequent regimen among the retrieved neighbors, ties broken by name
    for determinism. Unlike recommend()'s outcome-ranked choice, this ignores
    success rate and just takes the modal neighbor regimen, so it tracks the
    per-profile mode that the majority baseline also predicts. Reported as a
    diagnostic and it separates 'the objective is mismatched' from 'retrieval is
    weak' for the below-baseline regimen result."""
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
    """Per-fold temperature scaling of the predicted success probability, fit on the
    other folds (leak-free). Retained only to document that post-hoc scaling was
    tested and rejected: it raises ECE here, so the raw probability is reported.
    Returns the fold temperatures and the pooled scaled (probability, actual) pairs.
    """
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
            'ece_temperature_scaled': expected_calibration_error(calibrated),
            'temperature_mean': round(float(np.mean(temperatures)), 3),
            'temperature_per_fold': temperatures,
            'reliability': reliability_diagram(predictions)
        }
    }


def validate_cbr(cases, k=5, seed=42):
    print(f"\nCBR {k}-Fold Cross-Validation")
    print("-" * 50)

    splits = stratified_folds(cases, k, random.Random(seed))
    all_results = []

    for i, (train, test) in enumerate(splits):
        results = run_fold(train, test, i + 1)
        all_results.append(results)

    return aggregate_cbr_folds(all_results, k, seed)


# EXPERT SYSTEM VALIDATION

STANDARD_QUERIES = [
    {'id': 1, 'question': 'What mutations cause rifampin resistance?',
     'category': 'drug_resistance', 'expected_contains': ['rpoB', 'S450L'],
     'min_results': 5, 'requires_rules': False},
    {'id': 2, 'question': 'What drugs should be excluded for strain TB001?',
     'category': 'treatment', 'expected_contains': ['rifampin', 'isoniazid'],
     'min_results': 2, 'requires_rules': True, 'expected_classification': 'MDR',
     'expected_rules': ['RC001']},
    {'id': 3, 'question': 'Show all MDR strains',
     'category': 'profile_search', 'expected_contains': ['TB001', 'TB006'],
     'min_results': 7, 'requires_rules': False},
    {'id': 4, 'question': 'What is the resistance classification of strain TB011?',
     'category': 'classification', 'expected_contains': [], 'min_results': 1,
     'requires_rules': True, 'expected_classification': 'XDR',
     'expected_rules': ['RC001', 'RC002']},
    {'id': 5, 'question': 'Which genes have the most resistance mutations?',
     'category': 'statistical', 'expected_contains': ['rpoB', 'katG'],
     'min_results': 3, 'requires_rules': False}
]

EDGE_CASE_QUERIES = [
    {'id': 101, 'question': 'What is the classification of strain TB002?',
     'category': 'edge_case', 'edge_type': 'mono_resistance',
     'expected_contains': ['MonoResistant'], 'min_results': 1,
     'requires_rules': True, 'expected_classification': 'MonoResistant'},
    {'id': 102, 'question': 'Show rifampicin resistant strains',
     'category': 'edge_case', 'edge_type': 'spelling_variation',
     'expected_contains': ['rpoB'], 'min_results': 5, 'requires_rules': False},
    {'id': 103, 'question': 'What about TB003?',
     'category': 'edge_case', 'edge_type': 'ambiguous',
     # Ambiguous NL: the model's interpretation legitimately varies run to
     # run, so this case is measured and reported, not pass/fail gated.
     'expected_contains': ['TB003'], 'min_results': 1, 'requires_rules': False,
     'scored': False},
    {'id': 104, 'question': 'Show MDR strains from India with gyrA mutations',
     'category': 'edge_case', 'edge_type': 'multi_condition',
     # The seed graph holds no MDR strain from India carrying a gyrA
     # mutation, so the correct answer to this conjunction is empty. The
     # test checks the model builds the constraint correctly, not that a
     # strain is manufactured to satisfy it.
     'expected_contains': [], 'min_results': 0, 'requires_rules': False},
    {'id': 105, 'question': 'Which strains do NOT have rifampin resistance?',
     'category': 'edge_case', 'edge_type': 'negation',
     # Non-rifampin strains are the Mono/Poly/Susceptible classes, so the
     # real test is that rifampin-resistant strains are excluded. These ids
     # are rifampin-resistant (MDR/PreXDR/XDR) and must not appear.
     'expected_contains': [], 'expected_absent': ['TB001', 'TB003', 'TB011'],
     'min_results': 1, 'requires_rules': False},
    {'id': 106, 'question': 'Classify strain TB015 resistance profile',
     'category': 'edge_case', 'edge_type': 'borderline',
     'expected_contains': ['PreXDR'], 'min_results': 1,
     'requires_rules': True, 'expected_classification': 'PreXDR'},
    {'id': 107, 'question': 'What resistance does mutation rpoB_X999Y confer?',
     'category': 'edge_case', 'edge_type': 'unknown_mutation',
     'expected_contains': [], 'min_results': 0, 'requires_rules': False},
    {'id': 108, 'question': 'Show XDR strains from Antarctica',
     'category': 'edge_case', 'edge_type': 'no_results',
     'expected_contains': [], 'min_results': 0, 'requires_rules': False}
]


def standard_queries():
    return STANDARD_QUERIES


def edge_case_queries():
    return EDGE_CASE_QUERIES


def evaluate_query(test, nl_interface):
    start = time.time()

    try:
        cypher = nl_interface.generate_cypher(test['question'])
        is_valid, _ = nl_interface.validate_cypher(cypher)
        no_results_expected = test.get('min_results', 1) == 0

        if not is_valid or "UNANSWERABLE" in cypher:
            return _rejected_result(test, no_results_expected, start)

        results = nl_interface.execute_query(cypher)
        return _scored_result(test, results, no_results_expected, nl_interface, start)

    except Exception as e:
        return query_result(test, False, 0.0, 0, time.time() - start, [], str(e))


def _rejected_result(test, no_results_expected, start):
    # An explicit unanswerable/invalid query is correct only when no rows are
    # expected; otherwise the query genuinely failed.
    return query_result(test, no_results_expected,
                        1.0 if no_results_expected else 0.0,
                        0, time.time() - start, [],
                        None if no_results_expected else "Query failed")


def _scored_result(test, results, no_results_expected, nl_interface, start):
    elapsed = time.time() - start

    if no_results_expected:
        # A valid query whose correct answer is empty passes; one that
        # wrongly returns rows fails.
        empty = len(results) == 0
        return query_result(test, empty, 1.0 if empty else 0.0,
                            len(results), elapsed, [],
                            None if empty else f"Expected none, got {len(results)}")

    if len(results) < test.get('min_results', 1):
        return query_result(test, False, 0.0, len(results), elapsed, [],
                            f"Insufficient: {len(results)}")

    passed, confidence, rules = check_query_results(test, results, nl_interface)
    return query_result(test, passed, confidence, len(results), elapsed, rules)


def query_check_type(test):
    """The verification a query's pass actually rests on, so the headline rate
    can be read by strength of evidence rather than as one flat number:
      classification - exact match against the rule engine's predicted profile
      empty_expected - passes only if the query correctly returns no rows
      absence        - forbidden ids must not appear (negation)
      content_match  - substring containment of expected tokens (weakest)"""
    if test.get('expected_classification'):
        return 'classification'
    if test.get('min_results', 1) == 0:
        return 'empty_expected'
    if test.get('expected_absent'):
        return 'absence'
    return 'content_match'


def query_result(test, passed, confidence, count, elapsed, rules, failure=None):
    result = {
        'id': test['id'],
        'category': test['category'],
        'edge_type': test.get('edge_type'),
        'check_type': query_check_type(test),
        'passed': passed,
        'confidence': confidence,
        'result_count': count,
        'time_ms': round(elapsed * 1000, 1),
        'rules_fired': rules,
        'scored': test.get('scored', True)
    }
    if failure:
        result['failure'] = failure
    return result


def check_query_results(test, results, nl_interface):
    results_str = str(results).lower()

    if _absent_violation(test, results_str):
        return False, 0.0, []

    confidence = _content_confidence(test, results, results_str)
    passed = confidence >= 0.5

    if test.get('requires_rules'):
        return _rule_check(test, results, nl_interface, passed, confidence)

    return passed, confidence, []


def _absent_violation(test, results_str):
    absent = test.get('expected_absent')
    return bool(absent) and any(str(item).lower() in results_str for item in absent)


def _content_confidence(test, results, results_str):
    expected = test.get('expected_contains')
    if expected:
        matched = sum(1 for item in expected if str(item).lower() in results_str)
        return matched / len(expected)
    return 1.0 if results else 0.0


def _rule_check(test, results, nl_interface, passed, confidence):
    rules = []
    nl_interface.last_question = test['question']
    qtype = nl_interface.needs_rules(test['question'])

    if qtype:
        output = nl_interface.rule_recommend(results, qtype)
        if output:
            rules = output.get('rules_fired', [])
            classifications = output['recommendations'].get('classifications', [])

            if test.get('expected_classification') and classifications:
                predicted = classifications[0]['type']
                passed = predicted == test['expected_classification']
                confidence = 1.0 if passed else 0.5

    return passed, confidence, rules


def validate_expert_system(nl_interface):
    print("\nExpert System Validation")
    print("-" * 50)

    tests = standard_queries() + edge_case_queries()
    results = []

    for test in tests:
        result = evaluate_query(test, nl_interface)
        results.append(result)
        if not result.get('scored', True):
            status = "MEASURED"
        else:
            status = "PASS" if result['passed'] else "FAIL"
        edge = f" [{result['edge_type']}]" if result.get('edge_type') else ""
        print(f"  {test['id']:3d}: {status}{edge}")

    return aggregate_expert_results(results)


def pass_rate_breakdown(results, key):
    groups = defaultdict(lambda: {'passed': 0, 'total': 0})
    for r in results:
        g = groups[r.get(key, 'unknown')]
        g['total'] += 1
        g['passed'] += bool(r['passed'])
    return {k: {'rate': round(s['passed'] / s['total'], 3), 'n': s['total']}
            for k, s in groups.items()}


def aggregate_expert_results(results):
    scored = [r for r in results if r.get('scored', True)]
    measured = [r for r in results if not r.get('scored', True)]
    standard = [r for r in scored if r['category'] != 'edge_case']
    edge = [r for r in scored if r['category'] == 'edge_case']

    def rate(lst):
        return round(sum(r['passed'] for r in lst) / len(lst), 3) if lst else 0.0

    return {
        'overall': {'rate': rate(scored), 'n': len(scored)},
        'standard': {'rate': rate(standard), 'n': len(standard)},
        'edge_cases': {'rate': rate(edge), 'n': len(edge)},
        'by_edge_type': pass_rate_breakdown(edge, 'edge_type'),
        'by_check_type': pass_rate_breakdown(scored, 'check_type'),
        'measured': [{'id': r['id'], 'edge_type': r.get('edge_type'),
                      'passed': r['passed'], 'result_count': r['result_count']}
                     for r in measured]
    }



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


def print_expert_summary(kg):
    print("\nExpert System")
    print(f"  Standard:   {kg['standard']['rate']:.1%} (n={kg['standard']['n']})")
    print(f"  Edge Cases: {kg['edge_cases']['rate']:.1%} (n={kg['edge_cases']['n']})")
    print(f"  Overall:    {kg['overall']['rate']:.1%}")
    print("  By check type:")
    for check, s in kg['by_check_type'].items():
        print(f"    {check:15s}: {s['rate']:.1%} (n={s['n']})")


def print_cbr_summary(cbr):
    reg = cbr['regimen_accuracy']
    out = cbr['outcome_accuracy']
    mode = cbr['regimen_mode_accuracy']

    print(f"\nCBR ({cbr['k']}-fold CV)")
    print(f"  Regimen:       {reg['mean']:.1%} [{reg['ci_lower']:.1%}, {reg['ci_upper']:.1%}]")
    print(f"  Regimen (mode): {mode['mean']:.1%} [{mode['ci_lower']:.1%}, {mode['ci_upper']:.1%}]")
    print(f"  Outcome:       {out['mean']:.1%} [{out['ci_lower']:.1%}, {out['ci_upper']:.1%}]")
    cal = cbr['calibration']
    print(f"  ECE:      {cal['ece']:.4f} (raw); temp-scaling rejected -> {cal['ece_temperature_scaled']:.4f} (T={cal['temperature_mean']})")
    base = cbr['baseline']
    print(f"  Baseline: regimen {base['regimen']:.1%}, outcome {base['outcome']:.1%}")

    print("\nCBR by Profile")
    for profile in ['Susceptible', 'MonoResistant', 'PolyResistant', 'MDR', 'PreXDR', 'XDR']:
        if profile in cbr['by_profile']:
            p = cbr['by_profile'][profile]
            print(f"  {profile:12s}: {p['accuracy']:.1%} (n={p['n']})")


def print_edge_summary(kg):
    print("\nEdge Case Breakdown")
    for etype, stats in kg['by_edge_type'].items():
        print(f"  {etype:20s}: {stats['rate']:.1%} (n={stats['n']})")


def print_summary(data):

    print("VALIDATION SUMMARY")
    print("-" * 60)

    print_expert_summary(data['expert_system'])
    print_cbr_summary(data['cbr'])
    print_edge_summary(data['expert_system'])

    print("\n" + "=" * 60)


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
    correct = prediction == truth
    by_tier = {}
    for tier in COLLAPSED:
        mask = truth == tier
        n = int(mask.sum())
        if n:
            by_tier[tier] = {'accuracy': round(float(correct[mask].mean()), 3), 'n': n}
    return {'overall': round(float(correct.mean()), 3), 'by_tier': by_tier}


def confusion(truth, prediction):
    import pandas as pd
    table = pd.crosstab(truth, prediction).reindex(index=COLLAPSED, columns=COLLAPSED, fill_value=0)
    return {t: {c: int(table.loc[t, c]) for c in COLLAPSED} for t in COLLAPSED}


def agreement(truth, engine, catalog):
    """Splits resistant-tier errors into shared (biological ceiling) and engine-only."""
    resistant = truth.isin(RESISTANT_TIERS)
    engine_ok = engine == truth
    catalog_ok = catalog == truth
    return {
        'engine_catalog_match': round(float((engine == catalog).mean()), 3),
        'resistant_isolates': int(resistant.sum()),
        'both_correct': int((resistant & engine_ok & catalog_ok).sum()),
        'both_wrong': int((resistant & ~engine_ok & ~catalog_ok).sum()),
        'engine_only_wrong': int((resistant & ~engine_ok & catalog_ok).sum()),
        'catalog_only_wrong': int((resistant & engine_ok & ~catalog_ok).sum()),
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
    """A validated sub-unit: maps test isolates to a collapsed resistance class."""

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
    """Runs every classification sub-unit on the shared test split and scores them together."""

    def __init__(self):
        from feature_engineering import dataset, drug_map, DATA
        data = dataset()
        self.data_dir = DATA
        self.test = data[data['split'] == 'test'].set_index('uniqueid')
        self.drugs = drug_map(DATA / 'DRUG_CODES.csv')
        self.truth = self.test['label'].map(collapse_tier)

    def run(self):
        engine_eval = RuleEngineEvaluator(self.data_dir / 'EFFECTS.parquet', self.drugs)
        catalog_eval = CatalogEvaluator(self.test['catalog'])
        preds = {
            engine_eval.name: engine_eval.predictions(self.test.index),
            catalog_eval.name: catalog_eval.predictions(self.test.index),
        }
        scores = {name: {**tier_accuracy(self.truth, p), 'confusion': confusion(self.truth, p)}
                  for name, p in preds.items()}
        return {
            'test_isolates': len(self.test),
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
        print(f"\n{name}: overall {score['overall']:.1%}")
        for tier in COLLAPSED:
            if tier in score['by_tier']:
                stat = score['by_tier'][tier]
                print(f"  {tier:10s}: {stat['accuracy']:.1%} (n={stat['n']})")


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


def print_classification(summary):
    print("\n" + "=" * 60)
    print("CRYPTIC CLASSIFICATION VALIDATION")
    print("=" * 60)
    print(f"test isolates: {summary['test_isolates']:,}")
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
        print(f"WHO data skipped: {e}")

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
    print("-" * 60)

    expert = cbr = cryptic = None

    try:
        expert, cbr = run_system_validation()
    except Exception as e:
        print(f"\nSystem validation skipped - graph/API unavailable: {e}")

    try:
        cryptic = validate_classification()
        print_classification(cryptic)
    except Exception as e:
        print(f"\nClassification validation skipped: {e}")

    data = report_file(expert, cbr, cryptic)
    if expert and cbr:
        print_summary(data)
    print("\nResults saved: validation_results.json")


if __name__ == "__main__":
    main()