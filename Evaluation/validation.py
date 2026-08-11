"""Validation across three arms: case-based cross-validation with calibration,
query translation against gold queries, and CRyPTIC classification."""

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "SRC"))

from calibration import (
    fit_platt,
    fit_temperature,
    platt_confidence,
    scaled_confidence,
)
from cbr_cases import marginal_success, regimen_ceiling
from config import EXAMPLES, RESISTANT_TIERS, SCHEMA
from metrics import (
    RuleEngineEvaluator,
    auc,
    balanced_accuracy,
    brier,
    brier_constant,
    class_rates,
    expected_calibration_error,
    macro_f1,
    mcnemar,
    reliability_diagram,
    safe_ratio,
    wilson_interval,
)

SEED = 42
K_FOLDS = 5
N_CASES = 1000
BOOTSTRAP_SAMPLES = 1000
CBR_NEIGHBORS = 10

RESULTS = EVAL_DIR / "validation_results.json"
EXPERT_CHECKPOINT = EVAL_DIR / ".expert_checkpoint.json"


def prompt_tag(model):
    """Digest of everything shaping a generated query, so editing the schema or the
    examples discards the old results rather than resuming on top of them."""
    return hashlib.sha256((model + SCHEMA + EXAMPLES).encode()).hexdigest()[:12]


def expert_checkpoint(tag, results=None):
    """Read or write the expert journal. Cleanly scored queries only, so an
    infrastructure failure is retried. A file left on disk means the run did not
    finish."""
    if results is not None:
        EXPERT_CHECKPOINT.write_text(json.dumps({"tag": tag, "results": results}))
        return results
    if not EXPERT_CHECKPOINT.exists():
        return None
    saved = json.loads(EXPERT_CHECKPOINT.read_text())
    return saved["results"] if saved.get("tag") == tag else None


def expert_checkpoint_clear():
    EXPERT_CHECKPOINT.unlink(missing_ok=True)


def bootstrap_ci(values, rng, confidence=0.95):
    if not values:
        return 0.0, 0.0, 0.0

    arr = np.asarray(values, dtype=float)
    draws = rng.choice(arr, size=(BOOTSTRAP_SAMPLES, arr.size), replace=True).mean(axis=1)

    alpha = 1.0 - confidence
    lower, upper = np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(arr.mean()), float(lower), float(upper)


def rate(results, key):
    return sum(r[key] for r in results) / len(results) if results else 0.0


def accuracy_with_ci(all_results, key, rng):
    """Pooled accuracy with a bootstrap interval over cases, plus the fold spread.
    The cases carry the interval; the spread answers how far the split moves it."""
    per_case = [float(r[key]) for fold in all_results for r in fold]
    fold_rates = [rate(fold, key) for fold in all_results]
    center, lower, upper = bootstrap_ci(per_case, rng)
    return {
        'mean': round(center, 3),
        'fold_std': round(float(np.std(fold_rates, ddof=1)), 3) if len(fold_rates) > 1 else 0.0,
        'ci_lower': round(lower, 3),
        'ci_upper': round(upper, 3),
    }


def stratified_folds(cases, k, rng):
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


def neighbor_regimen_mode(similar_cases, applicable):
    """Modal applicable regimen among the neighbors, ties broken by name. Drawn
    from the recommender's own set, so the two differ only in ranking."""
    counts = Counter(case['regimen'] for _, case in similar_cases
                     if case['regimen'] in applicable)
    if not counts:
        return None
    top = max(counts.values())
    return min(regimen for regimen, c in counts.items() if c == top)


def evaluate_cbr_case(test_case, engine):
    analysis = engine.recommend(cbr_query(test_case), k=CBR_NEIGHBORS)

    recs = analysis['recommendations']
    predicted_regimen = recs[0]['regimen'] if recs else None
    applicable = engine.profile_regimens.get(test_case['profile'], set())
    mode_regimen = neighbor_regimen_mode(analysis['similar_cases'], applicable)
    actual_success = test_case['outcome'] == 'success'

    return {
        'regimen_correct': predicted_regimen == test_case['regimen'],
        'regimen_mode_correct': mode_regimen == test_case['regimen'],
        'outcome_correct': (analysis['success_rate'] >= 0.5) == actual_success,
        'abstained': predicted_regimen is None,
        'success_prob': analysis['success_rate'],
        'profile': test_case['profile'],
        'actual_success': actual_success,
    }


def fold_scores(train, test):
    from cbr_engine import CBREngine

    engine = CBREngine(train)
    return [evaluate_cbr_case(case, engine) for case in test]


def profile_accuracy(flat_results):
    """Regimen accuracy per profile, least to most severe."""
    from config import SEVERITY

    by_profile = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in flat_results:
        by_profile[r['profile']]['total'] += 1
        by_profile[r['profile']]['correct'] += bool(r['regimen_correct'])

    order = sorted(by_profile, key=lambda p: SEVERITY.index(p) if p in SEVERITY else len(SEVERITY))
    return {p: {'accuracy': round(by_profile[p]['correct'] / by_profile[p]['total'], 3),
                'n': by_profile[p]['total']} for p in order}


def baseline_accuracy(train, test):
    """Floors fit on train and scored on test, outcome by majority class and
    regimen by profile mode. regimen_modes is shared with the layer it bounds."""
    from cbr_engine import regimen_modes

    if not train or not test:
        return {'outcome': 0.0, 'regimen': 0.0}
    modes = regimen_modes(train)
    predict_success = sum(c['outcome'] == 'success' for c in train) / len(train) >= 0.5
    outcome = sum((c['outcome'] == 'success') == predict_success for c in test) / len(test)
    regimen = sum(modes.get(c['profile']) == c['regimen'] for c in test) / len(test)
    return {'outcome': round(outcome, 3), 'regimen': round(regimen, 3)}


def baseline_means(baselines):
    """Mean of each baseline across folds."""
    if not baselines:
        return {'outcome': 0.0, 'regimen': 0.0}
    return {key: round(float(np.mean([b[key] for b in baselines])), 3)
            for key in baselines[0]}


def fold_calibrations(all_results):
    """Both scalings, fit on the other folds and applied to the held-out one.
    Temperature is kept because its failure here is a result, not an omission."""
    fold_preds = [[(r['success_prob'], 1.0 if r['actual_success'] else 0.0) for r in fold]
                  for fold in all_results]
    temperatures, platts, tempered, platted = [], [], [], []

    for i, preds in enumerate(fold_preds):
        other = [p for j, fold in enumerate(fold_preds) if j != i for p in fold]
        confidence = [c for c, _ in other]
        labels = [y for _, y in other]
        held = np.array([c for c, _ in preds])

        temperature = fit_temperature(confidence, labels)
        slope, intercept = fit_platt(confidence, labels)
        temperatures.append(round(temperature, 3))
        platts.append((round(slope, 3), round(intercept, 3)))

        tempered.extend((float(s), bool(y)) for s, (_, y)
                        in zip(scaled_confidence(held, temperature), preds, strict=True))
        platted.extend((float(s), bool(y)) for s, (_, y)
                       in zip(platt_confidence(held, slope, intercept), preds, strict=True))

    return temperatures, platts, tempered, platted


def outcome_ceiling(cases):
    """Best outcome scores the retrieved features reach, by the same functions."""
    if not cases:
        return {'auc': 0.0, 'brier': 0.0, 'accuracy': 0.0}
    predictions = [(marginal_success(c), c['outcome'] == 'success') for c in cases]
    hits = sum((p >= 0.5) == y for p, y in predictions)
    return {'auc': auc(predictions), 'brier': brier(predictions),
            'accuracy': round(hits / len(cases), 3)}


def calibration_summary(all_results, predictions, cases):
    """Raw and rescaled calibration of the reported success probability."""
    temperatures, platts, tempered, platted = fold_calibrations(all_results)
    return {
        'ceiling': outcome_ceiling(cases),
        'ece': expected_calibration_error(predictions),
        'brier': brier(predictions),
        'brier_constant': brier_constant(predictions),
        'auc': auc(predictions),
        'ece_temperature_scaled': expected_calibration_error(tempered),
        'ece_platt_scaled': expected_calibration_error(platted),
        'brier_platt_scaled': brier(platted),
        'temperature_mean': round(float(np.mean(temperatures)), 3),
        'temperature_per_fold': temperatures,
        'platt_per_fold': platts,
        'platt_note': 'the slope shrinks the score toward the base rate, so most of '
                      'the lower scaled ECE is shrinkage rather than signal. Ranking '
                      'is unchanged, so auc against the ceiling is the figure to read',
        'reliability': reliability_diagram(predictions),
        'reliability_platt': reliability_diagram(platted),
    }


def aggregate_cbr_folds(all_results, baselines, cases, k, seed=SEED):
    flat = [r for fold in all_results for r in fold]
    rng = np.random.default_rng(seed)
    predictions = [(r['success_prob'], r['actual_success']) for r in flat]

    return {
        'k': k,
        'total_cases': len(flat),
        'regimen_accuracy': accuracy_with_ci(all_results, 'regimen_correct', rng),
        'outcome_accuracy': accuracy_with_ci(all_results, 'outcome_correct', rng),
        'regimen_mode_accuracy': accuracy_with_ci(all_results, 'regimen_mode_correct', rng),
        'by_profile': profile_accuracy(flat),
        'abstentions': sum(r['abstained'] for r in flat),
        'baseline': baseline_means(baselines),
        'ceiling': regimen_ceiling(),
        'calibration': calibration_summary(all_results, predictions, cases),
    }


def validate_cbr(cases, k=K_FOLDS, seed=SEED):
    print(f"\nCBR {k}-fold cross-validation")
    splits = stratified_folds(cases, k, random.Random(seed))
    folds = [fold_scores(train, test) for train, test in splits]
    baselines = [baseline_accuracy(train, test) for train, test in splits]
    return aggregate_cbr_folds(folds, baselines, cases, k, seed)


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


def row_values(row):
    """Value multiset of one row, free of column order and name. Repeats carry an
    occurrence number, so a doubled gold value needs a doubled produced value."""
    seen = Counter()
    tagged = set()
    for value in row.values():
        text = json.dumps(value, sort_keys=True, default=str)
        tagged.add((text, seen[text]))
        seen[text] += 1
    return frozenset(tagged)


def covers(gold, produced):
    """True when each gold row sits inside a distinct produced row. Solved as a
    matching, since greedy assignment strands a later row that would pair up."""
    wanted = [row_values(r) for r in gold]
    pool = [row_values(r) for r in produced]
    if not wanted:
        return True
    if Counter(wanted) == Counter(pool):
        return True

    holders = defaultdict(set)
    for i, have in enumerate(pool):
        for value in have:
            holders[value].add(i)

    rows, cols = [], []
    for i, want in enumerate(wanted):
        fits = set.intersection(*(holders[v] for v in want)) if want else set(range(len(pool)))
        rows += [i] * len(fits)
        cols += fits
    graph = csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)),
                       shape=(len(wanted), len(pool)))
    return bool((maximum_bipartite_matching(graph, perm_type='column') >= 0).all())


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

    expected = nl_interface.execute_query(item['gold'])
    try:
        produced = nl_interface.execute_query(cypher)
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
    """Pass rate with a Wilson interval, since eleven queries carry no more. A
    query that errored on the key or the connection is unscored, not wrong."""
    scored = [r for r in results if not r.get('errored')]
    total = len(scored)
    hits = sum(r['passed'] for r in scored)
    lower, upper = wilson_interval(hits, total)
    return {
        'model': model,
        'method': 'execution match of generated Cypher against a gold query. An '
                  'unanswerable question scores on rejection, whatever caused it',
        'overall': {'rate': round(hits / total, 3) if total else 0.0, 'n': total,
                    'ci_lower': lower, 'ci_upper': upper},
        'by_category': category_rates(scored),
        'errored': [r['id'] for r in results if r.get('errored')],
        'failures': [r for r in scored if not r['passed']],
    }


def validate_expert_system(nl_interface, resume=False):
    from nl_interface import MODEL

    print("\nExpert system validation")
    tag = prompt_tag(MODEL)
    results = (expert_checkpoint(tag) if resume else None) or []
    done = {r['id'] for r in results}
    errored = False

    for item in EXPERT_QUERIES:
        if item['id'] in done:
            continue
        try:
            result = evaluate_query(item, nl_interface)
        except Exception as exc:
            result = query_result(item, False, 0, time.perf_counter(), str(exc))
            result['errored'] = True
            errored = True
        results.append(result)
        expert_checkpoint(tag, [r for r in results if not r.get('errored')])
        state = 'ERROR' if result.get('errored') else ('PASS' if result['passed'] else 'FAIL')
        print(f"  {item['id']:>3} {state:5s} {item['category']}")

    if not errored:
        expert_checkpoint_clear()
    return expert_accuracy(results, MODEL)


# Describes the CBR arm. The cbr key is filled in at write time from the fold count.
METHODOLOGY = {
    'confidence_intervals': f'95% bootstrap (n={BOOTSTRAP_SAMPLES}) over the pooled '
                            f'out-of-fold cases. fold_std is the spread of the per-fold '
                            f'rates and is reported separately, not as the interval',
    'calibration': 'ECE of the predicted success probability against the outcome. The '
                   'probability is Laplace-smoothed so the logit both scalings fit stays '
                   'finite, and each scaling is fit per fold on the other folds.',
    'baseline': 'outcome=majority outcome class, regimen=most-frequent-regimen-per-profile, '
                'fit per training fold and scored on the held-out fold',
    'regimen_mode': 'diagnostic predictor, the most frequent regimen among retrieved '
                    'neighbors, ignoring outcome, to separate objective mismatch from '
                    'weak retrieval',
}


def report_file(expert=None, cbr=None, cryptic=None):
    """Merges into any existing report, so a skipped arm keeps its last result."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    arms = {'expert_system': expert, 'cbr': cbr, 'cryptic_classification': cryptic}
    fresh = {name: result for name, result in arms.items() if result is not None}

    data.update(fresh)
    data['timestamp'] = datetime.now().isoformat()
    data['arms_this_run'] = sorted(fresh)

    if cbr is not None:
        data['methodology'] = dict(METHODOLOGY, cbr=f"{cbr['k']}-fold stratified cross-validation")

    RESULTS.write_text(json.dumps(data, indent=2))
    return data


def rows(title, pairs, indent=2):
    """Aligned label and value block, column width taken from the labels."""
    print(f"\n{title}")
    width = max(len(label) for label in pairs)
    pad = " " * indent
    for label, value in pairs.items():
        print(f"{pad}{label:{width}s}  {value}")


def interval(score):
    return f"{score['mean']:.1%} [{score['ci_lower']:.1%}, {score['ci_upper']:.1%}]"


def print_expert_summary(expert):
    overall = expert['overall']
    missed = {n: c for n, c in expert['by_category'].items() if c['rate'] < 1.0}
    clean = len(expert['by_category']) - len(missed)

    rows("Expert system, natural language to Cypher", {
        'model': expert['model'],
        'accuracy': f"{overall['rate']:.1%} [{overall['ci_lower']:.1%}, "
                    f"{overall['ci_upper']:.1%}] (n={overall['n']}, execution match)",
        **{name: f"{c['rate']:.1%} (n={c['n']})" for name, c in missed.items()},
        **({'other': f"{clean} categories at 100.0%"} if clean else {}),
    })


def print_cbr_summary(cbr):
    cal, base = cbr['calibration'], cbr['baseline']

    rows(f"CBR, {cbr['k']}-fold cross-validation", {
        'regimen': interval(cbr['regimen_accuracy']),
        'regimen mode': interval(cbr['regimen_mode_accuracy']),
        'outcome': interval(cbr['outcome_accuracy']),
        'ECE': f"{cal['ece']:.4f} raw, {cal['ece_platt_scaled']:.4f} platt scaled, "
               f"{cal['ece_temperature_scaled']:.4f} temperature scaled "
               f"(no gain, T={cal['temperature_mean']})",
        'Brier': f"{cal['brier']:.4f} raw, {cal['brier_platt_scaled']:.4f} platt scaled, "
                 f"{cal['brier_constant']:.4f} constant at base rate",
        'AUC': f"{cal['auc']:.3f} raw probability, ceiling {cal['ceiling']['auc']:.3f}",
        'baseline': f"regimen {base['regimen']:.1%}, outcome {base['outcome']:.1%}",
        'ceiling': f"regimen {cbr['ceiling']:.1%} from profile alone, "
                   f"outcome {cal['ceiling']['accuracy']:.1%} from the retrieved features",
    })
    rows("CBR by profile",
         {profile: f"{p['accuracy']:.1%} (n={p['n']})"
          for profile, p in cbr['by_profile'].items()})


def print_summary(data):
    print_expert_summary(data['expert_system'])
    print_cbr_summary(data['cbr'])


# CRYPTIC CLASSIFICATION VALIDATION
# The rule engine's resistance classification against measured CRyPTIC phenotypes,
# with the WHO catalog as reference. Database-free. Classes collapse to
# below-MDR / MDR / PreXDR / XDR, and no genotypic call counts as below-MDR.
# Heavy imports stay deferred so the expert-system and CBR paths load light.

COLLAPSED = ["below-MDR", *RESISTANT_TIERS]


def collapse_tier(label):
    return label if label in RESISTANT_TIERS else "below-MDR"


def tier_accuracy(truth, prediction):
    """Overall and balanced accuracy with macro-F1. The per-tier figures live in
    rates alone, so sensitivity is not repeated under a second name."""
    rates = {t: r for t, r in class_rates(truth, prediction, COLLAPSED).items() if r['n']}
    overall = float((prediction == truth).mean()) if len(truth) else 0.0
    return {
        'overall': round(overall, 3),
        'balanced': balanced_accuracy(rates),
        'macro_f1': macro_f1(rates),
        'rates': rates,
    }


def confusion(truth, prediction):
    import pandas as pd

    table = pd.crosstab(truth, prediction).reindex(index=COLLAPSED, columns=COLLAPSED,
                                                   fill_value=0)
    return {t: {c: int(table.loc[t, c]) for c in COLLAPSED} for t in COLLAPSED}


def agreement(truth, engine, catalog):
    """Prediction match over all isolates, then resistant-tier errors split into
    shared, engine-only, and catalog-only."""
    resistant = truth.isin(RESISTANT_TIERS)
    engine_ok, catalog_ok = engine == truth, catalog == truth
    engine_only = int((resistant & ~engine_ok & catalog_ok).sum())
    catalog_only = int((resistant & engine_ok & ~catalog_ok).sum())

    return {
        'scheme': 'match spans every isolate, mcnemar counts only resistant truth and '
                  'only where the arms differ in correctness',
        'engine_catalog_match': round(float((engine == catalog).mean()), 3),
        'engine_catalog_disagreements': int((engine != catalog).sum()),
        'resistant_isolates': int(resistant.sum()),
        'both_correct': int((resistant & engine_ok & catalog_ok).sum()),
        'both_wrong': int((resistant & ~engine_ok & ~catalog_ok).sum()),
        'engine_only_wrong': engine_only,
        'catalog_only_wrong': catalog_only,
        'resistant_disagreements': int((resistant & (engine != catalog)).sum()),
        'mcnemar': mcnemar(engine_only, catalog_only),
    }


def sensitivity_ceiling(truth, reached, sequenced):
    """Reachable share of resistant truth per tier and pooled. Unreachable
    isolates split by whether the release sequenced them."""
    resistant = truth.isin(RESISTANT_TIERS).to_numpy()
    tiers = truth.to_numpy()[resistant]
    hit, seq = reached[resistant], sequenced[resistant]
    return {
        'by_tier': {t: round(safe_ratio(int(hit[tiers == t].sum()),
                                        int((tiers == t).sum())), 3)
                    for t in RESISTANT_TIERS},
        'pooled': round(safe_ratio(int(hit.sum()), hit.size), 3),
        'sequenced': round(safe_ratio(int(hit[seq].sum()), int(seq.sum())), 3),
        'unreachable_sequenced': int((~hit & seq).sum()),
        'unreachable_unsequenced': int((~hit & ~seq).sum()),
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


class CatalogEvaluator:
    name = 'who_catalog'

    def __init__(self, catalog):
        self.catalog = catalog

    def predictions(self, isolates):
        profiles = self.catalog.reindex(isolates)
        collapsed = profiles.map(lambda p: collapse_tier(p) if isinstance(p, str) else p)
        return collapsed.fillna('below-MDR')


COVERAGE_NOTE = ('untested drugs read as susceptible, so the full-cohort precision on '
                 'pre-XDR and XDR carries coverage gaps as errors')
GENOTYPE_NOTE = ('an isolate the release did not sequence reaches no rule and scores '
                 'below-MDR for want of input')


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
        covered = self.labeled['second_line_tested'].to_numpy(dtype=bool)
        sequenced = self.labeled['catalog'].notna().to_numpy()
        reached = self.labeled.index.isin(engine_eval.graded()['UNIQUEID'].unique())

        return {
            'eval_isolates': len(self.labeled),
            'scheme': 'below-MDR / MDR / PreXDR / XDR; no genotypic call counts as below-MDR',
            'scores': {name: {**tier_accuracy(self.truth, p),
                              'confusion': confusion(self.truth, p)}
                       for name, p in preds.items()},
            'second_line_covered': self.subset_scores(preds, covered, COVERAGE_NOTE),
            'genotype_covered': self.subset_scores(preds, sequenced, GENOTYPE_NOTE),
            'sensitivity_ceiling': sensitivity_ceiling(self.truth, reached, sequenced),
            'agreement': agreement(self.truth, preds['rule_engine'], preds['who_catalog']),
            'engine_only_cases': diagnose(engine_eval, self.truth,
                                          preds['rule_engine'], preds['who_catalog']),
        }

    def subset_scores(self, preds, mask, note):
        """Tiers rescored on a subset, by the function that scores the full cohort."""
        return {
            'isolates': int(mask.sum()),
            'share': round(float(mask.mean()), 3),
            'note': note,
            'scores': {name: tier_accuracy(self.truth[mask], p[mask])
                       for name, p in preds.items()},
        }


def validate_classification():
    return ClassificationValidation().summary()


def print_class_scores(scores, ceiling):
    """Headline rates per arm, then one row per tier with the reachable share."""
    rows("overall / balanced / macro-F1",
         {name: f"{s['overall']:.1%} / {s['balanced']:.1%} / {s['macro_f1']:.3f}"
          for name, s in scores.items()})
    print(f"\n  {'tier':11s}{'R':>8s}{'sens':>8s}{'spec':>7s}{'ppv':>7s}"
          f"{'F1':>8s}{'catalog sens':>14s}{'ceiling':>9s}")
    for tier in COLLAPSED:
        r = scores['rule_engine']['rates'].get(tier)
        if not r:
            continue
        reach = ceiling['by_tier'].get(tier)
        print(f"  {tier:11s}{r['n']:>8,}{r['sensitivity']:>8.1%}{r['specificity']:>7.1%}"
              f"{r['precision']:>7.1%}{r['f1']:>8.3f}"
              f"{scores['who_catalog']['rates'][tier]['sensitivity']:>14.1%}"
              f"{(f'{reach:.1%}' if reach else '-'):>9s}")


def print_class_confusion(score):
    print("\nrule engine confusion, rows truth and columns predicted")
    table = score['confusion']
    label = max(len(t) for t in COLLAPSED)
    counts = [f"{table[t][c]:,}" for t in COLLAPSED for c in COLLAPSED]
    cell = max(len(v) for v in counts + COLLAPSED) + 2

    print(" " * (label + 2) + "".join(f"{c:>{cell}s}" for c in COLLAPSED))
    for truth in COLLAPSED:
        row = "".join(f"{table[truth][c]:>{cell},}" for c in COLLAPSED)
        print(f"  {truth:{label}s}{row}")


def print_class_agreement(agree):
    mc = agree['mcnemar']
    rows(f"engine vs catalog, prediction match {agree['engine_catalog_match']:.1%} "
         f"over all isolates, {agree['engine_catalog_disagreements']} differ", {
            'resistant truth': f"both wrong {agree['both_wrong']}, "
                                f"engine only {agree['engine_only_wrong']}, "
                                f"catalog only {agree['catalog_only_wrong']}",
             'McNemar': f"chi2 {mc['chi2']}, p {mc['p_value']:.2e}, "
                        f"{mc['discordant']} discordant",
         })


def print_subset_scores(title, subset):
    """Overall, balanced, and macro-F1 per arm on a subset of the cohort."""
    scored = "   ".join(f"{name} {s['overall']:.1%} / {s['balanced']:.1%} / "
                        f"{s['macro_f1']:.3f}" for name, s in subset['scores'].items())
    print(f"  {title:24s}{subset['isolates']:>8,} ({subset['share']:.1%})   {scored}")


def print_ceiling(ceiling):
    """Pooled reachable share and the unreachable split, per tier sits in the table."""
    print(f"\nsensitivity ceiling {ceiling['pooled']:.1%} pooled, "
          f"{ceiling['sequenced']:.1%} on sequenced isolates, unreachable "
          f"{ceiling['unreachable_sequenced']:,} sequenced and "
          f"{ceiling['unreachable_unsequenced']:,} not")


def print_classification(summary):
    print(f"\nCRyPTIC classification, {summary['eval_isolates']:,} labeled isolates")
    print_class_scores(summary['scores'], summary['sensitivity_ceiling'])
    print_ceiling(summary['sensitivity_ceiling'])
    print_class_confusion(summary['scores']['rule_engine'])
    print_class_agreement(summary['agreement'])
    print("\nrescored subsets, overall / balanced / macro-F1")
    print_subset_scores("second-line tested", summary['second_line_covered'])
    print_subset_scores("sequenced", summary['genotype_covered'])


def graph_ontology():
    from tb_ontology import TBOntology

    print("\nRebuilding knowledge graph, the existing contents are cleared")
    ontology = TBOntology()
    ontology.rebuild()
    return ontology


def system_validation(resume=False):
    from cbr_cases import case_base
    from nl_interface import NLInterface

    ontology = graph_ontology()
    try:
        expert = validate_expert_system(NLInterface(ontology), resume)
    finally:
        ontology.close()

    cases = case_base(N_CASES, seed=SEED)
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
    if 'expert_system' in data and 'cbr' in data:
        print_summary(data)
    print(f"\nSaved {RESULTS.name} in {RESULTS.parent.name}")


if __name__ == "__main__":
    main()