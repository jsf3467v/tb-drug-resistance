"""Case-based reasoning over the synthetic case base. A query scores against
every stored case in one vectorized pass, and the engine reports the regimens
the retrieved neighbors received together with how often those regimens
succeeded.
"""

import numpy as np
from cbr_cases import DEFAULT_CASES, DEFAULT_SEED, case_base

FEATURE_ORDER = ['profile', 'previous_treatment', 'hiv_status', 'region', 'age', 'diabetes', 'sex']
FEATURE_WEIGHTS = {
    'profile': 0.32,
    'previous_treatment': 0.20,
    'hiv_status': 0.15,
    'region': 0.12,
    'age': 0.10,
    'diabetes': 0.07,
    'sex': 0.04
}
PROFILE_RANK = {
    'Susceptible': 0, 'MonoResistant': 1, 'PolyResistant': 2,
    'MDR': 3, 'PreXDR': 4, 'XDR': 5
}
PROFILE_SPAN = max(PROFILE_RANK.values())

# Substituted when a case or a query omits a feature.
DEFAULT_PROFILE = 'Susceptible'
DEFAULT_HIV = 'negative'
DEFAULT_REGION = 'global'
DEFAULT_SEX = 'M'
DEFAULT_AGE = 40
DEFAULT_YEAR = 2022

# Age gap at which age similarity reaches zero, and the credit a region
# mismatch keeps.
AGE_SCALE = 50.0
REGION_FLOOR = 0.5

# Bands that label one feature comparison, and how many features are named.
EXACT_MATCH = 0.95
CLOSE_MATCH = 0.70
PARTIAL_MATCH = 0.30
MAX_TOP_MATCHES = 3
MAX_KEY_DIFFERENCES = 2

# Similarity is rounded before thresholding and ranking, so float summation
# order cannot flip the cutoff or reorder ties.
MIN_SIMILARITY = 0.55
DEFAULT_NEIGHBORS = 10
RANK_PRECISION = 12

# Ranking discounts older cases. The floor bounds the discount over a wider
# span of years than the current case base holds.
TEMPORAL_DECAY = 0.10
TEMPORAL_FLOOR = 0.70

# A regimen needs this many supporting neighbors before it is offered.
MIN_EVIDENCE_CASES = 2
MAX_RECOMMENDATIONS = 3
EXPLAINED_CASES = 5
STRONG_EVIDENCE_CASES = 5
STRONG_EVIDENCE_RATE = 0.70
FAIR_EVIDENCE_CASES = 3
FAIR_EVIDENCE_RATE = 0.55

DEFAULT_REGIMENS = {
    'Susceptible': '2HRZE_4HR',
    'MonoResistant': '6REZ_Lfx',
    'PolyResistant': 'Individualized_12mo',
    'MDR': 'BPaLM',
    'PreXDR': 'BPaL',
    'XDR': 'BPaL'
}

# Graph writes.
CASE_BATCH = 100
CASE_LIMIT = 100


class SimilarityCalculator:
    def __init__(self, cases=()):
        self.weights = FEATURE_WEIGHTS
        self.feature_funcs = {
            'profile': self.profile_similarity,
            'previous_treatment': self.previous_tx_similarity,
            'hiv_status': self.hiv_similarity,
            'region': self.region_similarity,
            'age': self.age_similarity,
            'diabetes': self.diabetes_similarity,
            'sex': self.sex_similarity
        }
        self.columns = self.case_columns(cases)

    @staticmethod
    def case_columns(cases):
        """Case base as column arrays, built in a single pass."""
        values = {name: [] for name in FEATURE_ORDER}
        for case in cases:
            values['profile'].append(PROFILE_RANK.get(case.get('profile', DEFAULT_PROFILE), 0))
            values['previous_treatment'].append(bool(case.get('previous_treatment', False)))
            values['hiv_status'].append(case.get('hiv_status', DEFAULT_HIV))
            values['region'].append(case.get('region', DEFAULT_REGION))
            values['age'].append(case.get('age', DEFAULT_AGE))
            values['diabetes'].append(bool(case.get('diabetes', False)))
            values['sex'].append(case.get('sex', DEFAULT_SEX))

        numeric = ('profile', 'age')
        return {name: np.array(column, dtype=float if name in numeric else None)
                for name, column in values.items()}

    def scores(self, query_case):
        """Weighted similarity of the query to every stored case, vectorized."""
        p = self.columns
        w = self.weights
        q_rank = PROFILE_RANK.get(query_case.get('profile', DEFAULT_PROFILE), 0)
        age_gap = np.abs(p['age'] - query_case.get('age', DEFAULT_AGE))
        q_tx = bool(query_case.get('previous_treatment', False))

        total = w['profile'] * (1.0 - np.abs(p['profile'] - q_rank) / PROFILE_SPAN)
        total = total + w['hiv_status'] * (p['hiv_status'] ==
                                           query_case.get('hiv_status', DEFAULT_HIV))
        total = total + w['age'] * np.maximum(0.0, 1.0 - age_gap / AGE_SCALE)
        total = total + w['region'] * np.where(
            p['region'] == query_case.get('region', DEFAULT_REGION), 1.0, REGION_FLOOR)
        total = total + w['diabetes'] * (p['diabetes'] ==
                                         bool(query_case.get('diabetes', False)))
        total = total + w['previous_treatment'] * (p['previous_treatment'] == q_tx)
        return total + w['sex'] * (p['sex'] == query_case.get('sex', DEFAULT_SEX))

    def explain(self, query, case):
        """Per-feature breakdown of one pairing, ranked by the similarity each
        feature contributed or cost. The ranking key is rounded first, so equal
        contributions tie on feature order rather than on float representation."""
        sims = {f: self.feature_funcs[f](query, case) for f in FEATURE_ORDER}
        contribution = {f: round(sims[f] * self.weights[f], RANK_PRECISION)
                        for f in FEATURE_ORDER}
        deficit = {f: round(self.weights[f] - contribution[f], RANK_PRECISION)
                   for f in FEATURE_ORDER if sims[f] < 1.0}

        breakdown = [{
            'feature': f,
            'query_value': self.readable(query.get(f)),
            'case_value': self.readable(case.get(f)),
            'similarity': round(sims[f], 3),
            'weight': self.weights[f],
            'contribution': round(contribution[f], 3),
            'match': self.match_type(sims[f])
        } for f in FEATURE_ORDER]

        return {
            'breakdown': breakdown,
            'top_matches': sorted(FEATURE_ORDER, key=contribution.get,
                                  reverse=True)[:MAX_TOP_MATCHES],
            'key_differences': sorted(deficit, key=deficit.get,
                                      reverse=True)[:MAX_KEY_DIFFERENCES]
        }

    def readable(self, val):
        if val is None:
            return 'N/A'
        if isinstance(val, bool):
            return 'Yes' if val else 'No'
        return str(val)

    def match_type(self, sim):
        if sim >= EXACT_MATCH:
            return 'exact'
        if sim >= CLOSE_MATCH:
            return 'close'
        if sim >= PARTIAL_MATCH:
            return 'partial'
        return 'different'

    def profile_similarity(self, case1, case2):
        r1 = PROFILE_RANK.get(case1.get('profile', DEFAULT_PROFILE), 0)
        r2 = PROFILE_RANK.get(case2.get('profile', DEFAULT_PROFILE), 0)
        return 1.0 - abs(r1 - r2) / PROFILE_SPAN

    def hiv_similarity(self, case1, case2):
        return float(case1.get('hiv_status', DEFAULT_HIV) == case2.get('hiv_status', DEFAULT_HIV))

    def age_similarity(self, case1, case2):
        diff = abs(case1.get('age', DEFAULT_AGE) - case2.get('age', DEFAULT_AGE))
        return max(0.0, 1.0 - diff / AGE_SCALE)

    def region_similarity(self, case1, case2):
        same = case1.get('region', DEFAULT_REGION) == case2.get('region', DEFAULT_REGION)
        return 1.0 if same else REGION_FLOOR

    def diabetes_similarity(self, case1, case2):
        return float(bool(case1.get('diabetes', False)) == bool(case2.get('diabetes', False)))

    def previous_tx_similarity(self, case1, case2):
        return float(bool(case1.get('previous_treatment', False)) ==
                     bool(case2.get('previous_treatment', False)))

    def sex_similarity(self, case1, case2):
        return float(case1.get('sex', DEFAULT_SEX) == case2.get('sex', DEFAULT_SEX))


# Confidence factors. Retrieval is evidence volume and closeness, consistency
# is neighbor agreement, evidence is support behind the top regimen.
GOOD_CASE_COUNT = 8
EVIDENCE_CASE_TARGET = 5
CONFIDENCE_WEIGHTS = {'retrieval': 0.40, 'consistency': 0.35, 'evidence': 0.25}
HIGH_CONFIDENCE = 0.70
MODERATE_CONFIDENCE = 0.45
GOOD_COVERAGE = 0.70
MODERATE_COVERAGE = 0.40
CLEAR_PATTERN = 0.60
MIXED_PATTERN = 0.40


class ConfidenceCalculator:
    def score(self, similar_cases, outcome_dist, recommendations):
        """Confidence for a non-empty neighbor set. The caller handles the empty
        case, so every factor here has something to read."""
        n = len(similar_cases)
        avg_sim = sum(s for s, _ in similar_cases) / n
        retrieval = self.retrieval_score(n, avg_sim)
        consistency = self.consistency_score(outcome_dist)
        evidence = self.evidence_score(recommendations)

        w = CONFIDENCE_WEIGHTS
        total = (retrieval * w['retrieval'] + consistency * w['consistency'] +
                 evidence * w['evidence'])

        return {
            'level': self.level_name(total),
            'score': round(total, 2),
            'factors': {
                'retrieval': {
                    'score': round(retrieval, 2),
                    'reason': f"{n} cases found, avg similarity {avg_sim:.2f}"},
                'consistency': {
                    'score': round(consistency, 2),
                    'reason': self.consistency_reason(outcome_dist)},
                'evidence': {
                    'score': round(evidence, 2),
                    'reason': self.evidence_reason(recommendations)}
            },
            'interpretation': self.interpretation(retrieval, consistency, n)
        }

    def empty_confidence(self):
        return {
            'level': 'low',
            'score': 0.0,
            'factors': {},
            'interpretation': 'No similar cases found in case base.'
        }

    def retrieval_score(self, n, avg_sim):
        """Coverage and closeness, averaged. Closeness spans the range retrieval
        admits, from MIN_SIMILARITY to 1.0, so it stays in [0, 1] unclamped."""
        count_score = min(1.0, n / GOOD_CASE_COUNT)
        sim_score = (avg_sim - MIN_SIMILARITY) / (1.0 - MIN_SIMILARITY)
        return (count_score + sim_score) / 2

    def consistency_score(self, outcome_dist):
        """Share held by the most common outcome among the neighbors."""
        total = sum(outcome_dist.values())
        return max(outcome_dist.values()) / total if total else 0.0

    def evidence_score(self, recommendations):
        if not recommendations:
            return 0.0
        top = recommendations[0]
        case_score = min(1.0, top['evidence_cases'] / EVIDENCE_CASE_TARGET)
        return (case_score + top['success_rate']) / 2

    def level_name(self, score):
        if score >= HIGH_CONFIDENCE:
            return 'high'
        if score >= MODERATE_CONFIDENCE:
            return 'moderate'
        return 'low'

    def consistency_reason(self, outcome_dist):
        total = sum(outcome_dist.values())
        if not total:
            return "No outcome data"
        success = outcome_dist.get('success', 0)
        return f"Outcomes: {success}/{total} success ({success / total:.0%})"

    def evidence_reason(self, recommendations):
        if not recommendations:
            return "No regimen recommendations"
        top = recommendations[0]
        return (f"Top regimen: {top['evidence_cases']} cases, "
                f"{top['success_rate']:.0%} success")

    def interpretation(self, retrieval, consistency, n_cases):
        if retrieval >= GOOD_COVERAGE:
            coverage = f"Good case coverage ({n_cases} similar cases)"
        elif retrieval >= MODERATE_COVERAGE:
            coverage = f"Moderate case coverage ({n_cases} cases)"
        else:
            coverage = f"Limited case coverage ({n_cases} cases)"

        if consistency >= CLEAR_PATTERN:
            agreement = "outcomes show clear pattern"
        elif consistency >= MIXED_PATTERN:
            agreement = "mixed outcomes among similar cases"
        else:
            agreement = "highly variable outcomes"

        return f"{coverage}. {agreement}."


# A feature counts as a risk when it is this much more common among failures.
RISK_GAP = 0.15
AGE_RISK_GAP = 8


class OutcomeAnalyzer:
    def distribution(self, similar_cases):
        dist = {}
        for _, case in similar_cases:
            outcome = case.get('outcome', 'unknown')
            dist[outcome] = dist.get(outcome, 0) + 1
        return dist

    def risk_factors(self, similar_cases):
        """Features over-represented among the failures. Needs both outcomes
        present. All four are reported when all four hold, since they sit on
        different scales and cannot be ranked against each other."""
        failed = [c for _, c in similar_cases if c.get('outcome') != 'success']
        succeeded = [c for _, c in similar_cases if c.get('outcome') == 'success']
        if not failed or not succeeded:
            return []

        factors = []
        factors.extend(self.factor_gap(failed, succeeded, 'hiv_status', 'positive', 'HIV+'))
        factors.extend(self.factor_gap(failed, succeeded, 'diabetes', True, 'Diabetes'))
        factors.extend(self.factor_gap(failed, succeeded, 'previous_treatment', True,
                                        'Previous Tx'))
        factors.extend(self.age_risk(failed, succeeded))
        return factors

    def factor_gap(self, failed, succeeded, key, risk_val, label):
        fail_rate = sum(c.get(key) == risk_val for c in failed) / len(failed)
        success_rate = sum(c.get(key) == risk_val for c in succeeded) / len(succeeded)
        return [label] if fail_rate > success_rate + RISK_GAP else []

    def age_risk(self, failed, succeeded):
        fail_avg = sum(c.get('age', DEFAULT_AGE) for c in failed) / len(failed)
        success_avg = sum(c.get('age', DEFAULT_AGE) for c in succeeded) / len(succeeded)
        return ['Older age'] if fail_avg > success_avg + AGE_RISK_GAP else []


class CaseRetriever:
    def __init__(self, cases):
        self.cases = cases
        self.calculator = SimilarityCalculator(cases)
        self.case_ids = np.array([c.get('case_id') for c in cases])
        self.reference_year = max((c.get('year', DEFAULT_YEAR) for c in cases),
                                  default=DEFAULT_YEAR)
        self.temporal = self.temporal_weights()

    def temporal_weights(self):
        years = np.array([c.get('year', DEFAULT_YEAR) for c in self.cases], dtype=float)
        return np.maximum(TEMPORAL_FLOOR, 1.0 - (self.reference_year - years) * TEMPORAL_DECAY)

    def retrieve(self, query_case, k=DEFAULT_NEIGHBORS, min_similarity=MIN_SIMILARITY,
                 exclude_id=None):
        """Nearest cases above the similarity cutoff. Ranking discounts older
        cases, so the similarity returned with each case is not the sort key."""
        sims = np.round(self.calculator.scores(query_case), RANK_PRECISION)
        mask = sims >= min_similarity
        if exclude_id is not None:
            mask = mask & (self.case_ids != exclude_id)

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []

        ranking = sims[idx] * self.temporal[idx]
        order = idx[np.argsort(-ranking, kind='stable')][:k]
        return [(float(sims[i]), self.cases[i]) for i in order]


class CBREngine:
    def __init__(self, cases):
        self.retriever = CaseRetriever(cases)
        self.calculator = self.retriever.calculator
        self.confidence_calc = ConfidenceCalculator()
        self.outcome_analyzer = OutcomeAnalyzer()
        self.prior_success = self.success_share(cases)
        self.profile_regimens = self.regimens_by_profile(cases)

    @staticmethod
    def regimens_by_profile(cases):
        """Regimens the case base pairs with each profile. Retrieval admits
        neighbors from adjacent profiles, so without this a query can be handed
        a regimen that is not applicable to it."""
        seen = {}
        for case in cases:
            seen.setdefault(case.get('profile'), set()).add(case.get('regimen'))
        return seen

    @staticmethod
    def success_share(cases):
        """Success share of the whole case base, used when retrieval finds nothing."""
        if not cases:
            return 0.0
        return sum(c.get('outcome') == 'success' for c in cases) / len(cases)

    def recommend(self, query_case, k=DEFAULT_NEIGHBORS, exclude_id=None):
        similar_cases = self.retriever.retrieve(query_case, k=k, exclude_id=exclude_id)
        if not similar_cases:
            return self.default_recommendation(query_case)

        outcome_dist = self.outcome_analyzer.distribution(similar_cases)
        recommendations = self.regimen_recommendations(
            similar_cases, query_case.get('profile', DEFAULT_PROFILE))

        return {
            'query_profile': self.query_summary(query_case),
            'similar_cases': similar_cases,
            'success_rate': self.success_rate(similar_cases),
            'recommendations': recommendations,
            'confidence': self.confidence_calc.score(similar_cases, outcome_dist,
                                                     recommendations),
            'outcome_analysis': {
                'distribution': outcome_dist,
                'risk_factors': self.outcome_analyzer.risk_factors(similar_cases)
            }
        }

    def query_summary(self, query_case):
        return {
            'profile': query_case.get('profile', 'Unknown'),
            'age': query_case.get('age', 'N/A'),
            'sex': query_case.get('sex', 'N/A'),
            'hiv_status': query_case.get('hiv_status', 'N/A'),
            'region': query_case.get('region', 'N/A'),
            'diabetes': query_case.get('diabetes', False),
            'previous_treatment': query_case.get('previous_treatment', False)
        }

    def success_rate(self, similar_cases):
        """Share of neighbors that succeeded. The interface and validation both
        read this one figure, unrounded, so calibration sees the estimate."""
        successes = sum(case['outcome'] == 'success' for _, case in similar_cases)
        return successes / len(similar_cases)

    def regimen_stats(self, similar_cases):
        stats = {}
        for _, case in similar_cases:
            entry = stats.setdefault(case.get('regimen', 'Unknown'), {'count': 0, 'success': 0})
            entry['count'] += 1
            entry['success'] += case['outcome'] == 'success'
        for entry in stats.values():
            entry['success_rate'] = entry['success'] / entry['count']
        return stats

    def regimen_recommendations(self, similar_cases, profile):
        """Regimens with enough supporting neighbors, best success rate first,
        restricted to those the case base pairs with the query profile. An
        empty list means the neighbors carried nothing applicable, which is the
        honest answer. Both filters must run before the cut."""
        applicable = self.profile_regimens.get(profile, set())
        stats = self.regimen_stats(similar_cases)
        supported = [(regimen, entry) for regimen, entry in stats.items()
                     if entry['count'] >= MIN_EVIDENCE_CASES and regimen in applicable]
        supported.sort(key=lambda item: (item[1]['success_rate'], item[1]['count']), reverse=True)

        return [{
            'regimen': regimen,
            'success_rate': entry['success_rate'],
            'evidence_cases': entry['count'],
            'confidence': self.regimen_confidence(entry['count'], entry['success_rate'])
        } for regimen, entry in supported[:MAX_RECOMMENDATIONS]]

    def regimen_confidence(self, case_count, success_rate):
        if case_count >= STRONG_EVIDENCE_CASES and success_rate >= STRONG_EVIDENCE_RATE:
            return 'high'
        if case_count >= FAIR_EVIDENCE_CASES and success_rate >= FAIR_EVIDENCE_RATE:
            return 'moderate'
        return 'low'

    def explanations(self, query_case, similar_cases):
        """Feature breakdown of the nearest neighbors. Separate from recommend so
        the evaluation path does not build it."""
        explained = []
        for sim, case in similar_cases[:EXPLAINED_CASES]:
            explanation = self.calculator.explain(query_case, case)
            explained.append({
                'case_id': case.get('case_id', 'Unknown'),
                'similarity': round(sim, 3),
                'outcome': case.get('outcome', 'unknown'),
                'regimen': case.get('regimen', 'Unknown'),
                'feature_breakdown': explanation['breakdown'],
                'top_matches': explanation['top_matches'],
                'key_differences': explanation['key_differences']
            })
        return explained

    def default_recommendation(self, query_case):
        """No neighbor cleared the cutoff, so the reported probability falls back
        to the case base prior rather than zero."""
        profile = query_case.get('profile', DEFAULT_PROFILE)
        return {
            'query_profile': self.query_summary(query_case),
            'similar_cases': [],
            'success_rate': self.prior_success,
            'recommendations': [{
                'regimen': DEFAULT_REGIMENS.get(profile, DEFAULT_REGIMENS[DEFAULT_PROFILE]),
                'success_rate': 0.0,
                'evidence_cases': 0,
                'confidence': 'low'
            }],
            'confidence': self.confidence_calc.empty_confidence(),
            'outcome_analysis': {'distribution': {}, 'risk_factors': []}
        }


class CaseStore:
    def __init__(self, ontology=None):
        self.ontology = ontology

    def merge(self, cases, batch_size=CASE_BATCH):
        if not self.ontology:
            return 0

        self.constraints()
        for i in range(0, len(cases), batch_size):
            self.batch(cases[i:i + batch_size])
        return len(cases)

    def constraints(self):
        # Idempotent. A backend holding the constraint already, or spelling it
        # differently, must not stop the write that follows.
        query = "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE"
        try:
            self.ontology.query(query)
        except Exception:
            pass

    def batch(self, cases):
        query = """
        UNWIND $cases AS case_data
        MERGE (c:Case {case_id: case_data.case_id})
        SET c.patient_id = case_data.patient_id,
            c.strain_id = case_data.strain_id,
            c.age = case_data.age,
            c.sex = case_data.sex,
            c.region = case_data.region,
            c.hiv_status = case_data.hiv_status,
            c.diabetes = case_data.diabetes,
            c.previous_treatment = case_data.previous_treatment,
            c.profile = case_data.profile,
            c.regimen = case_data.regimen,
            c.outcome = case_data.outcome,
            c.duration_months = case_data.duration_months,
            c.year = case_data.year
        """
        self.ontology.query(query, {'cases': cases})

    def case_count(self):
        if not self.ontology:
            return 0
        result = self.ontology.query("MATCH (c:Case) RETURN count(c) as count")
        return result[0]['count'] if result else 0

    def clear_cases(self):
        if self.ontology:
            self.ontology.query("MATCH (c:Case) DETACH DELETE c")

    def retrieve_cases(self, profile=None, limit=CASE_LIMIT):
        if not self.ontology:
            return []

        if profile:
            query = "MATCH (c:Case {profile: $profile}) RETURN c LIMIT $limit"
            results = self.ontology.query(query, {'profile': profile, 'limit': limit})
        else:
            query = "MATCH (c:Case) RETURN c LIMIT $limit"
            results = self.ontology.query(query, {'limit': limit})

        return [dict(row['c']) for row in results]


def graph_cases(n_cases=DEFAULT_CASES, seed=DEFAULT_SEED):
    from tb_ontology import TBOntology

    cases = case_base(n_cases, seed)
    store = CaseStore(TBOntology())
    store.clear_cases()
    return store.merge(cases)