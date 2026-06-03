import numpy as np
from cbr_cases import generate_cases

FEATURE_ORDER = ['profile', 'previous_treatment', 'hiv_status', 'region', 'age', 'diabetes', 'sex']
PROFILE_RANK = {
    'Susceptible': 0, 'MonoResistant': 1, 'PolyResistant': 2,
    'MDR': 3, 'PreXDR': 4, 'XDR': 5
}
PROFILE_SPAN = max(PROFILE_RANK.values())


class SimilarityCalculator:
    def __init__(self):
        self.weights = {
            'profile': 0.32,
            'previous_treatment': 0.20,
            'hiv_status': 0.15,
            'region': 0.12,
            'age': 0.10,
            'diabetes': 0.07,
            'sex': 0.04
        }
        self.age_scale = 50.0
        self.region_floor = 0.5
        self.feature_funcs = {
            'profile': self._profile_similarity,
            'previous_treatment': self._previous_tx_similarity,
            'hiv_status': self._hiv_similarity,
            'region': self._region_similarity,
            'age': self._age_similarity,
            'diabetes': self._diabetes_similarity,
            'sex': self._sex_similarity
        }
        self._prepared = None

    def prepare(self, cases):
        """Cache the case base as column arrays so a query scores in one pass."""
        self._prepared = {
            'profile': np.array([PROFILE_RANK.get(c.get('profile', 'Susceptible'), 0) for c in cases], dtype=float),
            'hiv_status': np.array([c.get('hiv_status', 'negative') for c in cases]),
            'age': np.array([c.get('age', 40) for c in cases], dtype=float),
            'region': np.array([c.get('region', 'global') for c in cases]),
            'diabetes': np.array([bool(c.get('diabetes', False)) for c in cases]),
            'previous_treatment': np.array([bool(c.get('previous_treatment', False)) for c in cases]),
            'sex': np.array([c.get('sex', 'M') for c in cases])
        }

    def scores(self, query_case):
        """Weighted similarity of `query_case` to every prepared case, vectorized."""
        p = self._prepared
        w = self.weights
        q_rank = PROFILE_RANK.get(query_case.get('profile', 'Susceptible'), 0)
        total = w['profile'] * (1.0 - np.abs(p['profile'] - q_rank) / PROFILE_SPAN)
        total = total + w['hiv_status'] * (p['hiv_status'] == query_case.get('hiv_status', 'negative'))
        age_sim = np.maximum(0.0, 1.0 - np.abs(p['age'] - query_case.get('age', 40)) / self.age_scale)
        total = total + w['age'] * age_sim
        region_eq = p['region'] == query_case.get('region', 'global')
        total = total + w['region'] * np.where(region_eq, 1.0, self.region_floor)
        total = total + w['diabetes'] * (p['diabetes'] == bool(query_case.get('diabetes', False)))
        total = total + w['previous_treatment'] * (p['previous_treatment'] == bool(query_case.get('previous_treatment', False)))
        total = total + w['sex'] * (p['sex'] == query_case.get('sex', 'M'))
        return total

    def explain(self, query, case):
        breakdown = []
        for feature in FEATURE_ORDER:
            weight = self.weights[feature]
            sim = self.feature_funcs[feature](query, case)
            contribution = sim * weight

            query_val = self._display_value(query.get(feature))
            case_val = self._display_value(case.get(feature))

            breakdown.append({
                'feature': feature,
                'query_value': query_val,
                'case_value': case_val,
                'similarity': round(sim, 3),
                'weight': weight,
                'contribution': round(contribution, 3),
                'match': self._match_type(sim)
            })

        total = sum(b['contribution'] for b in breakdown)
        top_matches = [b['feature'] for b in breakdown if b['similarity'] >= 0.9][:3]
        key_diffs = [b['feature'] for b in breakdown if b['similarity'] < 0.5][:2]

        return {
            'total_similarity': round(total, 3),
            'breakdown': breakdown,
            'top_matches': top_matches,
            'key_differences': key_diffs
        }

    def _display_value(self, val):
        if val is None:
            return 'N/A'
        if isinstance(val, bool):
            return 'Yes' if val else 'No'
        return str(val)

    def _match_type(self, sim):
        if sim >= 0.95:
            return 'exact'
        if sim >= 0.7:
            return 'close'
        if sim >= 0.3:
            return 'partial'
        return 'different'

    def _profile_similarity(self, case1, case2):
        r1 = PROFILE_RANK.get(case1.get('profile', 'Susceptible'), 0)
        r2 = PROFILE_RANK.get(case2.get('profile', 'Susceptible'), 0)
        return max(0.0, 1.0 - abs(r1 - r2) / PROFILE_SPAN)

    def _hiv_similarity(self, case1, case2):
        return 1.0 if case1.get('hiv_status', 'negative') == case2.get('hiv_status', 'negative') else 0.0

    def _age_similarity(self, case1, case2):
        diff = abs(case1.get('age', 40) - case2.get('age', 40))
        return max(0.0, 1.0 - (diff / self.age_scale))

    def _region_similarity(self, case1, case2):
        return 1.0 if case1.get('region', 'global') == case2.get('region', 'global') else self.region_floor

    def _diabetes_similarity(self, case1, case2):
        return 1.0 if case1.get('diabetes', False) == case2.get('diabetes', False) else 0.0

    def _previous_tx_similarity(self, case1, case2):
        return 1.0 if case1.get('previous_treatment', False) == case2.get('previous_treatment', False) else 0.0

    def _sex_similarity(self, case1, case2):
        return 1.0 if case1.get('sex', 'M') == case2.get('sex', 'M') else 0.0


class ConfidenceCalculator:
    def __init__(self):
        self.min_cases = 3
        self.good_cases = 8
        self.high_similarity = 0.75
        self.acceptable_similarity = 0.60

    def score(self, similar_cases, outcome_dist, recommendations):
        if not similar_cases:
            return self._empty_confidence()

        retrieval = self._retrieval_score(similar_cases)
        consistency = self._consistency_score(similar_cases, outcome_dist)
        evidence = self._evidence_score(recommendations)

        total = (retrieval * 0.4) + (consistency * 0.35) + (evidence * 0.25)
        level = self._level(total)
        interpretation = self._interpretation(retrieval, consistency, evidence, len(similar_cases))

        return {
            'level': level,
            'score': round(total, 2),
            'factors': {
                'retrieval': {'score': round(retrieval, 2), 'reason': self._retrieval_reason(similar_cases)},
                'consistency': {'score': round(consistency, 2), 'reason': self._consistency_reason(outcome_dist)},
                'evidence': {'score': round(evidence, 2), 'reason': self._evidence_reason(recommendations)}
            },
            'interpretation': interpretation
        }

    def _empty_confidence(self):
        return {
            'level': 'low',
            'score': 0.0,
            'factors': {},
            'interpretation': 'No similar cases found in case base.'
        }

    def _retrieval_score(self, similar_cases):
        n = len(similar_cases)
        avg_sim = sum(s for s, _ in similar_cases) / n if n > 0 else 0

        count_score = min(1.0, n / self.good_cases)
        sim_score = (avg_sim - self.acceptable_similarity) / (self.high_similarity - self.acceptable_similarity)
        sim_score = max(0, min(1.0, sim_score))

        return (count_score * 0.5) + (sim_score * 0.5)

    def _consistency_score(self, similar_cases, outcome_dist):
        if not similar_cases:
            return 0.0

        total = sum(outcome_dist.values())
        if total == 0:
            return 0.0

        max_outcome = max(outcome_dist.values())
        return max_outcome / total

    def _evidence_score(self, recommendations):
        if not recommendations:
            return 0.0

        top_rec = recommendations[0] if recommendations else None
        if not top_rec:
            return 0.0

        cases = top_rec.get('evidence_cases', 0)
        rate = top_rec.get('success_rate', 0)

        case_score = min(1.0, cases / 5)
        rate_score = rate

        return (case_score * 0.5) + (rate_score * 0.5)

    def _level(self, score):
        if score >= 0.70:
            return 'high'
        if score >= 0.45:
            return 'moderate'
        return 'low'

    def _retrieval_reason(self, similar_cases):
        n = len(similar_cases)
        avg_sim = sum(s for s, _ in similar_cases) / n if n > 0 else 0
        return f"{n} cases found, avg similarity {avg_sim:.2f}"

    def _consistency_reason(self, outcome_dist):
        total = sum(outcome_dist.values())
        if total == 0:
            return "No outcome data"
        success = outcome_dist.get('success', 0)
        return f"Outcomes: {success}/{total} success ({success / total * 100:.0f}%)"

    def _evidence_reason(self, recommendations):
        if not recommendations:
            return "No regimen recommendations"
        top = recommendations[0]
        return f"Top regimen: {top['evidence_cases']} cases, {top['success_rate'] * 100:.0f}% success"

    def _interpretation(self, retrieval, consistency, evidence, n_cases):
        parts = []

        if retrieval >= 0.7:
            parts.append(f"Good case coverage ({n_cases} similar cases)")
        elif retrieval >= 0.4:
            parts.append(f"Moderate case coverage ({n_cases} cases)")
        else:
            parts.append(f"Limited case coverage ({n_cases} cases)")

        if consistency >= 0.6:
            parts.append("outcomes show clear pattern")
        elif consistency >= 0.4:
            parts.append("mixed outcomes among similar cases")
        else:
            parts.append("highly variable outcomes")

        return ". ".join(parts) + "."


class OutcomeAnalyzer:
    def analyze(self, similar_cases, distribution=None):
        if not similar_cases:
            return self._empty_analysis()

        if distribution is None:
            distribution = self.distribution(similar_cases)

        return {
            'distribution': distribution,
            'weighted_success_rate': round(self._weighted_success_rate(similar_cases), 3),
            'risk_factors': self._risk_factors(similar_cases)
        }

    def _empty_analysis(self):
        return {
            'distribution': {},
            'weighted_success_rate': 0.0,
            'risk_factors': []
        }

    def distribution(self, similar_cases):
        dist = {}
        for _, case in similar_cases:
            outcome = case.get('outcome', 'unknown')
            dist[outcome] = dist.get(outcome, 0) + 1
        return dist

    def _weighted_success_rate(self, similar_cases):
        if not similar_cases:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for sim, case in similar_cases:
            is_success = 1.0 if case.get('outcome') == 'success' else 0.0
            weighted_sum += sim * is_success
            weight_total += sim

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _risk_factors(self, similar_cases):
        factors = []

        failed_cases = [c for _, c in similar_cases if c.get('outcome') != 'success']
        success_cases = [c for _, c in similar_cases if c.get('outcome') == 'success']

        if not failed_cases or not success_cases:
            return factors

        factors.extend(self._factor_gap(failed_cases, success_cases, 'hiv_status', 'positive', 'HIV+'))
        factors.extend(self._factor_gap(failed_cases, success_cases, 'diabetes', True, 'Diabetes'))
        factors.extend(self._factor_gap(failed_cases, success_cases, 'previous_treatment', True, 'Previous Tx'))
        factors.extend(self._age_risk(failed_cases, success_cases))

        return factors[:3]

    def _factor_gap(self, failed, success, key, risk_val, label):
        fail_rate = sum(1 for c in failed if c.get(key) == risk_val) / len(failed)
        success_rate = sum(1 for c in success if c.get(key) == risk_val) / len(success)

        if fail_rate > success_rate + 0.15:
            return [label]
        return []

    def _age_risk(self, failed, success):
        fail_avg = sum(c.get('age', 40) for c in failed) / len(failed)
        success_avg = sum(c.get('age', 40) for c in success) / len(success)

        if fail_avg > success_avg + 8:
            return ['Older age']
        return []


class CaseRetriever:
    def __init__(self, cases):
        self.cases = cases
        self.calculator = SimilarityCalculator()
        self.calculator.prepare(cases)
        self.case_ids = np.array([c.get('case_id') for c in cases])
        self.reference_year = max((c.get('year', 2022) for c in cases), default=2022)
        self.temporal = self._temporal_weights()

    def _temporal_weights(self):
        years = np.array([c.get('year', 2022) for c in self.cases], dtype=float)
        return np.maximum(0.7, 1.0 - (self.reference_year - years) * 0.10)

    def retrieve(self, query_case, k=10, min_similarity=0.55, exclude_id=None):
        # Round before thresholding/ranking so float summation-order noise
        # cannot flip the 0.55 cutoff or the order of effectively-tied cases,
        # keeping retrieval deterministic across platforms.
        sims = np.round(self.calculator.scores(query_case), 12)
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
        self.cases = cases
        self.retriever = CaseRetriever(cases)
        self.calculator = self.retriever.calculator
        self.confidence_calc = ConfidenceCalculator()
        self.outcome_analyzer = OutcomeAnalyzer()

    def recommend(self, query_case, k=10, exclude_id=None):
        similar_cases = self.retriever.retrieve(query_case, k=k, exclude_id=exclude_id)

        if not similar_cases:
            return self._default_recommendation(query_case)

        outcome_dist = self.outcome_analyzer.distribution(similar_cases)
        recommendations = self._regimen_recommendations(similar_cases)
        explained_cases = self._case_explanations(query_case, similar_cases[:5])
        confidence = self.confidence_calc.score(similar_cases, outcome_dist, recommendations)
        success_rate = self._success_rate(similar_cases)

        return {
            'query_profile': self._query_summary(query_case),
            'similar_cases': similar_cases,
            'success_rate': success_rate,
            'outcome_probability': round(success_rate, 3),
            'recommendations': recommendations,
            'explained_cases': explained_cases,
            'confidence': confidence,
            'outcome_analysis': self.outcome_analyzer.analyze(similar_cases, outcome_dist)
        }

    def _query_summary(self, query_case):
        return {
            'profile': query_case.get('profile', 'Unknown'),
            'age': query_case.get('age', 'N/A'),
            'sex': query_case.get('sex', 'N/A'),
            'hiv_status': query_case.get('hiv_status', 'N/A'),
            'region': query_case.get('region', 'N/A'),
            'diabetes': query_case.get('diabetes', False),
            'previous_treatment': query_case.get('previous_treatment', False)
        }

    def _success_rate(self, similar_cases):
        if not similar_cases:
            return 0.0
        successes = sum(1 for _, case in similar_cases if case['outcome'] == 'success')
        return successes / len(similar_cases)

    def _regimen_stats(self, similar_cases):
        stats = {}
        for _, case in similar_cases:
            reg = case.get('regimen', 'Unknown')
            entry = stats.setdefault(reg, {'count': 0, 'success': 0})
            entry['count'] += 1
            if case['outcome'] == 'success':
                entry['success'] += 1
        for entry in stats.values():
            entry['success_rate'] = entry['success'] / entry['count'] if entry['count'] else 0.0
        return stats

    def _regimen_recommendations(self, similar_cases):
        stats = self._regimen_stats(similar_cases)
        ranked = sorted(stats.items(), key=lambda x: (x[1]['success_rate'], x[1]['count']), reverse=True)

        recommendations = []
        for regimen, entry in ranked[:3]:
            if entry['count'] >= 2:
                recommendations.append({
                    'regimen': regimen,
                    'success_rate': entry['success_rate'],
                    'evidence_cases': entry['count'],
                    'confidence': self._rec_confidence(entry['count'], entry['success_rate'])
                })
        return recommendations

    def _rec_confidence(self, case_count, success_rate):
        if case_count >= 5 and success_rate >= 0.70:
            return 'high'
        if case_count >= 3 and success_rate >= 0.55:
            return 'moderate'
        return 'low'

    def _case_explanations(self, query_case, similar_cases):
        explained = []
        for sim, case in similar_cases:
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

    def _default_recommendation(self, query_case):
        profile = query_case.get('profile', 'Susceptible')
        defaults = {
            'Susceptible': '2HRZE_4HR',
            'MonoResistant': '6REZ_Lfx',
            'PolyResistant': 'Individualized_12mo',
            'MDR': 'BPaLM',
            'PreXDR': 'BPaL',
            'XDR': 'BPaL'
        }

        return {
            'query_profile': self._query_summary(query_case),
            'similar_cases': [],
            'success_rate': 0.0,
            'outcome_probability': 0.0,
            'recommendations': [{
                'regimen': defaults.get(profile, '2HRZE_4HR'),
                'success_rate': 0.0,
                'evidence_cases': 0,
                'confidence': 'low'
            }],
            'explained_cases': [],
            'confidence': self.confidence_calc._empty_confidence(),
            'outcome_analysis': self.outcome_analyzer._empty_analysis()
        }


class CaseStore:
    def __init__(self, ontology=None):
        self.ontology = ontology

    def write(self, cases, batch_size=100):
        if not self.ontology:
            return 0

        self._constraints()

        for i in range(0, len(cases), batch_size):
            self._batch(cases[i:i + batch_size])

        return len(cases)

    def _constraints(self):
        query = "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.case_id IS UNIQUE"
        try:
            self.ontology.query(query)
        except Exception:
            pass

    def _batch(self, cases):
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

    def count_cases(self):
        if not self.ontology:
            return 0
        result = self.ontology.query("MATCH (c:Case) RETURN count(c) as count")
        return result[0]['count'] if result else 0

    def clear_cases(self):
        if self.ontology:
            self.ontology.query("MATCH (c:Case) DETACH DELETE c")

    def retrieve_cases(self, profile=None, limit=100):
        if not self.ontology:
            return []

        if profile:
            query = "MATCH (c:Case {profile: $profile}) RETURN c LIMIT $limit"
            results = self.ontology.query(query, {'profile': profile, 'limit': limit})
        else:
            query = "MATCH (c:Case) RETURN c LIMIT $limit"
            results = self.ontology.query(query, {'limit': limit})

        return [dict(row['c']) for row in results]


def graph_cases(n_cases=1000, seed=42):
    from tb_ontology import TBOntology

    cases = generate_cases(n_cases, seed)
    ontology = TBOntology()
    store = CaseStore(ontology)

    store.clear_cases()
    return store.write(cases)