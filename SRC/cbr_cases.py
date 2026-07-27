"""Synthetic patient case base for case-based reasoning. No open dataset links
genotype, regimen, and outcome at the scale retrieval needs, so the cases are
generated from a fixed seed and are reproducible rather than sampled.
"""

import random
from collections import Counter

DEFAULT_CASES = 1000
DEFAULT_SEED = 42

# WHO-informed regional structure. The regions, lineages, and regimen names are
# real, but the rates and demographic magnitudes are synthetic approximations,
# not figures transcribed from any specific WHO publication. See README
# Limitations.
#
# European sits far above the others on mdr_rate on purpose. The region stands
# for the Eastern Europe and Central Asia belt, Moldova, Ukraine, Russia, and
# Kazakhstan in the seed graph, which carries the highest multidrug-resistant
# burden of any region. Its prev_tx_rate is the highest for the same reason,
# since prior treatment is a leading risk factor for resistance.
REGION_DATA = {
    'African': {
        'hiv_rate': 0.26, 'diabetes_rate': 0.08, 'age_mean': 34, 'age_std': 12,
        'mdr_rate': 0.035, 'male_ratio': 0.65, 'prev_tx_rate': 0.15, 'weight': 0.25
    },
    'SE_Asia': {
        'hiv_rate': 0.05, 'diabetes_rate': 0.15, 'age_mean': 42, 'age_std': 14,
        'mdr_rate': 0.028, 'male_ratio': 0.68, 'prev_tx_rate': 0.12, 'weight': 0.30
    },
    'E_Mediterranean': {
        'hiv_rate': 0.02, 'diabetes_rate': 0.18, 'age_mean': 38, 'age_std': 13,
        'mdr_rate': 0.041, 'male_ratio': 0.62, 'prev_tx_rate': 0.14, 'weight': 0.10
    },
    'W_Pacific': {
        'hiv_rate': 0.03, 'diabetes_rate': 0.12, 'age_mean': 48, 'age_std': 15,
        'mdr_rate': 0.052, 'male_ratio': 0.70, 'prev_tx_rate': 0.18, 'weight': 0.15
    },
    'European': {
        'hiv_rate': 0.08, 'diabetes_rate': 0.10, 'age_mean': 44, 'age_std': 15,
        'mdr_rate': 0.18, 'male_ratio': 0.70, 'prev_tx_rate': 0.25, 'weight': 0.10
    },
    'Americas': {
        'hiv_rate': 0.09, 'diabetes_rate': 0.14, 'age_mean': 40, 'age_std': 14,
        'mdr_rate': 0.032, 'male_ratio': 0.67, 'prev_tx_rate': 0.16, 'weight': 0.10
    }
}

# Base success rate for each profile and regimen the generator can produce.
BASE_SUCCESS = {
    ('Susceptible', '2HRZE_4HR'): 0.88,
    ('MonoResistant', '6REZ_Lfx'): 0.84,
    ('PolyResistant', 'Individualized_12mo'): 0.76,
    ('PolyResistant', 'AllOral_9mo'): 0.74,
    ('MDR', 'BPaLM'): 0.82,
    ('MDR', 'AllOral_9mo'): 0.73,
    ('MDR', 'Long_1820mo'): 0.63,
    ('PreXDR', 'BPaL'): 0.68,
    ('PreXDR', 'Individualized_18mo'): 0.62,
    ('XDR', 'BPaL'): 0.58,
    ('XDR', 'Individualized_18mo'): 0.52,
    ('XDR', 'Individualized_20mo'): 0.48
}

# Regimen share by profile and year. Every profile carries every year, and the
# shares within a year are weights rather than probabilities.
REGIMEN_OPTIONS = {
    'Susceptible': {
        '2022': [('2HRZE_4HR', 1.00)],
        '2023': [('2HRZE_4HR', 1.00)],
        '2024': [('2HRZE_4HR', 1.00)]
    },
    'MonoResistant': {
        '2022': [('6REZ_Lfx', 1.00)],
        '2023': [('6REZ_Lfx', 1.00)],
        '2024': [('6REZ_Lfx', 1.00)]
    },
    'PolyResistant': {
        '2022': [('Individualized_12mo', 0.70), ('AllOral_9mo', 0.30)],
        '2023': [('Individualized_12mo', 0.55), ('AllOral_9mo', 0.45)],
        '2024': [('AllOral_9mo', 0.55), ('Individualized_12mo', 0.45)]
    },
    'MDR': {
        '2022': [('BPaLM', 0.35), ('AllOral_9mo', 0.35), ('Long_1820mo', 0.30)],
        '2023': [('BPaLM', 0.45), ('AllOral_9mo', 0.30), ('Long_1820mo', 0.25)],
        '2024': [('BPaLM', 0.55), ('AllOral_9mo', 0.28), ('Long_1820mo', 0.17)]
    },
    'PreXDR': {
        '2022': [('Individualized_18mo', 0.60), ('BPaL', 0.40)],
        '2023': [('Individualized_18mo', 0.50), ('BPaL', 0.50)],
        '2024': [('BPaL', 0.55), ('Individualized_18mo', 0.45)]
    },
    'XDR': {
        '2022': [('Individualized_20mo', 0.50), ('Individualized_18mo', 0.30), ('BPaL', 0.20)],
        '2023': [('BPaL', 0.35), ('Individualized_20mo', 0.35), ('Individualized_18mo', 0.30)],
        '2024': [('BPaL', 0.45), ('Individualized_18mo', 0.30), ('Individualized_20mo', 0.25)]
    }
}

REGIMEN_DURATION = {
    '2HRZE_4HR': 6, '6REZ_Lfx': 6, 'BPaLM': 6, 'BPaL': 6, 'AllOral_9mo': 9,
    'Individualized_12mo': 12, 'Long_1820mo': 18, 'Individualized_18mo': 18,
    'Individualized_20mo': 20
}

# Share of the case base each profile should hold. Chosen for retrieval
# coverage rather than to match population prevalence.
PROFILE_TARGETS = {
    'Susceptible': 0.50,
    'MonoResistant': 0.12,
    'PolyResistant': 0.06,
    'MDR': 0.18,
    'PreXDR': 0.08,
    'XDR': 0.06
}

# Relative chance of each profile before the regional and treatment adjustments.
PROFILE_BASE_WEIGHT = {
    'Susceptible': 1.0, 'MonoResistant': 0.5, 'PolyResistant': 0.25,
    'MDR': 0.3, 'PreXDR': 0.1, 'XDR': 0.05
}

# Regional mdr_rate is expressed against this reference, so a region at the
# reference neither raises nor lowers the chance of a resistant profile.
REFERENCE_MDR_RATE = 0.05
MINOR_RESISTANCE = ('MonoResistant', 'PolyResistant')
MAJOR_RESISTANCE = ('MDR', 'PreXDR', 'XDR')
PREV_TX_MINOR_BOOST = 1.6
PREV_TX_MAJOR_BOOST = 2.5

YEARS = [2022, 2023, 2024]
YEAR_WEIGHTS = [0.30, 0.35, 0.35]

# Demographic bounds and adjustments.
HIV_RESISTANT_BOOST = 1.3
HIV_CEILING = 0.40
HIV_AGE_SHIFT = 5
AGE_MIN = 18
AGE_MAX = 80
DIABETES_CEILING = 0.35
DIABETES_OLDER = (50, 1.8)
DIABETES_MIDDLE = (40, 1.3)

# Outcome floor. The multipliers below can drive a hard case under this, and a
# treatment success rate of zero is not a claim the case base should make.
SUCCESS_FLOOR = 0.25

FAILURE_TYPES = ['death', 'failed', 'ltfu', 'not_evaluated']
FAILURE_WEIGHTS = {
    'minor': [0.25, 0.17, 0.42, 0.16],
    'major': [0.47, 0.19, 0.28, 0.06]
}


def profile_quota(n):
    """Case count per profile, largest remainder so the parts sum to n exactly.
    Truncating instead would leave a shortfall that the sampler had to absorb
    somewhere, which silently favored whichever profile it fell back to."""
    exact = {p: n * share for p, share in PROFILE_TARGETS.items()}
    quota = {p: int(v) for p, v in exact.items()}
    order = sorted(exact, key=lambda p: exact[p] - quota[p], reverse=True)
    for profile in order[:n - sum(quota.values())]:
        quota[profile] += 1
    return quota


class CaseGenerator:
    def __init__(self, seed=DEFAULT_SEED):
        self.rng = random.Random(seed)
        self.regions = list(REGION_DATA)
        self.region_weights = [REGION_DATA[r]['weight'] for r in self.regions]

    def cases(self, n=DEFAULT_CASES):
        """One case per index, with the profile mix held to its quota."""
        quota = profile_quota(n)
        counts = dict.fromkeys(PROFILE_TARGETS, 0)

        built = []
        for i in range(n):
            case = self.one_case(i, counts, quota)
            counts[case['profile']] += 1
            built.append(case)
        return built

    def one_case(self, index, counts, quota):
        region = self.region()
        year = self.year()
        previous_treatment = self.previous_treatment(region)

        case = {
            'case_id': f'CASE{index + 1:04d}',
            'patient_id': f'P{index + 1000:04d}',
            'strain_id': f'TB{index + 200:03d}',
            'region': region,
            'year': year,
            'previous_treatment': previous_treatment,
            'profile': self.profile_draw(region, previous_treatment, counts, quota)
        }

        self.demographics(case)
        case['regimen'] = self.regimen(case['profile'], year)
        case['duration_months'] = REGIMEN_DURATION[case['regimen']]
        case['outcome'] = self.outcome(case)
        return case

    def region(self):
        return self.rng.choices(self.regions, weights=self.region_weights)[0]

    def year(self):
        return self.rng.choices(YEARS, weights=YEAR_WEIGHTS)[0]

    def previous_treatment(self, region):
        return self.rng.random() < REGION_DATA[region]['prev_tx_rate']

    def profile_draw(self, region, previous_treatment, counts, quota):
        """Weighted draw among profiles below quota. The quota sums to the case
        count, so one profile always has room and there is no fallback to make.
        random.choices normalizes the weights, so they are passed as they are."""
        open_profiles = [p for p in quota if counts[p] < quota[p]]
        weights = [self.profile_weight(p, region, previous_treatment) for p in open_profiles]
        return self.rng.choices(open_profiles, weights=weights)[0]

    def profile_weight(self, profile, region, previous_treatment):
        weight = PROFILE_BASE_WEIGHT[profile]
        mdr_mult = REGION_DATA[region]['mdr_rate'] / REFERENCE_MDR_RATE

        if profile in MINOR_RESISTANCE:
            weight *= mdr_mult ** 0.5
            if previous_treatment:
                weight *= PREV_TX_MINOR_BOOST
        elif profile in MAJOR_RESISTANCE:
            weight *= mdr_mult
            if previous_treatment:
                weight *= PREV_TX_MAJOR_BOOST

        return weight

    def demographics(self, case):
        region_data = REGION_DATA[case['region']]
        case['hiv_status'] = self.hiv_status(region_data, case['profile'])
        case['age'] = self.age(region_data, case['hiv_status'])
        case['diabetes'] = self.diabetes(region_data, case['age'])
        case['sex'] = 'M' if self.rng.random() < region_data['male_ratio'] else 'F'

    def hiv_status(self, region_data, profile):
        rate = region_data['hiv_rate']
        if profile in MAJOR_RESISTANCE:
            rate *= HIV_RESISTANT_BOOST
        return 'positive' if self.rng.random() < min(rate, HIV_CEILING) else 'negative'

    def age(self, region_data, hiv_status):
        mean = region_data['age_mean']
        if hiv_status == 'positive':
            mean -= HIV_AGE_SHIFT
        age = int(self.rng.gauss(mean, region_data['age_std']))
        return max(AGE_MIN, min(AGE_MAX, age))

    def diabetes(self, region_data, age):
        rate = region_data['diabetes_rate']
        older, older_mult = DIABETES_OLDER
        middle, middle_mult = DIABETES_MIDDLE
        if age > older:
            rate *= older_mult
        elif age > middle:
            rate *= middle_mult
        return self.rng.random() < min(rate, DIABETES_CEILING)

    def regimen(self, profile, year):
        options = REGIMEN_OPTIONS[profile][str(year)]
        return self.rng.choices([r for r, _ in options], weights=[w for _, w in options])[0]

    def outcome(self, case):
        if self.rng.random() < self.success_rate(case):
            return 'success'
        return self.failure_type(case['profile'])

    def success_rate(self, case):
        """Base rate for the profile and regimen, adjusted downward for risk.
        Every adjustment is at most 1.0, so only the floor can bind."""
        base = BASE_SUCCESS[(case['profile'], case['regimen'])]
        rate = base * self.outcome_modifier(case) * self.interaction_modifier(case)
        return max(SUCCESS_FLOOR, rate)

    def outcome_modifier(self, case):
        modifier = 1.0
        if case['hiv_status'] == 'positive':
            modifier *= 0.90
        if case['diabetes']:
            modifier *= 0.94
        if case['age'] > 60:
            modifier *= 0.88
        elif case['age'] > 50:
            modifier *= 0.94
        if case['previous_treatment']:
            modifier *= 0.85
        if case['sex'] == 'M':
            modifier *= 0.98
        return modifier

    def interaction_modifier(self, case):
        modifier = 1.0
        if case['hiv_status'] == 'positive' and case['diabetes']:
            modifier *= 0.94
        if case['hiv_status'] == 'positive' and case['age'] > 55:
            modifier *= 0.92
        if case['previous_treatment'] and case['profile'] in MAJOR_RESISTANCE:
            modifier *= 0.90
        if case['diabetes'] and case['age'] > 60:
            modifier *= 0.95
        return modifier

    def failure_type(self, profile):
        band = 'major' if profile in MAJOR_RESISTANCE else 'minor'
        return self.rng.choices(FAILURE_TYPES, weights=FAILURE_WEIGHTS[band])[0]

    @staticmethod
    def distribution_summary(cases):
        """Shares and means over the case base, as proportions rather than
        percentages so the figures match what the notebook computes."""
        n = len(cases)
        if not n:
            return {}

        shares = {key: {k: round(v / n, 3) for k, v in Counter(c[key] for c in cases).items()}
                  for key in ('profile', 'region', 'year', 'outcome')}
        return {
            'total': n,
            **shares,
            'hiv_rate': round(sum(c['hiv_status'] == 'positive' for c in cases) / n, 3),
            'diabetes_rate': round(sum(c['diabetes'] for c in cases) / n, 3),
            'prev_tx_rate': round(sum(c['previous_treatment'] for c in cases) / n, 3),
            'avg_age': round(sum(c['age'] for c in cases) / n, 1),
            'success_rate': round(sum(c['outcome'] == 'success' for c in cases) / n, 3)
        }

    @staticmethod
    def profile_outcomes(cases):
        """Success share within each profile, with the counts behind it."""
        total = Counter(c['profile'] for c in cases)
        success = Counter(c['profile'] for c in cases if c['outcome'] == 'success')
        return {p: {'total': total[p], 'success': success[p],
                    'rate': round(success[p] / total[p], 3)} for p in total}


def regimen_ceiling():
    """Best regimen accuracy any predictor can reach on this case base. The
    generator draws the regimen from profile and year alone, so the bound is
    the share held by the most common option, averaged over both. Reported
    next to the baseline, since a score near it means the task is saturated
    rather than the method strong."""
    total = 0.0
    for profile, share in PROFILE_TARGETS.items():
        for year, year_weight in zip(YEARS, YEAR_WEIGHTS, strict=True):
            options = REGIMEN_OPTIONS[profile][str(year)]
            total += share * year_weight * max(w for _, w in options) / sum(w for _, w in options)
    return round(total, 3)


def case_base(n=DEFAULT_CASES, seed=DEFAULT_SEED):
    return CaseGenerator(seed).cases(n)


def main():
    cases = case_base()
    summary = CaseGenerator.distribution_summary(cases)

    print(f"Generated {summary['total']} cases")
    for key, title in (('profile', 'Profile'), ('region', 'Region'),
                       ('year', 'Year'), ('outcome', 'Outcome')):
        print(f"\n{title} distribution:")
        for name, share in sorted(summary[key].items()):
            print(f"  {name}: {share:.1%}")

    print("\nDemographics:")
    print(f"  HIV+: {summary['hiv_rate']:.1%}")
    print(f"  Diabetes: {summary['diabetes_rate']:.1%}")
    print(f"  Previous Tx: {summary['prev_tx_rate']:.1%}")
    print(f"  Avg age: {summary['avg_age']}")

    print("\nSuccess rate by profile:")
    for profile, stats in sorted(CaseGenerator.profile_outcomes(cases).items()):
        print(f"  {profile}: {stats['rate']:.1%} ({stats['success']}/{stats['total']})")


if __name__ == '__main__':
    main()