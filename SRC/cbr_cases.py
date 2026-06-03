import random

# WHO-informed regional structure: the regions, lineages, and regimen names are
# real, but the rates and demographic magnitudes are synthetic approximations,
# not figures transcribed from any specific WHO publication. See README Limitations.
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

# Base success rates by profile and regimen
BASE_SUCCESS = {
    ('Susceptible', '2HRZE_4HR'): 0.88,
    ('MonoResistant', '6REZ_Lfx'): 0.84,
    ('PolyResistant', 'Individualized_12mo'): 0.76,
    ('PolyResistant', 'AllOral_9mo'): 0.74,
    ('MDR', 'BPaLM'): 0.82,
    ('MDR', 'BPaL'): 0.80,
    ('MDR', 'AllOral_9mo'): 0.73,
    ('MDR', 'Long_1820mo'): 0.63,
    ('PreXDR', 'BPaL'): 0.68,
    ('PreXDR', 'Individualized_18mo'): 0.62,
    ('XDR', 'BPaL'): 0.58,
    ('XDR', 'Individualized_18mo'): 0.52,
    ('XDR', 'Individualized_20mo'): 0.48
}

# Regimen options by profile and year
REGIMEN_OPTIONS = {
    'Susceptible': {'2022': ['2HRZE_4HR'], '2023': ['2HRZE_4HR'], '2024': ['2HRZE_4HR']},
    'MonoResistant': {'2022': ['6REZ_Lfx'], '2023': ['6REZ_Lfx'], '2024': ['6REZ_Lfx']},
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


class CaseGenerator:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.regions = list(REGION_DATA.keys())
        self.region_weights = [REGION_DATA[r]['weight'] for r in self.regions]

        # Targeting distribution for adequate CBR coverage (synthetic, not real epidemiology)
        self.profile_targets = {
            'Susceptible': 0.50,
            'MonoResistant': 0.12,
            'PolyResistant': 0.06,
            'MDR': 0.18,
            'PreXDR': 0.08,
            'XDR': 0.06
        }

    def generate(self, n=1000):
        cases = []
        profile_counts = {p: 0 for p in self.profile_targets}
        profile_limits = {p: int(n * pct) for p, pct in self.profile_targets.items()}

        for i in range(n):
            case = self._single_case(i, profile_counts, profile_limits)
            profile_counts[case['profile']] += 1
            cases.append(case)
        return cases

    def _single_case(self, index, profile_counts, profile_limits):
        region = self._sample_region()
        year = self._sample_year()
        previous_treatment = self._sample_previous_treatment(region)
        profile = self._sample_profile(region, previous_treatment, profile_counts, profile_limits)

        case = {
            'case_id': f'CASE{index + 1:04d}',
            'patient_id': f'P{index + 1000:04d}',
            'strain_id': f'TB{index + 200:03d}',
            'region': region,
            'year': year,
            'previous_treatment': previous_treatment,
            'profile': profile
        }

        self._assign_demographics(case)
        case['regimen'] = self._sample_regimen(profile, year)
        case['duration_months'] = REGIMEN_DURATION.get(case['regimen'], 6)
        case['outcome'] = self._sample_outcome(case)

        return case

    def _sample_region(self):
        return self.rng.choices(self.regions, weights=self.region_weights)[0]

    def _sample_year(self):
        return self.rng.choices([2022, 2023, 2024], weights=[0.30, 0.35, 0.35])[0]

    def _sample_previous_treatment(self, region):
        base_rate = REGION_DATA[region]['prev_tx_rate']
        return self.rng.random() < base_rate

    def _sample_profile(self, region, previous_treatment, profile_counts, profile_limits):
        available = []
        weights = []

        for profile, limit in profile_limits.items():
            if profile_counts[profile] < limit:
                available.append(profile)
                weight = self._profile_weight(profile, region, previous_treatment)
                weights.append(weight)

        if not available:
            return 'Susceptible'

        total = sum(weights)
        weights = [w / total for w in weights]

        return self.rng.choices(available, weights=weights)[0]

    def _profile_weight(self, profile, region, previous_treatment):
        base_weights = {'Susceptible': 1.0, 'MonoResistant': 0.5, 'PolyResistant': 0.25,
                        'MDR': 0.3, 'PreXDR': 0.1, 'XDR': 0.05}
        weight = base_weights.get(profile, 0.1)

        mdr_mult = REGION_DATA[region]['mdr_rate'] / 0.05

        if profile in ['MonoResistant', 'PolyResistant']:
            weight *= mdr_mult ** 0.5
            if previous_treatment:
                weight *= 1.6
        elif profile in ['MDR', 'PreXDR', 'XDR']:
            weight *= mdr_mult
            if previous_treatment:
                weight *= 2.5

        return weight

    def _assign_demographics(self, case):
        region = case['region']
        profile = case['profile']
        region_data = REGION_DATA[region]

        case['hiv_status'] = self._sample_hiv(region_data, profile)
        case['age'] = self._sample_age(region_data, case['hiv_status'])
        case['diabetes'] = self._sample_diabetes(region_data, case['age'])
        case['sex'] = 'M' if self.rng.random() < region_data['male_ratio'] else 'F'

    def _sample_hiv(self, region_data, profile):
        hiv_rate = region_data['hiv_rate']
        if profile in ['MDR', 'PreXDR', 'XDR']:
            hiv_rate *= 1.3
        return 'positive' if self.rng.random() < min(hiv_rate, 0.40) else 'negative'

    def _sample_age(self, region_data, hiv_status):
        age_mean = region_data['age_mean']
        age_std = region_data['age_std']

        if hiv_status == 'positive':
            age_mean -= 5

        age = int(self.rng.gauss(age_mean, age_std))
        return max(18, min(80, age))

    def _sample_diabetes(self, region_data, age):
        diabetes_rate = region_data['diabetes_rate']

        if age > 50:
            diabetes_rate *= 1.8
        elif age > 40:
            diabetes_rate *= 1.3

        return self.rng.random() < min(diabetes_rate, 0.35)

    def _sample_regimen(self, profile, year):
        options = REGIMEN_OPTIONS.get(profile, {}).get(str(year))

        if not options:
            return '2HRZE_4HR'

        if isinstance(options[0], str):
            return self.rng.choice(options)

        regimens = [r[0] for r in options]
        weights = [r[1] for r in options]
        return self.rng.choices(regimens, weights=weights)[0]

    def _sample_outcome(self, case):
        success_rate = self._success_rate(case)

        if self.rng.random() < success_rate:
            return 'success'

        return self._sample_failure_type(case['profile'])

    def _success_rate(self, case):
        key = (case['profile'], case['regimen'])
        base_rate = BASE_SUCCESS.get(key, 0.70)

        modifier = self._outcome_modifier(case)
        interaction = self._interaction_modifier(case)

        final_rate = base_rate * modifier * interaction
        return max(0.25, min(0.95, final_rate))

    def _outcome_modifier(self, case):
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

    def _interaction_modifier(self, case):
        modifier = 1.0

        if case['hiv_status'] == 'positive' and case['diabetes']:
            modifier *= 0.94
        if case['hiv_status'] == 'positive' and case['age'] > 55:
            modifier *= 0.92
        if case['previous_treatment'] and case['profile'] in ['MDR', 'PreXDR', 'XDR']:
            modifier *= 0.90
        if case['diabetes'] and case['age'] > 60:
            modifier *= 0.95

        return modifier

    def _sample_failure_type(self, profile):
        types = ['death', 'failed', 'ltfu', 'not_evaluated']
        if profile in ('Susceptible', 'MonoResistant', 'PolyResistant'):
            weights = [0.25, 0.17, 0.42, 0.16]
        else:
            weights = [0.47, 0.19, 0.28, 0.06]

        return self.rng.choices(types, weights=weights)[0]

    def distribution_summary(self, cases):
        n = len(cases)
        if not n:
            return {'total': 0, 'profiles': {}, 'regions': {}, 'years': {}, 'outcomes': {},
                    'hiv_rate': 0.0, 'diabetes_rate': 0.0, 'prev_tx_rate': 0.0,
                    'avg_age': 0.0, 'success_rate': 0.0}
        tally = self._tally(cases)

        return {
            'total': n,
            'profiles': self._pct(tally['profiles'], n),
            'regions': self._pct(tally['regions'], n),
            'years': self._pct(tally['years'], n),
            'outcomes': self._pct(tally['outcomes'], n),
            'hiv_rate': round(tally['hiv'] / n * 100, 1),
            'diabetes_rate': round(tally['diabetes'] / n * 100, 1),
            'prev_tx_rate': round(tally['prev_tx'] / n * 100, 1),
            'avg_age': round(tally['age_sum'] / n, 1),
            'success_rate': round(tally['outcomes'].get('success', 0) / n * 100, 1)
        }

    @staticmethod
    def _pct(counter, n):
        return {k: round(v / n * 100, 1) for k, v in counter.items()}

    @staticmethod
    def _tally(cases):
        profiles, regions, years, outcomes = {}, {}, {}, {}
        hiv = diabetes = prev_tx = age_sum = 0

        for c in cases:
            profiles[c['profile']] = profiles.get(c['profile'], 0) + 1
            regions[c['region']] = regions.get(c['region'], 0) + 1
            years[c['year']] = years.get(c['year'], 0) + 1
            outcomes[c['outcome']] = outcomes.get(c['outcome'], 0) + 1
            if c['hiv_status'] == 'positive':
                hiv += 1
            if c['diabetes']:
                diabetes += 1
            if c['previous_treatment']:
                prev_tx += 1
            age_sum += c['age']

        return {'profiles': profiles, 'regions': regions, 'years': years,
                'outcomes': outcomes, 'hiv': hiv, 'diabetes': diabetes,
                'prev_tx': prev_tx, 'age_sum': age_sum}

    def profile_outcomes(self, cases):
        profile_stats = {}

        for c in cases:
            p = c['profile']
            if p not in profile_stats:
                profile_stats[p] = {'total': 0, 'success': 0}
            profile_stats[p]['total'] += 1
            if c['outcome'] == 'success':
                profile_stats[p]['success'] += 1

        for p, stats in profile_stats.items():
            stats['rate'] = round(stats['success'] / stats['total'] * 100, 1)

        return profile_stats


def generate_cases(n=1000, seed=42):
    generator = CaseGenerator(seed)
    return generator.generate(n)


if __name__ == '__main__':
    generator = CaseGenerator(seed=42)
    cases = generator.generate(1000)

    summary = generator.distribution_summary(cases)
    print(f"Generated {summary['total']} cases")
    print("\nProfile Distribution:")
    for profile, pct in sorted(summary['profiles'].items()):
        print(f"  {profile}: {pct}%")

    print("\nRegion Distribution:")
    for region, pct in sorted(summary['regions'].items()):
        print(f"  {region}: {pct}%")

    print("\nYear Distribution:")
    for year, pct in sorted(summary['years'].items()):
        print(f"  {year}: {pct}%")

    print("\nDemographics:")
    print(f"  HIV+: {summary['hiv_rate']}%")
    print(f"  Diabetes: {summary['diabetes_rate']}%")
    print(f"  Previous Tx: {summary['prev_tx_rate']}%")
    print(f"  Avg Age: {summary['avg_age']}")

    print("\nOutcomes:")
    for outcome, pct in sorted(summary['outcomes'].items()):
        print(f"  {outcome}: {pct}%")

    print("\nSuccess Rate by Profile:")
    profile_outcomes = generator.profile_outcomes(cases)
    for profile, stats in sorted(profile_outcomes.items()):
        print(f"  {profile}: {stats['rate']}% ({stats['success']}/{stats['total']})")