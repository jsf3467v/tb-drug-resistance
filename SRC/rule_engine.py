import re

from config import FLUOROQUINOLONES, INJECTABLES

HIGH_CONF_GENES = ('rpoB', 'katG', 'inhA', 'embB', 'pncA', 'gyrA')

# Which resistance flag each drug raises. The class members come from config, so
# the engine grades a class exactly as wide as the label definition treats it.
DRUG_FLAG = {'rifampin': 'rifampin_resistance', 'isoniazid': 'isoniazid_resistance'}
DRUG_FLAG.update(dict.fromkeys(FLUOROQUINOLONES, 'fluoroquinolone_resistance'))
DRUG_FLAG.update(dict.fromkeys(INJECTABLES, 'injectable_resistance'))

# Cross-resistance is a class effect, so one member firing excludes the class.
# Sorted, because set order varies between processes and the report would too.
# A source may label a row with the class name rather than a drug, which still
# sets the flag, but a class is not something a patient can be excluded from and
# must not reach the exclusion list beside real drug names.
CLASS_SPEC = (('fluoroquinolone_resistance', 'fluoroquinolone', FLUOROQUINOLONES),
              ('injectable_resistance', 'injectable', INJECTABLES))
DRUG_CLASSES = {flag: (label, sorted(members - {label}))
                for flag, label, members in CLASS_SPEC}

# The katG codon whose substitution carries high-level isoniazid resistance.
KATG_HIGH_CONFIDENCE = 315

# Leading residue or nucleotide number in an HGVS or short-form token, so
# p.Ser315Thr and S315T both read as 315 and p.Ala3150Val does not.
POSITION_PATTERN = re.compile(r'-?\d+')

CLASS_SEVERITY = {'MDR': 1, 'PreXDR': 2, 'XDR': 3}

# Conditions naming a classification read the flag the classifier set.
CLASSIFICATION_GOALS = ('mdr', 'xdr', 'prexdr')


def mutation_position(mut):
    """Residue number, from the position field where the source carries one and
    from the mutation token otherwise. The CRyPTIC path supplies no position,
    so a substring test there would match any token containing the digits."""
    position = str(mut.get('position', '')).strip()
    if position.lstrip('-').isdigit():
        return int(position)
    found = POSITION_PATTERN.search(str(mut.get('mutation', '')))
    return int(found.group()) if found else None

# BPaLM adds a fluoroquinolone (moxifloxacin) to BPaL, so it is valid only when
# fluoroquinolones are not contraindicated. With fluoroquinolone resistance the
# fluoroquinolone-free BPaL regimen is used instead.
BPAL_DRUGS = ['bedaquiline', 'pretomanid', 'linezolid']
BPALM_DRUGS = BPAL_DRUGS + ['moxifloxacin']


class Rule:
    def __init__(self, rule_id, priority, conditions, actions, source):
        self.id = rule_id
        self.priority = priority
        self.conditions = conditions
        self.actions = actions
        self.source = source
        self.confidence = 1.0


class RuleEngine:
    def __init__(self, ontology):
        self.ontology = ontology
        self.fired = []
        self.working_memory = {}
        self.excluded = set()
        self.rules = [
            self.mdr_detection(),
            self.xdr_detection(),
            self.prexdr_detection(),
            self.treatment_mdr(),
            self.treatment_xdr(),
            self.treatment_prexdr(),
            self.bedaquiline_indication(),
            self.linezolid_indication()
        ]

    def mdr_detection(self):
        return Rule(
            rule_id='RC001',
            priority=1,
            conditions={'rifampin_resistance': True, 'isoniazid_resistance': True},
            actions={'classify': 'MDR', 'alert': 'MDR_protocol'},
            source='WHO 2022 Guidelines'
        )

    # XDR and pre-XDR use the pre-2021 (2006) injectable-based WHO definitions, not
    # the current Group A based ones, because the data carries no bedaquiline or
    # linezolid phenotypes the 2021 definition needs. Deliberate. See README Limitations.

    def xdr_detection(self):
        return Rule(
            rule_id='RC002',
            priority=1,
            conditions={'mdr': True, 'fluoroquinolone_resistance': True, 'injectable_resistance': True},
            actions={'classify': 'XDR', 'alert': 'XDR_protocol'},
            source='WHO pre-2021 (2006) XDR definition'
        )

    def prexdr_detection(self):
        return Rule(
            rule_id='RC003',
            priority=1,
            conditions={'mdr': True, 'fluoroquinolone_or_injectable': True},
            actions={'classify': 'PreXDR', 'alert': 'PreXDR_protocol'},
            source='WHO pre-2021 (informal) pre-XDR definition'
        )

    def treatment_mdr(self):
        return Rule(
            rule_id='TS002',
            priority=2,
            conditions={'mdr': True, 'xdr': False, 'fluoroquinolone_resistance': False},
            actions={
                'regimen': 'BPaLM',
                'drugs': list(BPALM_DRUGS),
                'duration': '6 months'
            },
            source='WHO 2022 MDR Guidelines'
        )

    def treatment_xdr(self):
        return Rule(
            rule_id='TS003',
            priority=2,
            conditions={'xdr': True},
            actions={
                'regimen': 'BPaL',
                'drugs': list(BPAL_DRUGS),
                'duration': '6-9 months'
            },
            source='WHO 2022 XDR Guidelines'
        )

    def treatment_prexdr(self):
        return Rule(
            rule_id='TS008',
            priority=2,
            conditions={'mdr': True, 'xdr': False, 'fluoroquinolone_resistance': True},
            actions={
                'regimen': 'BPaL',
                'drugs': list(BPAL_DRUGS),
                'duration': '6 months'
            },
            source='WHO 2022 Guidelines'
        )

    def bedaquiline_indication(self):
        return Rule(
            rule_id='TS004',
            priority=3,
            conditions={'fluoroquinolone_resistance': True},
            actions={'include': 'bedaquiline', 'rationale': 'FQ resistance'},
            source='WHO 2022 Guidelines'
        )

    def linezolid_indication(self):
        return Rule(
            rule_id='TS005',
            priority=3,
            conditions={'xdr': True},
            actions={'include': 'linezolid'},
            source='WHO 2022 Guidelines'
        )

    def evaluate_strain(self, strain_id, mode='forward', goal=None):
        facts = self.facts(strain_id)
        self.working_memory = facts
        self.fired = []

        if mode == 'backward' and goal:
            results = self.backward_chain(goal)
        else:
            results = self.forward_chain()

        return {
            'strain': strain_id,
            'recommendations': results,
            'rules_fired': self.fired,
            'canonical_gene_fraction': self.canonical_gene_fraction(facts['mutations'])
        }

    def facts(self, strain_id):
        if strain_id.startswith('P'):
            mapping = self.ontology.patient_strain_mapping(strain_id)
            if not mapping:
                return {'strain_id': strain_id, 'mutations': []}
            strain_id = mapping[0]['strain']

        mutations = self.ontology.strain_mutations_detailed(strain_id)
        facts = self.base_facts(strain_id, mutations)
        facts.update(self.mutation_flags(mutations))
        facts['fluoroquinolone_or_injectable'] = (
            facts['fluoroquinolone_resistance'] or facts['injectable_resistance'])
        return facts

    def base_facts(self, strain_id, mutations):
        flags = ['rifampin_resistance', 'isoniazid_resistance', 'fluoroquinolone_resistance',
                 'injectable_resistance', 'mdr_classified', 'xdr_classified',
                 'prexdr_classified', 'high_resistance', 'gyrA_mutation', 'rrs_mutation',
                 'katG_315_mutation']
        facts = {flag: False for flag in flags}
        facts['strain_id'] = strain_id
        facts['mutations'] = mutations
        return facts

    def mutation_flags(self, mutations):
        flags = {}
        for mut in mutations:
            flag = DRUG_FLAG.get(mut.get('drug'))
            if flag:
                flags[flag] = True
            gene = mut.get('gene')
            if gene == 'katG' and mutation_position(mut) == KATG_HIGH_CONFIDENCE:
                flags['katG_315_mutation'] = True
                flags['high_resistance'] = True
            if gene == 'gyrA':
                flags['gyrA_mutation'] = True
            if gene == 'rrs':
                flags['rrs_mutation'] = True
        return flags

    def forward_chain(self):
        """Fire every matching rule, repeating while the last pass changed the
        working memory. A rule fires once, so the passes are bounded by the rule
        count. A fixed bound smaller than that would silently drop the tail of a
        deeper chain if one were added."""
        recommendations = self.empty_recommendations()
        by_priority = sorted(self.rules, key=lambda r: r.priority)

        changed = True
        passes = 0
        while changed and passes < len(by_priority):
            changed = False
            passes += 1

            for rule in by_priority:
                if rule.id in self.fired:
                    continue

                if self.match(rule):
                    self.fire(rule, recommendations)
                    self.fired.append(rule.id)
                    changed = True

        self.resolve_classification(recommendations)
        self.direct_exclusions(recommendations)
        self.class_exclusions(recommendations)
        self.regimen_monitoring(recommendations)
        return recommendations

    def resolve_classification(self, recommendations):
        """Keep the most severe classification. The protocol alerts belonging to
        the ones it supersedes go with it, and alerts of any other kind are left
        alone rather than being filtered out with them."""
        classes = recommendations['classifications']
        if not classes:
            return
        top = max(classes, key=lambda c: CLASS_SEVERITY.get(c['type'], 0))
        superseded = {f"{c['type']}_protocol" for c in classes if c is not top}
        recommendations['classifications'] = [top]
        recommendations['alerts'] = [a for a in recommendations['alerts']
                                     if a.get('type') not in superseded]

    def empty_recommendations(self):
        self.excluded = set()
        return {'classifications': [], 'exclusions': [], 'alerts': [],
                'regimens': [], 'monitoring': [], 'inclusions': []}

    def direct_exclusions(self, recommendations):
        if not self.working_memory.get('mutations'):
            return
        for mut in self.working_memory['mutations']:
            self.add_exclusion(recommendations, mut.get('drug'), 'DIRECT_RESISTANCE',
                                f"mutation_{mut.get('mutation', 'detected')}")

    def class_exclusions(self, recommendations):
        for flag, (label, drugs) in DRUG_CLASSES.items():
            if not self.working_memory.get(flag):
                continue
            for drug in drugs:
                self.add_exclusion(recommendations, drug, 'CLASS_CROSS_RESISTANCE',
                                    f'{label}_cross_resistance')

    def match(self, rule):
        for condition, value in rule.conditions.items():
            fact = f'{condition}_classified' if condition in CLASSIFICATION_GOALS else condition
            if self.working_memory.get(fact, False) != value:
                return False
        return True

    def fire(self, rule, recommendations):
        for action, value in rule.actions.items():
            if action == 'classify':
                self.fire_classify(rule, value, recommendations)
            elif action == 'alert':
                self.fire_alert(rule, value, recommendations)
            elif action == 'regimen':
                self.fire_regimen(rule, value, recommendations)
            elif action == 'include':
                self.fire_include(rule, value, recommendations)

    def fire_classify(self, rule, value, recommendations):
        self.working_memory[f'{value.lower()}_classified'] = True
        recommendations['classifications'].append({
            'type': value, 'rule': rule.id, 'source': rule.source, 'confidence': rule.confidence})

    def add_exclusion(self, recommendations, drug, rule_id, reason):
        """First reason for excluding a drug wins. Membership is read from a set
        rather than by scanning the list, which was rescanning every entry on
        every call and is the one quadratic step on the CRyPTIC path."""
        if not drug or drug in self.excluded:
            return
        self.excluded.add(drug)
        recommendations['exclusions'].append({'drug': drug, 'rule': rule_id, 'reason': reason})

    def fire_alert(self, rule, value, recommendations):
        recommendations['alerts'].append({'type': value, 'rule': rule.id})

    def fire_regimen(self, rule, value, recommendations):
        recommendations['regimens'].append({
            'name': value, 'drugs': rule.actions.get('drugs', []),
            'duration': rule.actions.get('duration'), 'rule': rule.id, 'source': rule.source})

    def fire_include(self, rule, value, recommendations):
        recommendations['inclusions'].append({
            'drug': value, 'rationale': rule.actions.get('rationale'), 'rule': rule.id})

    def regimen_monitoring(self, recommendations):
        drugs = {d for r in recommendations['regimens'] for d in r.get('drugs', [])}
        schedule = [
            ('bedaquiline', 'ECG monthly', 'QTc >500ms', 'TS011'),
            ('linezolid', 'CBC monthly', 'myelosuppression', 'TS005'),
            ('pyrazinamide', 'LFTs monthly', 'ALT >3x ULN', 'TS010')
        ]
        existing = {m['parameter'] for m in recommendations['monitoring']}
        for drug, parameter, threshold, rule_id in schedule:
            if drug in drugs and parameter not in existing:
                recommendations['monitoring'].append(
                    {'parameter': parameter, 'threshold': threshold, 'rule': rule_id})

    def canonical_gene_fraction(self, mutations):
        # Share of distinct mutations whose gene is a canonical resistance gene.
        # This reads gene membership only, not the WHO grading tier. One row per
        # mutation-drug edge arrives here, so distinct mutations keep it stable.
        genes = {(mut.get('gene'), mut.get('mutation')): mut.get('gene') for mut in mutations}
        if not genes:
            return 0.0
        high = sum(gene in HIGH_CONF_GENES for gene in genes.values())
        return round(high / len(genes), 2)

    def backward_chain(self, goal):
        recommendations = self.empty_recommendations()

        if goal == 'treatment':
            self.backward_treatment(recommendations)
        elif goal == 'classification':
            self.backward_classification(recommendations)

        self.direct_exclusions(recommendations)
        self.class_exclusions(recommendations)
        self.regimen_monitoring(recommendations)
        return recommendations

    def backward_treatment(self, recommendations):
        detected = next((rule for goal, rule in (
            ('xdr', self.xdr_detection), ('prexdr', self.prexdr_detection),
            ('mdr', self.mdr_detection)) if self.prove_goal(goal)), None)
        if detected:
            self.fire(detected(), recommendations)
            self.fire_treatment(recommendations)

        if self.working_memory.get('fluoroquinolone_resistance'):
            self.fire(self.bedaquiline_indication(), recommendations)
        if self.working_memory.get('xdr_classified'):
            self.fire(self.linezolid_indication(), recommendations)

    def fire_treatment(self, recommendations):
        """First treatment rule whose conditions hold. Reading the same rules
        forward chaining uses keeps the two paths from drifting apart, which a
        second copy of the conditions in Python would not."""
        for rule in (self.treatment_xdr(), self.treatment_prexdr(),
                     self.treatment_mdr()):
            if self.match(rule):
                self.fire(rule, recommendations)
                return

    def backward_classification(self, recommendations):
        if self.prove_goal('xdr'):
            self.fire(self.xdr_detection(), recommendations)
        elif self.prove_goal('prexdr'):
            self.fire(self.prexdr_detection(), recommendations)
        elif self.prove_goal('mdr'):
            self.fire(self.mdr_detection(), recommendations)

    def prove_goal(self, goal):
        if goal == 'xdr':
            return self.prove_xdr()
        if goal == 'prexdr':
            return self.prove_prexdr()
        if goal == 'mdr':
            return self.prove_mdr()
        return False

    def prove_mdr(self):
        if self.working_memory.get('rifampin_resistance') and self.working_memory.get('isoniazid_resistance'):
            self.mark_classified('mdr', 'RC001')
            return True
        return False

    def prove_prexdr(self):
        if self.prove_mdr() and self.working_memory.get('fluoroquinolone_or_injectable'):
            self.mark_classified('prexdr', 'RC003')
            return True
        return False

    def prove_xdr(self):
        fq_and_inj = (self.working_memory.get('fluoroquinolone_resistance')
                      and self.working_memory.get('injectable_resistance'))
        if self.prove_mdr() and fq_and_inj:
            self.mark_classified('xdr', 'RC002')
            return True
        return False

    def mark_classified(self, key, rule_id):
        self.working_memory[f'{key}_classified'] = True
        if rule_id not in self.fired:
            self.fired.append(rule_id)