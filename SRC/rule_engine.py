from config import CLASS_LABELS, CROSS_RESISTANCE

HIGH_CONF_GENES = ('rpoB', 'katG', 'inhA', 'embB', 'pncA', 'gyrA')

BPAL_DRUGS = ['bedaquiline', 'pretomanid', 'linezolid']
BPALM_DRUGS = BPAL_DRUGS + ['moxifloxacin']

# Derived from config, so the engine grades a class exactly as wide as the label.
DRUG_FLAG = {'rifampin': 'rifampin_resistance', 'isoniazid': 'isoniazid_resistance'}
DRUG_FLAG.update({drug: f'{label}_resistance'
                  for label, members in CROSS_RESISTANCE.items() for drug in members})

# No BPaL drug is a class member, so this guard is inert today.
DRUG_FLAG.update({drug: f'{drug}_resistance' for drug in BPAL_DRUGS
                  if drug not in DRUG_FLAG})

# One member firing excludes the whole class. Sorted, because set order varies
# between processes.
DRUG_CLASSES = {f'{label}_resistance': (label, sorted(members - CLASS_LABELS))
                for label, members in CROSS_RESISTANCE.items()}

CLASS_SEVERITY = {'MDR': 1, 'PreXDR': 2, 'XDR': 3}

# Conditions naming a classification read the flag the classifier set.
CLASSIFICATION_GOALS = ('mdr', 'xdr', 'prexdr')

# RC003 implements a fixed external definition, so the pair is named rather than
# derived from CROSS_RESISTANCE, which would widen pre-XDR whenever a class is
# registered. They are also the only flags facts() indexes without a default.
PREXDR_CLASSES = ('fluoroquinolone_resistance', 'injectable_resistance')


class Rule:
    def __init__(self, rule_id, priority, conditions, actions, source):
        self.id = rule_id
        self.priority = priority
        self.conditions = conditions
        self.actions = actions
        self.source = source


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

    # XDR and pre-XDR use the pre-2021 (2006) injectable-based definitions, not
    # the current Group A based ones. Bedaquiline and linezolid phenotypes exist
    # in the release but are thin and weakly graded. See README Limitations.

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
            conditions={'fluoroquinolone_resistance': True, 'bedaquiline_resistance': False},
            actions={'include': 'bedaquiline', 'rationale': 'FQ resistance'},
            source='WHO 2022 Guidelines'
        )

    def linezolid_indication(self):
        return Rule(
            rule_id='TS005',
            priority=3,
            conditions={'xdr': True, 'linezolid_resistance': False},
            actions={'include': 'linezolid'},
            source='WHO 2022 Guidelines'
        )

    def evaluate_strain(self, strain_id, mode='forward', goal=None):
        results = self.strain_recommendations(strain_id, mode, goal)
        return {
            'strain': strain_id,
            'recommendations': results,
            'rules_fired': self.fired,
            'canonical_gene_fraction': self.canonical_gene_fraction(
                self.working_memory['mutations'])
        }

    def strain_recommendations(self, strain_id, mode='forward', goal=None):
        """Recommendations alone. Scoring reads only these and runs once per
        isolate per arm, so the interface fields stay off that path."""
        self.working_memory = self.facts(strain_id)
        self.fired = []
        if mode == 'backward' and goal:
            return self.backward_chain(goal)
        return self.forward_chain()

    def facts(self, strain_id):
        if strain_id.startswith('P'):
            mapping = self.ontology.patient_strain_mapping(strain_id)
            if not mapping:
                return {'strain_id': strain_id, 'mutations': []}
            strain_id = mapping[0]['strain']

        mutations = self.ontology.strain_mutations_detailed(strain_id)
        facts = self.base_facts(strain_id, mutations)
        facts.update(self.mutation_flags(mutations))
        facts['fluoroquinolone_or_injectable'] = any(facts[f] for f in PREXDR_CLASSES)
        return facts

    def base_facts(self, strain_id, mutations):
        facts = dict.fromkeys(PREXDR_CLASSES, False)
        facts['strain_id'] = strain_id
        facts['mutations'] = mutations
        return facts

    def mutation_flags(self, mutations):
        """Resistance flag per drug the mutations name."""
        flags = {}
        for mut in mutations:
            flag = DRUG_FLAG.get(mut.get('drug'))
            if flag:
                flags[flag] = True
        return flags

    def forward_chain(self):
        """Repeat while the last pass changed working memory. A rule fires once, so
        the rule count bounds the passes; a smaller fixed bound would silently
        truncate a deeper chain."""
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
                    changed = True

        self.resolve_classification(recommendations)
        self.reconcile(recommendations)
        return recommendations

    def reconcile(self, recommendations):
        """Shared by both inference paths. regimen_conflicts reads self.excluded,
        so the exclusion passes must run before it."""
        self.direct_exclusions(recommendations)
        self.class_exclusions(recommendations)
        self.regimen_conflicts(recommendations)
        self.regimen_monitoring(recommendations)

    def resolve_classification(self, recommendations):
        """Keep the most severe classification and drop the protocol alerts of the
        ones it supersedes. Alerts of any other kind survive."""
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

    def regimen_conflicts(self, recommendations):
        """Name the regimen drugs the isolate is resistant to. Annotates rather
        than drops, since choosing a substitute is a clinical decision."""
        for regimen in recommendations['regimens']:
            blocked = sorted(self.excluded.intersection(regimen['drugs']))
            if blocked:
                regimen['contraindicated'] = blocked

    def match(self, rule):
        for condition, value in rule.conditions.items():
            fact = f'{condition}_classified' if condition in CLASSIFICATION_GOALS else condition
            if self.working_memory.get(fact, False) != value:
                return False
        return True

    def fire(self, rule, recommendations):
        """Recorded here rather than at the call site, so a rule reached by either
        inference path lands in the audit trail the same way."""
        if rule.id not in self.fired:
            self.fired.append(rule.id)

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
        recommendations['classifications'].append(
            {'type': value, 'rule': rule.id, 'source': rule.source})

    def add_exclusion(self, recommendations, drug, rule_id, reason):
        """First reason wins. Membership reads a set, not a list scan, which was the
        one quadratic step on the CRyPTIC path. Class labels are skipped: a
        patient cannot be excluded from a class."""
        if not drug or drug in CLASS_LABELS or drug in self.excluded:
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
        # Reads gene membership only, not the WHO grading tier. Deduped, because one
        # row per mutation-drug edge arrives here and would double-count.
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

        self.reconcile(recommendations)
        return recommendations

    def backward_treatment(self, recommendations):
        detected = next((rule for goal, rule in (
            ('xdr', self.xdr_detection), ('prexdr', self.prexdr_detection),
            ('mdr', self.mdr_detection)) if self.prove_goal(goal)), None)
        if detected:
            self.fire(detected(), recommendations)
            self.fire_treatment(recommendations)

        for rule in (self.bedaquiline_indication(), self.linezolid_indication()):
            if self.match(rule):
                self.fire(rule, recommendations)

    def fire_treatment(self, recommendations):
        """First treatment rule whose conditions hold. Built from the same factory
        methods forward chaining uses, so the two paths cannot drift apart."""
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