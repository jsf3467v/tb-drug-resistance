HIGH_CONF_GENES = ('rpoB', 'katG', 'inhA', 'embB', 'pncA', 'gyrA')

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
        self.rules = []
        self.fired = []
        self.working_memory = {}
        self.mode = 'forward'

    def build_rules(self):
        self.rules = [
            self._mdr_detection(),
            self._xdr_detection(),
            self._prexdr_detection(),
            self._treatment_selection_mdr(),
            self._treatment_selection_xdr(),
            self._treatment_selection_prexdr(),
            self._bedaquiline_indication(),
            self._linezolid_indication()
        ]

    def _mdr_detection(self):
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

    def _xdr_detection(self):
        return Rule(
            rule_id='RC002',
            priority=1,
            conditions={'mdr': True, 'fluoroquinolone_resistance': True, 'injectable_resistance': True},
            actions={'classify': 'XDR', 'alert': 'XDR_protocol'},
            source='WHO pre-2021 (2006) XDR definition'
        )

    def _prexdr_detection(self):
        return Rule(
            rule_id='RC003',
            priority=1,
            conditions={'mdr': True, 'fluoroquinolone_or_injectable': True},
            actions={'classify': 'PreXDR', 'alert': 'PreXDR_protocol'},
            source='WHO pre-2021 (informal) pre-XDR definition'
        )

    def _treatment_selection_mdr(self):
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

    def _treatment_selection_xdr(self):
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

    def _treatment_selection_prexdr(self):
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

    def _mdr_treatment(self):
        """MDR-class regimen: fluoroquinolone-free BPaL when a fluoroquinolone is
        contraindicated, otherwise BPaLM."""
        if self.working_memory.get('fluoroquinolone_resistance'):
            return self._treatment_selection_prexdr()
        return self._treatment_selection_mdr()

    def _bedaquiline_indication(self):
        return Rule(
            rule_id='TS004',
            priority=3,
            conditions={'fluoroquinolone_resistance': True},
            actions={'include': 'bedaquiline', 'rationale': 'FQ resistance'},
            source='WHO 2022 Guidelines'
        )

    def _linezolid_indication(self):
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
        self.mode = mode

        if mode == 'backward' and goal:
            results = self._backward_chain(goal)
        else:
            results = self._forward_chain()

        return {
            'strain': strain_id,
            'recommendations': results,
            'rules_fired': self.fired,
            'canonical_gene_fraction': self._canonical_gene_fraction(facts['mutations'])
        }

    def facts(self, strain_id):
        if strain_id.startswith('P'):
            mapping = self.ontology.patient_strain_mapping(strain_id)
            if not mapping:
                return {'strain_id': strain_id, 'mutations': []}
            strain_id = mapping[0]['strain']

        mutations = self.ontology.strain_mutations_detailed(strain_id)
        facts = self._base_facts(strain_id, mutations)
        facts.update(self._mutation_flags(mutations))
        facts['fluoroquinolone_or_injectable'] = (
            facts['fluoroquinolone_resistance'] or facts['injectable_resistance'])
        return facts

    def _base_facts(self, strain_id, mutations):
        flags = ['rifampin_resistance', 'isoniazid_resistance', 'fluoroquinolone_resistance',
                 'injectable_resistance', 'mdr_classified', 'xdr_classified',
                 'prexdr_classified', 'high_resistance', 'gyrA_mutation', 'rrs_mutation',
                 'katG_315_mutation']
        facts = {flag: False for flag in flags}
        facts['strain_id'] = strain_id
        facts['mutations'] = mutations
        return facts

    def _mutation_flags(self, mutations):
        drug_flag = {
            'rifampin': 'rifampin_resistance', 'isoniazid': 'isoniazid_resistance',
            'levofloxacin': 'fluoroquinolone_resistance', 'moxifloxacin': 'fluoroquinolone_resistance',
            'amikacin': 'injectable_resistance', 'kanamycin': 'injectable_resistance',
            'capreomycin': 'injectable_resistance'
        }
        flags = {}
        for mut in mutations:
            flag = drug_flag.get(mut.get('drug'))
            if flag:
                flags[flag] = True
            gene = mut.get('gene')
            position = str(mut.get('position', ''))
            if gene == 'katG' and (position == '315' or '315' in str(mut.get('mutation', ''))):
                flags['katG_315_mutation'] = True
                flags['high_resistance'] = True
            if gene == 'gyrA':
                flags['gyrA_mutation'] = True
            if gene == 'rrs':
                flags['rrs_mutation'] = True
        return flags

    def _forward_chain(self):
        changed = True
        recommendations = self._empty_recommendations()

        iterations = 0
        max_iterations = 5

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for rule in sorted(self.rules, key=lambda r: r.priority):
                if rule.id in self.fired:
                    continue

                if self._match(rule):
                    self._fire(rule, recommendations)
                    self.fired.append(rule.id)
                    changed = True

        self._resolve_classification(recommendations)
        self._direct_exclusions(recommendations)
        self._class_exclusions(recommendations)
        self._regimen_monitoring(recommendations)
        return recommendations

    def _resolve_classification(self, recommendations):
        severity = {'MDR': 1, 'PreXDR': 2, 'XDR': 3}
        classes = recommendations['classifications']
        if len(classes) <= 1:
            return
        top = max(classes, key=lambda c: severity.get(c['type'], 0))
        keep_alert = f"{top['type']}_protocol"
        recommendations['classifications'] = [top]
        recommendations['alerts'] = [a for a in recommendations['alerts']
                                     if a.get('type') == keep_alert]

    def _empty_recommendations(self):
        return {'classifications': [], 'exclusions': [], 'alerts': [],
                'regimens': [], 'monitoring': [], 'inclusions': []}

    def _direct_exclusions(self, recommendations):
        if not self.working_memory.get('mutations'):
            return
        for mut in self.working_memory['mutations']:
            self._add_exclusion(recommendations, mut.get('drug'), 'DIRECT_RESISTANCE',
                                f"mutation_{mut.get('mutation', 'detected')}")

    def _class_exclusions(self, recommendations):
        drug_classes = {
            'fluoroquinolone_resistance': ('fluoroquinolone', ['levofloxacin', 'moxifloxacin']),
            'injectable_resistance': ('injectable', ['amikacin', 'kanamycin', 'capreomycin'])
        }
        for flag, (label, drugs) in drug_classes.items():
            if not self.working_memory.get(flag):
                continue
            for drug in drugs:
                self._add_exclusion(recommendations, drug, 'CLASS_CROSS_RESISTANCE',
                                    f'{label}_cross_resistance')

    def _match(self, rule):
        for condition, value in rule.conditions.items():
            fact_value = self.working_memory.get(condition)

            if condition == 'mdr':
                fact_value = self.working_memory.get('mdr_classified', False)
            elif condition == 'xdr':
                fact_value = self.working_memory.get('xdr_classified', False)
            elif condition == 'prexdr':
                fact_value = self.working_memory.get('prexdr_classified', False)

            if fact_value != value:
                return False
        return True

    def _fire(self, rule, recommendations):
        for action, value in rule.actions.items():
            if action == 'classify':
                self._fire_classify(rule, value, recommendations)
            elif action == 'alert':
                self._fire_alert(rule, value, recommendations)
            elif action == 'regimen':
                self._fire_regimen(rule, value, recommendations)
            elif action == 'include':
                self._fire_include(rule, value, recommendations)

    def _fire_classify(self, rule, value, recommendations):
        self.working_memory[f'{value.lower()}_classified'] = True
        recommendations['classifications'].append({
            'type': value, 'rule': rule.id, 'source': rule.source, 'confidence': rule.confidence})

    def _add_exclusion(self, recommendations, drug, rule_id, reason):
        if not drug or any(e['drug'] == drug for e in recommendations['exclusions']):
            return
        recommendations['exclusions'].append({'drug': drug, 'rule': rule_id, 'reason': reason})

    def _fire_alert(self, rule, value, recommendations):
        recommendations['alerts'].append({'type': value, 'rule': rule.id})

    def _fire_regimen(self, rule, value, recommendations):
        recommendations['regimens'].append({
            'name': value, 'drugs': rule.actions.get('drugs', []),
            'duration': rule.actions.get('duration'), 'rule': rule.id, 'source': rule.source})

    def _fire_include(self, rule, value, recommendations):
        if 'inclusions' not in recommendations:
            recommendations['inclusions'] = []
        recommendations['inclusions'].append({
            'drug': value, 'rationale': rule.actions.get('rationale'), 'rule': rule.id})

    def _regimen_monitoring(self, recommendations):
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

    def _canonical_gene_fraction(self, mutations):
        # Share of distinct mutations whose gene is a canonical resistance gene.
        # This reads gene membership only, not the WHO grading tier. One row per
        # mutation-drug edge arrives here, so distinct mutations keep it stable.
        genes = {mut.get('mutation'): mut.get('gene') for mut in mutations}
        if not genes:
            return 0.0
        high = sum(gene in HIGH_CONF_GENES for gene in genes.values())
        return round(high / len(genes), 2)

    def _backward_chain(self, goal):
        recommendations = self._empty_recommendations()

        if goal == 'treatment':
            self._backward_treatment(recommendations)
        elif goal == 'classification':
            self._backward_classification(recommendations)

        self._direct_exclusions(recommendations)
        self._class_exclusions(recommendations)
        self._regimen_monitoring(recommendations)
        return recommendations

    def _backward_treatment(self, recommendations):
        if self._prove_goal('xdr'):
            self._fire(self._xdr_detection(), recommendations)
            self._fire(self._treatment_selection_xdr(), recommendations)
        elif self._prove_goal('prexdr'):
            self._fire(self._prexdr_detection(), recommendations)
            self._fire(self._mdr_treatment(), recommendations)
        elif self._prove_goal('mdr'):
            self._fire(self._mdr_detection(), recommendations)
            self._fire(self._mdr_treatment(), recommendations)

        if self.working_memory.get('fluoroquinolone_resistance'):
            self._fire(self._bedaquiline_indication(), recommendations)
        if self.working_memory.get('xdr_classified'):
            self._fire(self._linezolid_indication(), recommendations)

    def _backward_classification(self, recommendations):
        if self._prove_goal('xdr'):
            self._fire(self._xdr_detection(), recommendations)
        elif self._prove_goal('prexdr'):
            self._fire(self._prexdr_detection(), recommendations)
        elif self._prove_goal('mdr'):
            self._fire(self._mdr_detection(), recommendations)

    def _prove_goal(self, goal):
        if goal == 'xdr':
            return self._prove_xdr()
        if goal == 'prexdr':
            return self._prove_prexdr()
        if goal == 'mdr':
            return self._prove_mdr()
        return False

    def _prove_mdr(self):
        if self.working_memory.get('rifampin_resistance') and self.working_memory.get('isoniazid_resistance'):
            self._mark_classified('mdr', 'RC001')
            return True
        return False

    def _prove_prexdr(self):
        if self._prove_mdr() and self.working_memory.get('fluoroquinolone_or_injectable'):
            self._mark_classified('prexdr', 'RC003')
            return True
        return False

    def _prove_xdr(self):
        fq_and_inj = (self.working_memory.get('fluoroquinolone_resistance')
                      and self.working_memory.get('injectable_resistance'))
        if self._prove_mdr() and fq_and_inj:
            self._mark_classified('xdr', 'RC002')
            return True
        return False

    def _mark_classified(self, key, rule_id):
        self.working_memory[f'{key}_classified'] = True
        if rule_id not in self.fired:
            self.fired.append(rule_id)