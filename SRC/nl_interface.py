import anthropic
from anthropic import Anthropic
import json
import os
import re
import time
from config import SCHEMA, EXAMPLES, DRUG_ALIASES
from rule_engine import RuleEngine
from cbr_engine import CBREngine, CaseStore

MODEL = "claude-sonnet-4-6"
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 4
BACKOFF_BASE = 0.5

# Creating clauses the read-only interface must reject. Matched on word boundaries
# so identifiers that contain a keyword (created, asset, offset) are not
# mistaken for the clause itself.
WRITE_KEYWORDS = ('DELETE', 'DETACH', 'DROP', 'CREATE', 'MERGE', 'SET', 'REMOVE')
WRITE_PATTERN = re.compile(r'\b(' + '|'.join(WRITE_KEYWORDS) + r')\b')

# Read clauses a generated query may open with. The write deny-list above is the
# real guard, so this only rejects a query that opens with nothing recognizable.
READ_STARTS = ('MATCH', 'OPTIONAL MATCH', 'WITH', 'UNWIND')

LIST_PHRASES = ('show all', 'list all', 'show mdr')
TREATMENT_KEYWORDS = (
    'recommend', 'treat', 'should', 'prescribe', 'therapy', 'regimen', 'monitor',
    'safe', 'contraindication', 'best', 'suggest', 'exclude', 'avoid', 'drug',
    'medication', 'receive')
CLASSIFICATION_KEYWORDS = (
    'classification', 'classify', 'profile', 'type', 'mdr', 'xdr', 'prexdr',
    'resistant', 'resistance')
CLASSIFY_WORDS = ('classification', 'classify', 'profile', 'type')
RISK_KEYWORDS = ('risk', 'likely', 'probability', 'chance', 'predict')

_RETRYABLE = tuple(c for c in (
    getattr(anthropic, 'APIConnectionError', None),
    getattr(anthropic, 'APITimeoutError', None),
    getattr(anthropic, 'RateLimitError', None),
    getattr(anthropic, 'InternalServerError', None),
) if c is not None) or (Exception,)


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached after retries, so callers can
    treat an infrastructure failure distinctly from a model that declined."""


def first_text(message):
    """First text block of an LLM response, tolerant of non-text blocks."""
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


def canonical_drugs(cypher):
    """Rewrite known drug-name variants in a query to the catalog spelling."""
    for variant, canonical in DRUG_ALIASES.items():
        for quote in ("'", '"'):
            cypher = cypher.replace(quote + variant + quote, quote + canonical + quote)
    return cypher


def unquoted(cypher):
    """Query with single and double quoted spans removed, so a delimiter inside a
    string literal does not count toward the parenthesis and bracket balance."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", "", cypher)


AGGREGATES = ('collect(', 'count(', 'sum(', 'avg(', 'min(', 'max(')


def runnable_cypher(cypher):
    """Remove a trailing ORDER BY that sorts on a raw variable when the RETURN
    aggregates. Memgraph keeps only the projected aliases in scope after an
    aggregate, so a key like s.year is unbound and the query fails to run. Order
    never changes the result set, so dropping the clause is answer preserving."""
    low = cypher.lower()
    if not any(agg in low for agg in AGGREGATES):
        return cypher
    cut = low.rfind('order by')
    if cut == -1 or '.' not in cypher[cut:]:
        return cypher
    limit = low.find('limit', cut)
    tail = ' ' + cypher[limit:].strip() if limit != -1 else ''
    return (cypher[:cut].rstrip() + tail).rstrip()


class NLInterface:
    """NL interface for TB drug resistance knowledge graph"""

    def __init__(self, ontology, api_key=None):
        self.ontology = ontology
        self.client = Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
            timeout=REQUEST_TIMEOUT
        )
        self.schema = SCHEMA
        self.examples = EXAMPLES
        self.rule_engine = RuleEngine(ontology)
        self.rule_engine.build_rules()
        self.cbr_engine = None
        self.cbr_cases = []
        self.last_question = ""
        self._cache = {}

    def _complete(self, prompt, max_tokens, temperature):
        key = (prompt, max_tokens, temperature)
        if key in self._cache:
            return self._cache[key]
        last = None
        for attempt in range(MAX_RETRIES):
            try:
                message = self.client.messages.create(
                    model=MODEL, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}])
                text = first_text(message).strip()
                self._cache[key] = text
                return text
            except _RETRYABLE as exc:
                last = exc
                time.sleep(BACKOFF_BASE * (2 ** attempt))
        raise LLMUnavailable(f"model unreachable after {MAX_RETRIES} attempts: {last}")

    def generate_cypher(self, user_question):
        user_question = user_question.rstrip('.?!,;')
        cypher = self._complete(self._cypher_prompt(user_question), max_tokens=1024, temperature=0)
        return runnable_cypher(canonical_drugs(self._strip_fences(cypher)))

    def _cypher_prompt(self, user_question):
        return f"""You are a Cypher query expert for a TB drug resistance database.

DATABASE SCHEMA:
{self.schema}

EXAMPLE QUERIES:
{self.examples}

RULES:
1. Return ONLY valid Cypher syntax
2. No explanations, no markdown, no code blocks
3. Use exact property names from schema
4. Inline literal values directly; do not use $parameter placeholders
5. Include ORDER BY for readable results
6. If question is impossible to answer, return: UNANSWERABLE: [reason]
7. For patient treatment questions, ALWAYS include patient_id in the RETURN clause
8. When RETURN uses an aggregate such as collect() or count(), ORDER BY must reference the output aliases, not raw variables like s.year

USER QUESTION: {user_question}

CYPHER QUERY:"""

    def _strip_fences(self, cypher):
        cypher = cypher.strip()
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(line for line in lines if not line.startswith("```"))
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        return cypher.strip()

    def validate_cypher(self, cypher):
        match = WRITE_PATTERN.search(cypher.upper())
        if match:
            return False, f"Query contains forbidden keyword: {match.group(1)}"

        if not cypher.strip().upper().startswith(READ_STARTS) and 'UNANSWERABLE' not in cypher:
            return False, "Query must start with a read clause"

        bare = unquoted(cypher)
        if bare.count('(') != bare.count(')'):
            return False, "Unbalanced parentheses"

        if bare.count('[') != bare.count(']'):
            return False, "Unbalanced brackets"

        return True, None

    def execute_query(self, cypher, parameters=None):
        try:
            return self.ontology.read_query(cypher, parameters)
        except Exception as e:
            raise Exception(f"Query execution error: {str(e)}")

    def needs_rules(self, question):
        q = question.lower()
        if any(p in q for p in LIST_PHRASES):
            return False

        has_id = bool(re.search(r'TB\d{3}\b', question) or re.search(r'P\d{3}\b', question))

        if has_id and any(kw in q for kw in TREATMENT_KEYWORDS):
            return 'treatment'
        if any(w in q for w in CLASSIFY_WORDS) or (has_id and any(kw in q for kw in CLASSIFICATION_KEYWORDS)):
            return 'classification'
        if has_id and any(kw in q for kw in RISK_KEYWORDS):
            return 'classification'
        if re.search(r'TB\d{3}\b', question):
            return 'classification'
        if re.search(r'P\d{3}\b', question):
            return 'treatment'
        return False

    def strain_from_results(self, results):
        """Extract strain ID from query results"""
        for result in results:
            for key in ['strain', 'strain_id']:
                if key in result and result[key] and str(result[key]).startswith('TB'):
                    return result[key]
        return None

    def strain_from_patient(self, patient_id):
        """Get strain ID from patient ID"""
        query = """
            MATCH (p:Patient {patient_id: $pid})-[:INFECTED_WITH]->(s:Strain)
            RETURN s.strain_id as strain_id
        """
        result = self.ontology.query(query, {'pid': patient_id})
        return result[0]['strain_id'] if result else None

    def strain_from_question(self):
        """Extract strain ID from question text"""
        match = re.search(r'TB\d{3}\b', self.last_question)
        if match:
            return match.group()

        match = re.search(r'P\d{3}\b', self.last_question)
        if match:
            return self.strain_from_patient(match.group())

        return None

    def strain_from_mutations(self, results):
        """Infer strain ID from mutations in results"""
        mutations = []
        for r in results:
            for key in ['mutation', 'mutation_id', 'mutations']:
                if key in r:
                    if isinstance(r[key], list):
                        mutations.extend(r[key])
                    else:
                        mutations.append(r[key])

        if not mutations:
            return None

        query = """
            MATCH (s:Strain)-[:HAS_MUTATION]->(m:Mutation)
            WHERE m.mutation_id IN $mutations
            RETURN s.strain_id as strain_id, count(m) as match_count
            ORDER BY match_count DESC
            LIMIT 1
        """
        result = self.ontology.query(query, {'mutations': mutations})
        return result[0]['strain_id'] if result else None

    def identify_strain(self, results):
        """Find strain ID using multiple strategies"""
        strain_id = self.strain_from_results(results)
        if strain_id:
            return strain_id

        for result in results:
            for key in ['patient', 'patient_id']:
                if key in result and str(result[key]).startswith('P'):
                    strain_id = self.strain_from_patient(result[key])
                    if strain_id:
                        return strain_id

        strain_id = self.strain_from_question()
        if strain_id:
            return strain_id

        return self.strain_from_mutations(results)

    def rule_recommend(self, results, question_type=False):
        if not results:
            return None

        strain_id = self.identify_strain(results)
        if not strain_id:
            return None

        mode = 'forward'
        goal = None

        if question_type == 'treatment':
            mode = 'backward'
            goal = 'treatment'
        elif question_type == 'classification':
            mode = 'backward'
            goal = 'classification'

        result = self.rule_engine.evaluate_strain(strain_id, mode=mode, goal=goal)

        return {
            'strain': strain_id,
            'recommendations': result['recommendations'],
            'canonical_gene_fraction': result['canonical_gene_fraction'],
            'rules_fired': result['rules_fired']
        }

    def init_cbr(self):
        store = CaseStore(self.ontology)
        self.cbr_cases = store.retrieve_cases(limit=1000)
        if self.cbr_cases:
            self.cbr_engine = CBREngine(self.cbr_cases)
        return len(self.cbr_cases)

    def patient_from_results(self, results):
        """Extract patient ID from query results"""
        for result in results:
            if 'patient_id' in result and str(result['patient_id']).startswith('P'):
                return result['patient_id']
            if 'patient' in result and str(result['patient']).startswith('P'):
                return result['patient']
        return None

    def patient_from_question(self):
        """Extract patient ID from question text"""
        match = re.search(r'P\d{3}\b', self.last_question)
        return match.group() if match else None

    def patient_data_query(self, patient_id):
        """Query patient data from database"""
        check_query = "MATCH (p:Patient {patient_id: $pid}) RETURN p.patient_id LIMIT 1"
        exists = self.ontology.query(check_query, {'pid': patient_id})

        if not exists:
            return None

        query = """
        MATCH (p:Patient {patient_id: $pid})-[:INFECTED_WITH]->(s:Strain)
        OPTIONAL MATCH (s)-[:HAS_PROFILE]->(r:ResistanceProfile)
        RETURN p.patient_id as patient_id, p.hiv_status as hiv_status, p.age as age, p.sex as sex,
               p.diabetes as diabetes, p.region as region,
               p.previous_treatment as previous_treatment, r.type as profile
        """
        result = self.ontology.query(query, {'pid': patient_id})
        return result[0] if result else None

    def cbr_recommend(self, results):
        if not self.cbr_engine or not results:
            return None

        patient_id = self.patient_from_results(results)
        if not patient_id:
            patient_id = self.patient_from_question()

        if not patient_id:
            return None

        patient_data = self.patient_data_query(patient_id)
        if not patient_data:
            return None

        query_case = {
            'profile': patient_data.get('profile') or 'Susceptible',
            'hiv_status': patient_data.get('hiv_status', 'negative'),
            'age': patient_data.get('age', 40),
            'sex': patient_data.get('sex', 'M'),
            'region': patient_data.get('region') or 'global',
            'diabetes': bool(patient_data.get('diabetes')),
            'previous_treatment': bool(patient_data.get('previous_treatment'))
        }

        return self.cbr_engine.recommend(query_case, k=10)

    def rule_output(self, classifications, exclusions, regimens, monitoring, alerts):
        """Format rule engine recommendations"""
        output = []
        output += self._fmt_classifications(classifications)
        output += self._fmt_regimens(regimens)
        output += self._fmt_exclusions(exclusions)
        output += self._fmt_monitoring(monitoring)
        output += self._fmt_alerts(alerts)
        return '\n'.join(output) if output else "No specific recommendations generated."

    def _fmt_classifications(self, classifications):
        if not classifications:
            return []
        lines = ["Classifications:"]
        lines += [f"  - {c['type']} (Rule: {c['rule']}, Source: {c['source']})" for c in classifications]
        return lines

    def _fmt_regimens(self, regimens):
        if not regimens:
            return []
        lines = ["\nTreatment Regimens:"]
        for r in regimens:
            lines.append(f"  - {r['name']}: {', '.join(r['drugs'])}")
            lines.append(f"    Duration: {r['duration']} (Rule: {r['rule']})")
        return lines

    def _fmt_exclusions(self, exclusions):
        if not exclusions:
            return []
        lines = ["\nDrug Exclusions:"]
        lines += [f"  - Exclude {e['drug']} (Reason: {e['reason']}, Rule: {e['rule']})" for e in exclusions]
        return lines

    def _fmt_monitoring(self, monitoring):
        if not monitoring:
            return []
        lines = ["\nMonitoring Required:"]
        for m in monitoring:
            lines.append(f"  - {m['parameter']}")
            if m.get('threshold'):
                lines.append(f"    Threshold: {m['threshold']}")
        return lines

    def _fmt_alerts(self, alerts):
        if not alerts:
            return []
        lines = ["\nClinical Alerts:"]
        lines += [f"  - {a['type']} (Rule: {a['rule']})" for a in alerts]
        return lines

    def rule_context(self, rule_output):
        """Build context section for rules"""
        if not rule_output:
            return ""

        recs = rule_output['recommendations']
        formatted = self.rule_output(
            recs.get('classifications', []), recs.get('exclusions', []),
            recs.get('regimens', []), recs.get('monitoring', []), recs.get('alerts', []))

        return f"""

EXPERT SYSTEM ANALYSIS:
Strain: {rule_output['strain']}
Canonical Gene Fraction: {rule_output['canonical_gene_fraction']}
Rules Applied: {', '.join(rule_output['rules_fired'])}

{formatted}
"""

    def cbr_context(self, cbr_output):
        """Build context section for CBR"""
        if not cbr_output:
            return ""

        top_regimens = ', '.join([r['regimen'] for r in cbr_output['recommendations'][:3]])

        return f"""

CASE-BASED REASONING:
Similar Cases: {len(cbr_output['similar_cases'])}
Success Rate: {cbr_output['success_rate']:.1%}
Top Recommendations: {top_regimens}
"""

    def format_results(self, user_question, cypher, results, rule_output=None, cbr_output=None):
        if not results:
            return "No results found for this query. The database may not contain relevant information for this question."

        prompt = self._format_prompt(user_question, cypher, results, rule_output, cbr_output)
        try:
            return self._complete(prompt, max_tokens=2048, temperature=0.3)
        except LLMUnavailable:
            return self._fallback_summary(results, rule_output, cbr_output)

    def _format_prompt(self, user_question, cypher, results, rule_output, cbr_output):
        display_results = results[:20] if len(results) > 20 else results
        rule_text = self.rule_context(rule_output)
        cbr_text = self.cbr_context(cbr_output)

        return f"""Format these database query results into a clear, professional answer.

USER QUESTION: {user_question}

QUERY EXECUTED: {cypher}

RESULTS: {json.dumps(display_results, indent=2)}

TOTAL RESULTS: {len(results)}
{rule_text}
{cbr_text}

Provide:
1. Direct answer to the question
2. Key findings from the data
3. Clinical significance if relevant
4. If expert system analysis provided, integrate the recommendations naturally. Treat every drug listed under exclusions as contraindicated - never present an excluded drug as an available or recommended treatment option
5. If case-based reasoning provided, mention similar case evidence
6. Note if results were truncated (showing first 20 of {len(results)})

Keep the response concise and professional. Use bullet points for lists."""

    def _fallback_summary(self, results, rule_output, cbr_output):
        parts = [f"Results returned: {len(results)}. (Model formatting unavailable; showing structured findings.)"]
        rule_text = self.rule_context(rule_output).strip()
        cbr_text = self.cbr_context(cbr_output).strip()
        if rule_text:
            parts.append(rule_text)
        if cbr_text:
            parts.append(cbr_text)
        return "\n\n".join(parts)