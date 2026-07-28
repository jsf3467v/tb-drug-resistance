import json
import os
import re
import time

import anthropic
from anthropic import Anthropic

from cbr_cases import DEFAULT_CASES
from cbr_engine import DEFAULT_NEIGHBORS, CaseStore, CBREngine
from config import DRUG_ALIASES, EXAMPLES, SCHEMA
from rule_engine import RuleEngine

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
TREATMENT_WORDS = (
    'recommend', 'treat', 'should', 'prescribe', 'therapy', 'regimen', 'monitor',
    'safe', 'contraindication', 'best', 'suggest', 'exclude', 'avoid', 'drug',
    'medication', 'receive')
# Only these name a classification on their own. The rest need an id in the
# question, so they stay separate rather than folded into a superset.
CLASSIFY_WORDS = ('classification', 'classify', 'profile', 'type')
RESISTANCE_WORDS = ('mdr', 'xdr', 'prexdr', 'resistant', 'resistance')
RISK_WORDS = ('risk', 'likely', 'probability', 'chance', 'predict')

STRAIN_ID = re.compile(r'TB\d{3}\b')
PATIENT_ID = re.compile(r'P\d{3}\b')

RETRYABLE = tuple(c for c in (
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
RETURN_CLAUSE = re.compile(r'\breturn\b', re.IGNORECASE)


def runnable_cypher(cypher):
    """Drop a trailing ORDER BY that sorts on a raw variable when the final RETURN
    aggregates, since Memgraph keeps only projected aliases in scope there. Only
    the last clause is read, because an aggregate consumed by an earlier WITH
    leaves its ORDER BY legal. Dropping is answer preserving only without a LIMIT;
    with one the order picks which rows survive, so the query is left to fail
    loudly rather than quietly return a different set."""
    clauses = [m.start() for m in RETURN_CLAUSE.finditer(cypher)]
    if not clauses:
        return cypher

    low = cypher.lower()
    final = clauses[-1]
    if not any(agg in low[final:] for agg in AGGREGATES):
        return cypher

    cut = low.rfind('order by')
    if cut < final or '.' not in cypher[cut:] or 'limit' in low[cut:]:
        return cypher
    return cypher[:cut].rstrip()


class NLInterface:

    def __init__(self, ontology, api_key=None):
        self.ontology = ontology
        self.client = Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY'),
            timeout=REQUEST_TIMEOUT
        )
        self.schema = SCHEMA
        self.examples = EXAMPLES
        self.rule_engine = RuleEngine(ontology)
        self.cbr_engine = None
        self.cbr_cases = []
        self.last_question = ""
        self.cache = {}

    def model_text(self, prompt, max_tokens, temperature):
        """Cached model call. Backoff runs between attempts only, since a sleep
        after the last one delays the failure without buying another try."""
        key = (prompt, max_tokens, temperature)
        if key in self.cache:
            return self.cache[key]

        last = None
        for attempt in range(MAX_RETRIES):
            try:
                message = self.client.messages.create(
                    model=MODEL, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "user", "content": prompt}])
                text = first_text(message).strip()
                self.cache[key] = text
                return text
            except RETRYABLE as exc:
                last = exc
                if attempt + 1 < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE * (2 ** attempt))
        raise LLMUnavailable(f"model unreachable after {MAX_RETRIES} attempts") from last

    def generate_cypher(self, user_question):
        user_question = user_question.rstrip('.?!,;')
        cypher = self.model_text(self.cypher_prompt(user_question), max_tokens=1024, temperature=0)
        return runnable_cypher(canonical_drugs(self.unfenced(cypher)))

    def cypher_prompt(self, user_question):
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

    def unfenced(self, cypher):
        cypher = cypher.strip()
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(line for line in lines if not line.startswith("```"))
            if cypher.startswith("cypher"):
                cypher = cypher[6:]
        return cypher.strip()

    def validate_cypher(self, cypher):
        """The keyword and delimiter checks read the query with string literals
        removed, so a write keyword or bracket inside a literal is a value rather
        than a clause. The opening-clause check reads raw text, where no literal
        can precede the first keyword."""
        bare = unquoted(cypher)

        match = WRITE_PATTERN.search(bare.upper())
        if match:
            return False, f"Query contains forbidden keyword: {match.group(1)}"

        if not cypher.strip().upper().startswith(READ_STARTS) and 'UNANSWERABLE' not in cypher:
            return False, "Query must start with a read clause"

        if bare.count('(') != bare.count(')'):
            return False, "Unbalanced parentheses"

        if bare.count('[') != bare.count(']'):
            return False, "Unbalanced brackets"

        return True, None

    def execute_query(self, cypher, parameters=None):
        """Read-only execution of a generated query. Driver errors propagate as
        they are, since the caller already labels them and re-raising a bare
        Exception here only hid the type behind a second copy of the message."""
        return self.ontology.read_query(cypher, parameters)

    def needs_rules(self, question):
        """Goal for the rule engine to prove, or False. A strain id defaults to
        classification and a patient id to treatment; a keyword hit overrides that
        default, not another keyword, so the order below is the whole precedence."""
        q = question.lower()
        if any(p in q for p in LIST_PHRASES):
            return False

        strain = bool(STRAIN_ID.search(question))
        if not (strain or PATIENT_ID.search(question)):
            return 'classification' if any(w in q for w in CLASSIFY_WORDS) else False
        if any(w in q for w in TREATMENT_WORDS):
            return 'treatment'
        if any(w in q for w in CLASSIFY_WORDS + RESISTANCE_WORDS + RISK_WORDS):
            return 'classification'
        return 'classification' if strain else 'treatment'

    def strain_from_results(self, results):
        for result in results:
            for key in ['strain', 'strain_id']:
                if key in result and result[key] and str(result[key]).startswith('TB'):
                    return result[key]
        return None

    def strain_from_patient(self, patient_id):
        query = """
            MATCH (p:Patient {patient_id: $pid})-[:INFECTED_WITH]->(s:Strain)
            RETURN s.strain_id as strain_id
        """
        result = self.ontology.query(query, {'pid': patient_id})
        return result[0]['strain_id'] if result else None

    def strain_from_question(self):
        match = STRAIN_ID.search(self.last_question)
        if match:
            return match.group()

        match = PATIENT_ID.search(self.last_question)
        if match:
            return self.strain_from_patient(match.group())

        return None

    def strain_from_mutations(self, results):
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
        """Rule engine output for the strain behind these results. needs_rules
        returns the goal name, which is what backward chaining proves, so the
        question type is passed through. Without one the engine derives
        everything in a single forward pass."""
        if not results:
            return None

        strain_id = self.identify_strain(results)
        if not strain_id:
            return None

        goal = question_type or None
        result = self.rule_engine.evaluate_strain(
            strain_id, mode='backward' if goal else 'forward', goal=goal)

        return {
            'strain': strain_id,
            'recommendations': result['recommendations'],
            'canonical_gene_fraction': result['canonical_gene_fraction'],
            'rules_fired': result['rules_fired']
        }

    def init_cbr(self):
        store = CaseStore(self.ontology)
        self.cbr_cases = store.retrieve_cases(limit=DEFAULT_CASES)
        if self.cbr_cases:
            self.cbr_engine = CBREngine(self.cbr_cases)
        return len(self.cbr_cases)

    def patient_from_results(self, results):
        for result in results:
            if 'patient_id' in result and str(result['patient_id']).startswith('P'):
                return result['patient_id']
            if 'patient' in result and str(result['patient']).startswith('P'):
                return result['patient']
        return None

    def patient_from_question(self):
        match = PATIENT_ID.search(self.last_question)
        return match.group() if match else None

    def patient_data_query(self, patient_id):
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

        analysis = self.cbr_engine.recommend(query_case, k=DEFAULT_NEIGHBORS)
        analysis['explained_cases'] = self.cbr_engine.explanations(
            query_case, analysis['similar_cases'])
        return analysis

    def rule_lines(self, classifications, exclusions, regimens, monitoring, alerts):
        output = []
        output += self.classification_lines(classifications)
        output += self.regimen_lines(regimens)
        output += self.exclusion_lines(exclusions)
        output += self.monitoring_lines(monitoring)
        output += self.alert_lines(alerts)
        return '\n'.join(output) if output else "No specific recommendations generated."

    def classification_lines(self, classifications):
        if not classifications:
            return []
        lines = ["Classifications:"]
        lines += [f"  - {c['type']} (Rule: {c['rule']}, Source: {c['source']})" for c in classifications]
        return lines

    def regimen_lines(self, regimens):
        if not regimens:
            return []
        lines = ["\nTreatment Regimens:"]
        for r in regimens:
            lines.append(f"  - {r['name']}: {', '.join(r['drugs'])}")
            lines.append(f"    Duration: {r['duration']} (Rule: {r['rule']})")
        return lines

    def exclusion_lines(self, exclusions):
        if not exclusions:
            return []
        lines = ["\nDrug Exclusions:"]
        lines += [f"  - Exclude {e['drug']} (Reason: {e['reason']}, Rule: {e['rule']})" for e in exclusions]
        return lines

    def monitoring_lines(self, monitoring):
        if not monitoring:
            return []
        lines = ["\nMonitoring Required:"]
        for m in monitoring:
            lines.append(f"  - {m['parameter']}")
            if m.get('threshold'):
                lines.append(f"    Threshold: {m['threshold']}")
        return lines

    def alert_lines(self, alerts):
        if not alerts:
            return []
        lines = ["\nClinical Alerts:"]
        lines += [f"  - {a['type']} (Rule: {a['rule']})" for a in alerts]
        return lines

    def rule_context(self, rule_output):
        if not rule_output:
            return ""

        recs = rule_output['recommendations']
        formatted = self.rule_lines(
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

        prompt = self.answer_prompt(user_question, cypher, results, rule_output, cbr_output)
        try:
            return self.model_text(prompt, max_tokens=2048, temperature=0.3)
        except LLMUnavailable:
            return self.fallback_summary(results, rule_output, cbr_output)

    def answer_prompt(self, user_question, cypher, results, rule_output, cbr_output):
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
5. A drug with no resistance mutation in the graph is untested, not susceptible. Roughly a fifth of measured resistance carries no graded mutation, so never describe such a drug as susceptible, safe, or an available option, and do not list them as alternatives
6. If case-based reasoning provided, mention similar case evidence
7. Note if results were truncated (showing first 20 of {len(results)})

Use plain text with no emoji and no decorative symbols. Keep the response concise and professional. Use bullet points for lists."""

    def fallback_summary(self, results, rule_output, cbr_output):
        parts = [f"Results returned: {len(results)}. (Model formatting unavailable; showing structured findings.)"]
        rule_text = self.rule_context(rule_output).strip()
        cbr_text = self.cbr_context(cbr_output).strip()
        if rule_text:
            parts.append(rule_text)
        if cbr_text:
            parts.append(cbr_text)
        return "\n\n".join(parts)