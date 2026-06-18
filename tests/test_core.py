import ast
import sys
from pathlib import Path

import pytest

# Runs from Evaluation/. SRC/ and Evaluation/ are added to the import path so the core modules below resolve.

ROOT = Path(__file__).resolve().parent.parent
for _folder in (ROOT / "SRC", ROOT / "Evaluation"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

from rule_engine import RuleEngine
from cbr_cases import generate_cases
from cbr_engine import CBREngine, CaseRetriever, SimilarityCalculator, FEATURE_ORDER
from calibration import scaled_confidence, fit_temperature
import validation

SEVERITY = ["Susceptible", "MonoResistant", "PolyResistant", "MDR", "PreXDR", "XDR"]


class FakeOntology:
    """Stand-in so the rule engine runs without a database."""

    def __init__(self, mutations):
        self.mutations = mutations

    def patient_strain_mapping(self, strain_id):
        return None

    def strain_mutations_detailed(self, strain_id):
        return self.mutations


def mutation(drug=None, gene=None, mid="m", position=0):
    return {"mutation": mid, "gene": gene, "drug": drug, "position": position}


def evaluate(mutations, mode="forward", goal=None):
    """Build the rule engine on a fake ontology and evaluate one strain."""
    engine = RuleEngine(FakeOntology(mutations))
    engine.build_rules()
    return engine.evaluate_strain("TBX", mode=mode, goal=goal)


def classify(mutations):
    return [c["type"] for c in evaluate(mutations)["recommendations"]["classifications"]]


# rule engine. classification severity escalates correctly

def test_mdr_classification():
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "katG_S315T", 315)]
    assert classify(muts) == ["MDR"]


def test_prexdr_escalation():
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
            mutation("levofloxacin", "gyrA")]
    assert classify(muts) == ["PreXDR"]


def test_xdr_escalation():
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
            mutation("levofloxacin", "gyrA"), mutation("amikacin", "rrs")]
    assert classify(muts) == ["XDR"]


def test_no_classification_without_mdr():
    assert classify([mutation("isoniazid", "katG", "k", 315)]) == []


# rule engine. monitoring follows the regimen, not resistance

def test_monitoring_follows_regimen():
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315)]
    out = evaluate(muts)
    params = {m["parameter"] for m in out["recommendations"]["monitoring"]}
    assert "ECG monthly" in params   # bedaquiline is in BPaLM
    assert "CBC monthly" in params   # linezolid is in BPaLM


def test_backward_treatment_xdr():
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
            mutation("levofloxacin", "gyrA"), mutation("amikacin", "rrs")]
    out = evaluate(muts, mode="backward", goal="treatment")
    assert "BPaL" in [r["name"] for r in out["recommendations"]["regimens"]]


# rule engine. regimen never offers a contraindicated drug

def regimen_names(out):
    return [r["name"] for r in out["recommendations"]["regimens"]]


def regimen_drugs(out):
    return {d for r in out["recommendations"]["regimens"] for d in r.get("drugs", [])}


def test_prexdr_fq_uses_bpal_not_bpalm():
    # PreXDR by fluoroquinolone resistance. Moxifloxacin is contraindicated, so the
    # regimen must be BPaL, never the moxifloxacin-containing BPaLM.
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
            mutation("levofloxacin", "gyrA")]
    for mode, goal in (("forward", None), ("backward", "treatment")):
        out = evaluate(muts, mode=mode, goal=goal)
        assert "BPaL" in regimen_names(out) and "BPaLM" not in regimen_names(out), mode
        assert "moxifloxacin" not in regimen_drugs(out), mode


def test_prexdr_injectable_keeps_bpalm():
    # PreXDR by injectable resistance only. The fluoroquinolones remain usable, so BPaLM
    # is still appropriate.
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
            mutation("amikacin", "rrs")]
    out = evaluate(muts, mode="backward", goal="treatment")
    assert "BPaLM" in regimen_names(out)


def test_regimen_never_contains_excluded_drug():
    profiles = [
        [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315)],
        [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
         mutation("levofloxacin", "gyrA")],
        [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
         mutation("amikacin", "rrs")],
        [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
         mutation("levofloxacin", "gyrA"), mutation("amikacin", "rrs")],
    ]
    for muts in profiles:
        for mode, goal in (("forward", None), ("backward", "treatment")):
            out = evaluate(muts, mode=mode, goal=goal)
            excluded = {e["drug"] for e in out["recommendations"]["exclusions"]}
            assert not (regimen_drugs(out) & excluded), (mode, regimen_drugs(out) & excluded)


# rule engine. facts() derivation

def test_facts_flags():
    muts = [mutation("rifampin", "rpoB"), mutation("levofloxacin", "gyrA"),
            mutation("isoniazid", "katG", "katG_S315T", 315), mutation("amikacin", "rrs")]
    facts = RuleEngine(FakeOntology(muts)).facts("TBX")
    assert facts["rifampin_resistance"] and facts["isoniazid_resistance"]
    assert facts["fluoroquinolone_resistance"] and facts["injectable_resistance"]
    assert facts["fluoroquinolone_or_injectable"]
    assert facts["katG_315_mutation"] and facts["high_resistance"]
    assert facts["gyrA_mutation"] and facts["rrs_mutation"]


def test_facts_patient_without_mapping():
    facts = RuleEngine(FakeOntology([])).facts("P999")
    assert facts == {"strain_id": "P999", "mutations": []}


# calibration math

def test_scaled_confidence_identity():
    assert scaled_confidence(0.8, 1.0) == pytest.approx(0.8, abs=1e-6)


def test_scaled_confidence_softens():
    assert 0.5 < scaled_confidence(0.9, 3.0) < 0.9


def test_fit_temperature_degenerate():
    assert fit_temperature([0.8, 0.7], [1.0, 1.0]) == 1.0


def test_fit_temperature_overconfident():
    confidences = [0.9] * 100
    labels = [1.0] * 50 + [0.0] * 50
    assert fit_temperature(confidences, labels) > 1.0


# CBR engine. outcome_probability is the raw predicted success rate

@pytest.fixture(scope="module")
def base_cases():
    return generate_cases(300, seed=42)


def test_outcome_probability_is_raw_success_rate(base_cases):
    engine = CBREngine(base_cases)
    query = {"profile": "MDR", "hiv_status": "negative", "age": 45, "sex": "M",
             "region": "African", "diabetes": False, "previous_treatment": True}
    a = engine.recommend(dict(query))
    assert a["outcome_probability"] == pytest.approx(round(a["success_rate"], 3), abs=1e-9)
    assert 0.0 <= a["outcome_probability"] <= 1.0


def test_retrieve_exclude_id(base_cases):
    retriever = CaseRetriever(base_cases)
    excluded = base_cases[0]["case_id"]
    found = retriever.retrieve(base_cases[0], k=10, exclude_id=excluded)
    assert all(case.get("case_id") != excluded for _, case in found)


def test_vectorized_and_scalar_similarity_agree(base_cases):
    # scores() (vectorized, used for ranking) and the per-feature _*_similarity
    # functions (used by explain() for the UI breakdown) duplicate the same
    # weighted-similarity math. Pin them together so a change to one path that
    # is not mirrored in the other fails here, instead of silently making the
    # displayed breakdown disagree with the score that ranked the case.
    calc = SimilarityCalculator()
    calc.prepare(base_cases)
    for query in base_cases[:3]:
        vectorized = list(calc.scores(query))
        scalar = [sum(calc.feature_funcs[f](query, case) * calc.weights[f]
                      for f in FEATURE_ORDER)
                  for case in base_cases]
        assert vectorized == pytest.approx(scalar, abs=1e-9)


# generator. deterministic and covers all six profiles

def test_generator_deterministic():
    assert generate_cases(200, seed=7) == generate_cases(200, seed=7)


def test_generator_covers_six_profiles():
    profiles = {c["profile"] for c in generate_cases(1000, seed=42)}
    assert profiles == set(SEVERITY)


def test_susceptible_outperforms_xdr():
    cases = generate_cases(1000, seed=42)

    def success(profile):
        sub = [c for c in cases if c["profile"] == profile]
        return sum(c["outcome"] == "success" for c in sub) / len(sub)

    assert success("Susceptible") > success("XDR")


def test_regimen_mode_tracks_baseline():
    # The mode-of-neighbors predictor ignores outcome and takes the modal neighbor
    # regimen. It should recover accuracy the outcome-ranked recommender loses to
    # the objective mismatch and land closer to the majority baseline. Pins the
    # diagnostic so a retrieval change that stops separating the two effects fails here.
    cbr = validation.validate_cbr(generate_cases(1000, seed=42), k=5)
    outcome_ranked = cbr["regimen_accuracy"]["mean"]
    mode = cbr["regimen_mode_accuracy"]["mean"]
    baseline = cbr["baseline"]["regimen"]
    assert mode > outcome_ranked
    assert abs(baseline - mode) < abs(baseline - outcome_ranked)


def test_expert_queries_wellformed():
    queries = validation.expert_queries()
    ids = [q["id"] for q in queries]
    assert len(ids) == len(set(ids))                     # ids are unique
    for q in queries:
        assert q["question"] and q["category"]           # every query is labeled
        assert ("gold" in q) != bool(q.get("unanswerable"))  # exactly one of gold or unanswerable


# seed knowledge graph. static integrity (no database)

def seed_blobs():
    source = (ROOT / "SRC" / "tb_ontology.py").read_text()
    tree = ast.parse(source)
    blobs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("strain_data", "mutations"):
                    try:
                        blobs[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return blobs


def derived_profile(drugs):
    drugs = set(drugs)
    rif, inh = "rifampin" in drugs, "isoniazid" in drugs
    fq = {"levofloxacin", "moxifloxacin"} & drugs
    inj = {"amikacin", "kanamycin", "capreomycin"} & drugs
    if rif and inh and fq and inj:
        return "XDR"
    if rif and inh and (fq or inj):
        return "PreXDR"
    if rif and inh:
        return "MDR"
    first_line = len(drugs & {"rifampin", "isoniazid", "ethambutol", "pyrazinamide"})
    return "PolyResistant" if first_line > 1 else ("MonoResistant" if first_line else "Susceptible")


def test_seed_mutations_exist():
    blobs = seed_blobs()
    defined = {m["id"] for m in blobs["mutations"]}
    referenced = {mid for r in blobs["strain_data"] for mid in r["mutations"]}
    assert referenced <= defined


def test_stored_profiles_match_mutations():
    blobs = seed_blobs()
    drug = {m["id"]: m.get("drug") for m in blobs["mutations"]}
    mismatches = []
    for record in blobs["strain_data"]:
        drugs = {drug.get(m) for m in record["mutations"]}
        if derived_profile(drugs) != record["profile"]:
            mismatches.append((record["strain"], record["profile"], derived_profile(drugs)))
    assert mismatches == [], f"profile/mutation mismatches: {mismatches}"


# expert-system scoring. set match, empty results, negation leaks, unanswerable refusal

def test_same_answer_matches_equal_sets():
    gold = [{"strain": "TB002"}, {"strain": "TB003"}]
    produced = [{"s": "TB003", "lineage": 4}, {"s": "TB002", "lineage": 2}]
    assert validation.same_answer(gold, produced)        # row order and extra columns do not matter


def test_same_answer_rejects_row_count_changes():
    gold = [{"strain": "TB002"}, {"strain": "TB003"}]
    assert not validation.same_answer(gold, [{"strain": "TB002"}])    # a gold row is missing
    assert not validation.same_answer(gold, gold + [{"strain": "TB001"}])  # an extra row leaks in


class FakeNL:
    """Stand-in NL interface. execute_query returns the gold rows for the gold query
    and the produced rows otherwise, so evaluate_query exercises same_answer the way
    it does against a live graph."""

    def __init__(self, produced, gold=None, cypher="MATCH (s) RETURN s", valid=True):
        self.produced = produced
        self.gold = produced if gold is None else gold
        self.cypher = cypher
        self.valid = valid

    def generate_cypher(self, question):
        return self.cypher

    def validate_cypher(self, cypher):
        return self.valid, None

    def execute_query(self, cypher):
        return self.produced if cypher == self.cypher else self.gold


def query(test_id):
    return {q["id"]: q for q in validation.expert_queries()}[test_id]


def test_unanswerable_passes_when_refused():
    refused = FakeNL([], cypher="UNANSWERABLE: cannot edit the graph")
    assert validation.evaluate_query(query(10), refused)["passed"]
    answered = FakeNL([{"s": "TB001"}], cypher="MATCH (s) SET s.profile = 'Susceptible'")
    assert not validation.evaluate_query(query(10), answered)["passed"]


def test_empty_result_matches_empty_gold():
    result = validation.evaluate_query(query(9), FakeNL([], gold=[]))
    assert result["passed"]


def test_negation_leak_fails():
    excludes = FakeNL([{"strain": "TB002"}], gold=[{"strain": "TB002"}])
    assert validation.evaluate_query(query(6), excludes)["passed"]
    leaks = FakeNL([{"strain": "TB001"}, {"strain": "TB002"}], gold=[{"strain": "TB002"}])
    assert not validation.evaluate_query(query(6), leaks)["passed"]


# rule engine. class-level cross-resistance

def test_class_cross_resistance():
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "k", 315),
            mutation("levofloxacin", "gyrA"), mutation("amikacin", "rrs")]
    excluded = {e["drug"] for e in evaluate(muts)["recommendations"]["exclusions"]}
    assert {"levofloxacin", "moxifloxacin"} <= excluded
    assert {"amikacin", "kanamycin", "capreomycin"} <= excluded


def test_no_class_exclusion_when_no_class_resistance():
    muts = [mutation("isoniazid", "katG", "k", 315)]
    excluded = {e["drug"] for e in evaluate(muts)["recommendations"]["exclusions"]}
    assert "moxifloxacin" not in excluded
    assert "levofloxacin" not in excluded

# nl interface. read-only guard, query normalization, and routing (no database, no api)

@pytest.fixture(scope="module")
def nl_interface():
    from nl_interface import NLInterface
    return NLInterface(FakeOntology([]), api_key="test-key")


def test_write_guard_rejects_writes(nl_interface):
    for cypher in ("MATCH (n) DELETE n", "CREATE (n)", "MATCH (n) SET n.x = 1",
                   "MERGE (n)", "MATCH (n) DETACH DELETE n", "MATCH (n) REMOVE n.x"):
        assert not nl_interface.validate_cypher(cypher)[0], cypher


def test_write_guard_keyword_boundary(nl_interface):
    # a value that merely contains a keyword stays valid (asset holds SET)
    ok, _ = nl_interface.validate_cypher("MATCH (d:Drug) WHERE d.mechanism CONTAINS 'asset' RETURN d")
    assert ok


def test_read_guard_allows_read_clauses(nl_interface):
    for cypher in ("MATCH (n) RETURN n", "OPTIONAL MATCH (n) RETURN n",
                   "WITH 1 AS x RETURN x", "UNWIND [1, 2] AS x RETURN x"):
        ok, err = nl_interface.validate_cypher(cypher)
        assert ok and err is None, cypher


def test_unanswerable_passes_guard(nl_interface):
    assert nl_interface.validate_cypher("UNANSWERABLE: cannot edit the graph")[0]


def test_unbalanced_delimiters_rejected(nl_interface):
    assert not nl_interface.validate_cypher("MATCH (n RETURN n")[0]
    assert not nl_interface.validate_cypher("MATCH (n)-[r RETURN r")[0]


def test_canonical_drugs_rewrites_alias():
    from nl_interface import canonical_drugs
    out = canonical_drugs("MATCH (d:Drug {name: 'rifampicin'}) RETURN d")
    assert "'rifampin'" in out and "rifampicin" not in out


def test_runnable_cypher_drops_aggregate_orderby():
    from nl_interface import runnable_cypher
    agg = "MATCH (s:Strain) RETURN s.year AS y, count(s) AS n ORDER BY s.year"
    assert "order by" not in runnable_cypher(agg).lower()


def test_runnable_cypher_keeps_plain_orderby():
    from nl_interface import runnable_cypher
    plain = "MATCH (s:Strain) RETURN s.strain_id AS strain ORDER BY s.strain_id"
    assert runnable_cypher(plain) == plain


def test_needs_rules_routing(nl_interface):
    assert nl_interface.needs_rules("What treatment should patient P003 receive") == 'treatment'
    assert nl_interface.needs_rules("Classify strain TB001") == 'classification'
    assert nl_interface.needs_rules("Show all MDR strains") is False
    assert nl_interface.needs_rules("What mutations cause rifampin resistance") is False


def test_canonical_gene_fraction_counts_distinct_mutations():
    # one mutation confers resistance to several drugs, so it repeats once per drug
    # in the detailed view; the fraction counts the mutation once, not once per drug.
    rows = [mutation("amikacin", "rrs", "rrs_1401", 1401),
            mutation("kanamycin", "rrs", "rrs_1401", 1401),
            mutation("capreomycin", "rrs", "rrs_1401", 1401),
            mutation("rifampin", "rpoB", "rpoB_S450L", 450)]
    assert evaluate(rows)["canonical_gene_fraction"] == 0.5


def test_paren_in_literal_allowed(nl_interface):
    # a parenthesis inside a string literal must not fail the balance check
    ok, err = nl_interface.validate_cypher("MATCH (d:Drug) WHERE d.mechanism CONTAINS '(' RETURN d")
    assert ok and err is None


def test_needs_rules_ignores_four_digit_id(nl_interface):
    # P1000 is a four-digit case id and must not be read as the patient P100
    assert nl_interface.needs_rules("show case P1000") is False
