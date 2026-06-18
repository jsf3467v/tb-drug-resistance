import ast
import sys
from pathlib import Path

import pytest

# Runs from tests/. SRC/ and Evaluation/ are added to the import path so the core modules below resolve.

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


def test_query_check_types():
    by_id = {t["id"]: validation.query_check_type(t)
             for t in validation.standard_queries() + validation.edge_case_queries()}
    assert by_id[2] == "classification"    # requires_rules + expected_classification
    assert by_id[1] == "content_match"     # substring containment only
    assert by_id[104] == "empty_expected"  # min_results == 0
    assert by_id[105] == "absence"         # negation / expected_absent


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


# expert-system scoring. empty results, negation, measured cases

class FakeNL:
    """Stand-in NL interface returning canned Cypher and results."""

    def __init__(self, results, valid=True):
        self.results = results
        self.valid = valid
        self.last_question = None

    def generate_cypher(self, question):
        return "MATCH (s) RETURN s"

    def validate_cypher(self, cypher):
        return self.valid, None

    def execute_query(self, cypher):
        return self.results

    def needs_rules(self, question):
        return None

    def rule_recommend(self, results, qtype):
        return None


def edge(test_id):
    return {t["id"]: t for t in validation.edge_case_queries()}[test_id]


def test_empty_result_passes_when_none_expected():
    for test_id in (104, 107, 108):
        result = validation.evaluate_query(edge(test_id), FakeNL([]))
        assert result["passed"], test_id


def test_nonempty_fails_when_none_expected():
    result = validation.evaluate_query(edge(104), FakeNL([{"s": "TB001"}]))
    assert not result["passed"]


def test_negation_excludes_resistant():
    nl = FakeNL([{"strain_id": "TB002"}, {"strain_id": "TB052"}])
    assert validation.evaluate_query(edge(105), nl)["passed"]


def test_negation_detects_leak():
    nl = FakeNL([{"strain_id": "TB001"}, {"strain_id": "TB002"}])
    assert not validation.evaluate_query(edge(105), nl)["passed"]


def test_ambiguous_case_is_measured_not_scored():
    results = [validation.evaluate_query(t, FakeNL([{"x": "TB003"}]))
               for t in validation.edge_case_queries()]
    agg = validation.aggregate_expert_results(results)
    assert [m["id"] for m in agg["measured"]] == [103]
    assert agg["overall"]["n"] == len(validation.edge_case_queries()) - 1


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
