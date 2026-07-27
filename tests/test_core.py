import ast
import json
import sys
from pathlib import Path

import pytest

# Runs from Evaluation/. SRC/ and Evaluation/ are added to the import path so the core modules below resolve.

ROOT = Path(__file__).resolve().parent.parent
for _folder in (ROOT / "SRC", ROOT / "Evaluation"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))

import cbr_engine
import rule_engine
import validation
from calibration import fit_temperature, scaled_confidence
from cbr_cases import case_base
from cbr_engine import FEATURE_ORDER, CaseRetriever, CBREngine, SimilarityCalculator
from rule_engine import RuleEngine

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


def test_non_protocol_alerts_survive_classification_resolution():
    # PreXDR and MDR both fire here. Only the superseded protocol alert should
    # be dropped, not every alert that is not the winner's.
    muts = [mutation(drug="rifampin", gene="rpoB"), mutation(drug="isoniazid", gene="katG"),
            mutation(drug="amikacin", gene="rrs")]
    out = evaluate(muts)["recommendations"]

    assert [c["type"] for c in out["classifications"]] == ["PreXDR"]
    kinds = {a["type"] for a in out["alerts"]}
    assert "MDR_protocol" not in kinds
    assert "PreXDR_protocol" in kinds


FORWARD_BACKWARD_CASES = [
    [mutation(drug="rifampin", gene="rpoB"), mutation(drug="isoniazid", gene="katG")],
    [mutation(drug="rifampin", gene="rpoB"), mutation(drug="isoniazid", gene="katG"),
     mutation(drug="levofloxacin", gene="gyrA")],
    [mutation(drug="rifampin", gene="rpoB"), mutation(drug="isoniazid", gene="katG"),
     mutation(drug="amikacin", gene="rrs")],
    [mutation(drug="rifampin", gene="rpoB"), mutation(drug="isoniazid", gene="katG"),
     mutation(drug="levofloxacin", gene="gyrA"), mutation(drug="amikacin", gene="rrs")],
]


@pytest.mark.parametrize("muts", FORWARD_BACKWARD_CASES)
def test_forward_and_backward_agree(muts):
    # Both modes are reachable from the interface, so they must not disagree on
    # the same strain. Backward chaining reads the same treatment rules now.
    forward = evaluate(muts)["recommendations"]
    backward = evaluate(muts, mode="backward", goal="treatment")["recommendations"]

    assert [c["type"] for c in forward["classifications"]] == \
           [c["type"] for c in backward["classifications"]]
    assert [r["name"] for r in forward["regimens"]] == [r["name"] for r in backward["regimens"]]


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


# CBR engine. success_rate is the one reported probability, shared by the
# interface and by validation, so no second copy can drift away from it.

@pytest.fixture(scope="module")
def base_cases():
    return case_base(300, seed=42)


MDR_QUERY = {"profile": "MDR", "hiv_status": "negative", "age": 45, "sex": "M",
             "region": "African", "diabetes": False, "previous_treatment": True}


def test_success_rate_is_the_only_probability(base_cases):
    engine = CBREngine(base_cases)
    a = engine.recommend(dict(MDR_QUERY))
    neighbors = a["similar_cases"]
    expected = sum(c["outcome"] == "success" for _, c in neighbors) / len(neighbors)

    assert a["success_rate"] == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= a["success_rate"] <= 1.0
    assert "outcome_probability" not in a
    assert "weighted_success_rate" not in a["outcome_analysis"]


def test_evidence_filter_runs_before_the_cut(base_cases):
    # A regimen with enough support must survive even when thinly supported
    # regimens outrank it, which slicing before filtering used to discard.
    engine = CBREngine(base_cases)
    engine.profile_regimens["MDR"] = {"One", "Two", "Three", "Supported"}
    neighbors = [(1.0, {"regimen": "One", "outcome": "success"}),
                 (1.0, {"regimen": "Two", "outcome": "success"}),
                 (1.0, {"regimen": "Three", "outcome": "success"}),
                 (1.0, {"regimen": "Supported", "outcome": "success"}),
                 (1.0, {"regimen": "Supported", "outcome": "failed"})]

    recs = engine.regimen_recommendations(neighbors, "MDR")
    assert [r["regimen"] for r in recs] == ["Supported"]


def test_recommendation_is_applicable_to_the_query_profile(base_cases):
    # Retrieval admits neighbours from adjacent profiles, so a regimen the case
    # base never pairs with the query profile could reach the top of the list.
    # BPaLM carries moxifloxacin, which a fluoroquinolone-resistant patient
    # cannot take, so this is a safety property rather than an accuracy one.
    engine = CBREngine(base_cases)
    pairs = {(c["profile"], c["regimen"]) for c in base_cases}

    for profile in ("Susceptible", "MonoResistant", "PolyResistant",
                    "MDR", "PreXDR", "XDR"):
        query = dict(MDR_QUERY, profile=profile)
        for rec in engine.recommend(query)["recommendations"]:
            assert (profile, rec["regimen"]) in pairs


def test_no_neighbors_falls_back_to_the_prior():
    # Every feature is opposed, so only the region floor scores and nothing
    # clears the cutoff. Reporting zero here would assert certain failure.
    far = {"profile": "Susceptible", "hiv_status": "negative", "age": 20,
           "region": "African", "diabetes": False, "previous_treatment": False,
           "sex": "M", "regimen": "2HRZE_4HR", "year": 2024}
    engine = CBREngine([dict(far, case_id="C1", outcome="success"),
                        dict(far, case_id="C2", outcome="failed")])
    query = {"profile": "XDR", "hiv_status": "positive", "age": 80,
             "region": "Americas", "diabetes": True, "previous_treatment": True, "sex": "F"}
    a = engine.recommend(query)

    assert a["similar_cases"] == []
    assert engine.prior_success == pytest.approx(0.5)
    assert a["success_rate"] == pytest.approx(0.5)
    assert a["recommendations"][0]["regimen"] == "BPaL"


def test_explanations_stay_out_of_recommend(base_cases):
    # The evaluation path reads none of this, so recommend must not build it.
    engine = CBREngine(base_cases)
    a = engine.recommend(dict(MDR_QUERY))
    assert "explained_cases" not in a

    explained = engine.explanations(dict(MDR_QUERY), a["similar_cases"])
    assert 0 < len(explained) <= cbr_engine.EXPLAINED_CASES
    assert {"case_id", "similarity", "feature_breakdown"} <= set(explained[0])


def test_explanation_ranks_by_contribution(base_cases):
    # Ranking on weight times similarity, not on a threshold, so the named
    # features are the ones that actually moved the score.
    calc = SimilarityCalculator(base_cases)
    query = dict(MDR_QUERY)
    for case in base_cases[:40]:
        ex = calc.explain(query, case)
        by_share = sorted(ex["breakdown"], key=lambda b: -b["contribution"])
        assert ex["top_matches"] == [b["feature"] for b in by_share[:3]]
        for feature in ex["key_differences"]:
            assert next(b for b in ex["breakdown"] if b["feature"] == feature)["similarity"] < 1.0


def test_region_mismatch_can_be_a_key_difference():
    # The region floor once sat exactly on the old cutoff, which silently made
    # a region mismatch unreportable however much similarity it cost.
    calc = SimilarityCalculator([])
    query = {"profile": "MDR", "hiv_status": "positive", "age": 40, "sex": "M",
             "region": "African", "diabetes": True, "previous_treatment": True}
    ex = calc.explain(query, dict(query, region="Americas"))

    assert ex["key_differences"] == ["region"]


def test_closeness_spans_the_admitted_range(base_cases):
    engine = CBREngine(base_cases)
    lowest = engine.confidence_calc.retrieval_score(cbr_engine.GOOD_CASE_COUNT,
                                                     cbr_engine.MIN_SIMILARITY)
    highest = engine.confidence_calc.retrieval_score(cbr_engine.GOOD_CASE_COUNT, 1.0)
    assert lowest == pytest.approx(0.5)
    assert highest == pytest.approx(1.0)

    scores = {engine.recommend(c, exclude_id=c["case_id"])["confidence"]["score"]
              for c in base_cases[:120]}
    assert len(scores) > 1


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
    calc = SimilarityCalculator(base_cases)
    for query in base_cases[:3]:
        vectorized = list(calc.scores(query))
        scalar = [sum(calc.feature_funcs[f](query, case) * calc.weights[f]
                      for f in FEATURE_ORDER)
                  for case in base_cases]
        assert vectorized == pytest.approx(scalar, abs=1e-9)


# generator. deterministic and covers all six profiles

def test_generator_deterministic():
    assert case_base(200, seed=7) == case_base(200, seed=7)


def test_generator_covers_six_profiles():
    profiles = {c["profile"] for c in case_base(1000, seed=42)}
    assert profiles == set(SEVERITY)


def test_susceptible_outperforms_xdr():
    cases = case_base(1000, seed=42)

    def success(profile):
        sub = [c for c in cases if c["profile"] == profile]
        return sum(c["outcome"] == "success" for c in sub) / len(sub)

    assert success("Susceptible") > success("XDR")


def test_matching_retries_a_greedy_dead_end():
    # A greedy pass takes the first produced row that fits and can strand the
    # next gold row, reporting a mismatch on input that does pair up.
    gold = [{"x": "a", "y": "b"}, {"x": "c", "y": "d"}]
    produced = [{"x": "a", "y": "b", "z": "c", "w": "d"}, {"x": "a", "y": "b"}]
    assert validation.same_answer(gold, produced)

    unmatchable = [{"x": "a", "y": "b", "z": "c"}, {"x": "a", "y": "b"}]
    assert not validation.same_answer(gold, unmatchable)


def test_journal_survives_a_crash_and_clears_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "EXPERT_CHECKPOINT", tmp_path / "journal.json")
    monkeypatch.setattr(validation, "EXPERT_QUERIES", [
        {"id": 1, "category": "unanswerable", "question": "q1", "unanswerable": True},
        {"id": 2, "category": "unanswerable", "question": "q2", "unanswerable": True},
    ])

    class Crashing:
        def __init__(self):
            self.seen = 0

        def generate_cypher(self, question):
            self.seen += 1
            if self.seen == 2:
                raise RuntimeError("network down")
            return "UNANSWERABLE: no"

        def validate_cypher(self, cypher):
            return False, "refused"

    crashing = Crashing()
    validation.validate_expert_system(crashing, resume=True)
    assert validation.EXPERT_CHECKPOINT.exists()          # the paid call is kept
    assert len(json.loads(validation.EXPERT_CHECKPOINT.read_text())["results"]) == 1

    validation.validate_expert_system(crashing, resume=True)
    assert not validation.EXPERT_CHECKPOINT.exists()      # a clean run leaves nothing


def test_no_predictor_exceeds_the_generator_ceiling():
    # The generator draws the regimen from profile and year alone, so the share
    # held by the most common option bounds every predictor. A score above it
    # means a held-out case reached the case base it was scored against.
    # This replaced an assertion that the mode predictor beats the outcome-ranked
    # one, which held only while the recommender could return a regimen that was
    # not applicable to the query profile.
    cbr = validation.validate_cbr(case_base(1000, seed=42), k=5)
    ceiling = cbr["ceiling"]

    assert cbr["baseline"]["regimen"] <= ceiling
    assert cbr["regimen_accuracy"]["mean"] <= ceiling
    assert cbr["regimen_mode_accuracy"]["mean"] <= ceiling
    assert cbr["abstentions"] < len(case_base(1000, seed=42)) // 100


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


def test_drug_aliases_copies_agree():
    # feature_engineering keeps its own copy so it imports nothing from the
    # system. Deliberate, but the two must not drift apart unnoticed.
    import config
    import feature_engineering

    assert config.DRUG_ALIASES == feature_engineering.DRUG_ALIASES


def test_canonical_drugs_rewrites_alias():
    from nl_interface import canonical_drugs
    out = canonical_drugs("MATCH (d:Drug {name: 'rifampicin'}) RETURN d")
    assert "'rifampin'" in out and "rifampicin" not in out


def test_prompt_examples_obey_their_own_order_by_rule():
    # The prompt tells the model that when RETURN aggregates, ORDER BY must name
    # output aliases. Two examples demonstrated the opposite, and few-shot
    # examples outweigh a prose rule, which is what runnable_cypher was patching
    # at runtime. Pins the examples so an edit cannot reintroduce the conflict.
    import re

    from config import EXAMPLES
    aggregates = ("collect(", "count(", "sum(", "avg(", "min(", "max(")
    offenders = []

    for block in [b for b in re.split(r"\n\s*\n", EXAMPLES) if "Cypher:" in b]:
        body = block.split("Cypher:", 1)[1]
        low = body.lower()
        start, order = low.rfind("return"), low.rfind("order by")
        if start == -1 or order < start:
            continue
        if any(a in low[start:order] for a in aggregates) and re.search(
                r"\b[a-z]\w*\.\w+", body[order:]):
            offenders.append(block.splitlines()[0])

    assert not offenders, f"ORDER BY names a raw variable after an aggregate: {offenders}"


def test_runnable_cypher_drops_aggregate_orderby():
    from nl_interface import runnable_cypher
    agg = "MATCH (s:Strain) RETURN s.year AS y, count(s) AS n ORDER BY s.year"
    assert "order by" not in runnable_cypher(agg).lower()


def test_runnable_cypher_keeps_plain_orderby():
    from nl_interface import runnable_cypher
    plain = "MATCH (s:Strain) RETURN s.strain_id AS strain ORDER BY s.strain_id"
    assert runnable_cypher(plain) == plain


def test_runnable_cypher_keeps_orderby_after_with_aggregate():
    # An aggregate consumed by a WITH leaves the RETURN unaggregated, so its
    # ORDER BY is legal. Examples 2 and 6 have this shape and were rewritten.
    from nl_interface import runnable_cypher
    query = ("MATCH (s:Strain)-[:HAS_MUTATION]->(m:Mutation)-[:CONFERS_RESISTANCE]->(x:Drug) "
             "WITH s, collect(DISTINCT x.name) AS resistant "
             "MATCH (d:Drug) WHERE NOT d.name IN resistant "
             "RETURN d.name AS drug, d.class AS drug_class ORDER BY d.class, d.name")
    assert runnable_cypher(query) == query


def test_runnable_cypher_leaves_prompt_examples_alone():
    import re

    from config import EXAMPLES
    from nl_interface import runnable_cypher

    blocks = [b.split("Cypher:", 1)[1].strip()
              for b in re.split(r"\nExample \d+:", EXAMPLES)[1:]]
    rewritten = [q.splitlines()[0] for q in blocks if runnable_cypher(q) != q]
    assert not rewritten, f"runtime patch rewrote a valid example: {rewritten}"


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

def test_inference_modes_agree_on_every_resistance_combination():
    # The evaluation scores forward and the application runs backward, so the
    # graded fields must agree. Classification withholds regimens on purpose.
    import itertools

    drugs = {"rifampin": "rpoB", "isoniazid": "katG",
             "levofloxacin": "gyrA", "amikacin": "rrs"}

    def graded(recommendations):
        return (
            [c["type"] for c in recommendations["classifications"]],
            sorted({e["drug"] for e in recommendations["exclusions"] if e["drug"]}),
            sorted(a["type"] for a in recommendations["alerts"]),
        )

    def prescribed(recommendations):
        return (sorted(r["name"] for r in recommendations["regimens"]),
                sorted(m["parameter"] for m in recommendations["monitoring"]))

    for size in range(len(drugs) + 1):
        for combination in itertools.combinations(drugs, size):
            muts = [mutation(d, drugs[d], f"{drugs[d]}_S315T", 315) for d in combination]
            forward = evaluate(muts)["recommendations"]
            treatment = evaluate(muts, mode="backward", goal="treatment")["recommendations"]
            classification = evaluate(muts, mode="backward",
                                      goal="classification")["recommendations"]

            assert graded(treatment) == graded(forward), combination
            assert graded(classification) == graded(forward), combination
            assert prescribed(treatment) == prescribed(forward), combination
            assert prescribed(classification) == ([], []), combination


def test_forward_chain_bound_holds_the_whole_rule_set():
    # Pins the bound, not the depth. Nothing currently chains that deep.
    engine = RuleEngine(FakeOntology([]))
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG", "katG_S315T", 315),
            mutation("levofloxacin", "gyrA"), mutation("amikacin", "rrs")]
    fired = evaluate(muts)["rules_fired"]
    assert len(fired) == len(set(fired))
    assert len(fired) <= len(engine.rules)


def test_row_values_counts_repeats():
    # Untagged sets let a narrower produced row satisfy a wider gold row. No
    # current gold query repeats a value, so this guards rather than fixes.
    assert validation.row_values({"a": 1, "b": 1}) != validation.row_values({"a": 1})
    assert not validation.covers([{"a": 1, "b": 1}], [{"x": 1, "y": 2}])
    assert validation.covers([{"a": 1, "b": 1}], [{"x": 1, "y": 1, "z": 2}])


def test_case_generator_is_closed():
    # A regimen missing a rate or a duration raises partway through a run
    # rather than at import, so the tables are checked against each other.
    from cbr_cases import BASE_SUCCESS, REGIMEN_DURATION, REGIMEN_OPTIONS, YEARS

    pairs = {(profile, regimen) for profile, years in REGIMEN_OPTIONS.items()
             for options in years.values() for regimen, _ in options}
    regimens = {regimen for _, regimen in pairs}

    assert pairs == set(BASE_SUCCESS)
    assert regimens == set(REGIMEN_DURATION)
    assert all(set(years) == {str(y) for y in YEARS} for years in REGIMEN_OPTIONS.values())
    assert all(weight > 0 for options in REGIMEN_OPTIONS.values()
               for year in options.values() for _, weight in year)


def test_profile_vocabulary_agrees_across_modules():
    # Five modules spell out the same six names. A partial rename shifts a
    # number instead of raising.
    import cbr_cases
    import cbr_engine
    import feature_engineering
    import rule_engine

    severity = set(feature_engineering.SEVERITY)
    assert set(cbr_cases.PROFILE_TARGETS) == severity
    assert set(cbr_cases.PROFILE_BASE_WEIGHT) == severity
    assert set(cbr_cases.REGIMEN_OPTIONS) == severity
    assert set(cbr_engine.PROFILE_RANK) == severity
    assert set(cbr_cases.MINOR_RESISTANCE) | set(cbr_cases.MAJOR_RESISTANCE) | {
        "Susceptible"} == severity
    assert set(validation.RESISTANT_TIERS) <= severity
    assert set(validation.COLLAPSED) == set(validation.RESISTANT_TIERS) | {"below-MDR"}
    assert set(rule_engine.CLASS_SEVERITY) == set(validation.RESISTANT_TIERS)


def test_mdr_tiers_do_not_depend_on_the_mono_poly_rule():
    # Mono and poly deviate from WHO by counting every resistant drug. This
    # holds the deviation below MDR, so no tier figure moves if it is revisited.
    import itertools

    from config import FIRST_LINE, FLUOROQUINOLONES, INJECTABLES
    from feature_engineering import profile

    pool = sorted(FIRST_LINE | {"levofloxacin", "amikacin", "ethionamide", "bedaquiline"})
    tiers = {"MDR", "PreXDR", "XDR"}

    for size in range(len(pool) + 1):
        for combination in itertools.combinations(pool, size):
            drugs = set(combination)
            above = profile(drugs) in tiers
            assert above == ({"rifampin", "isoniazid"} <= drugs), drugs

    assert profile({"levofloxacin"}) == "MonoResistant"
    assert profile({"levofloxacin", "amikacin"}) == "PolyResistant"
    assert profile(set()) == "Susceptible"
    assert FLUOROQUINOLONES and INJECTABLES


def test_seed_mutation_ids_match_their_own_fields():
    # Identifier and fields are written separately and can disagree silently.
    import re

    three = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
             "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
             "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
             "Tyr": "Y", "Val": "V"}
    wrong = []

    for m in seed_blobs()["mutations"]:
        protein = re.fullmatch(r"(\w+)_p\.([A-Za-z]{3})(-?\d+)([A-Za-z]{3})", m["id"])
        base = re.fullmatch(r"(\w+)_[cn]\.(-?\d+)([ACGT])>([ACGT])", m["id"])
        if protein:
            gene, ref, pos, alt = protein.groups()
            fields = (m["gene"], m["ref"], m["position"], m["alt"])
            if (gene, three.get(ref), int(pos), three.get(alt)) != fields:
                wrong.append(m["id"])
        elif base:
            gene, pos, ref, alt = base.groups()
            if (gene, int(pos), ref, alt) != (m["gene"], m["position"], m["ref"], m["alt"]):
                wrong.append(m["id"])

    assert not wrong, wrong


def test_compensatory_mutations_are_not_resistance():
    # rpoC restores fitness lost to an rpoB mutation and confers no resistance
    # itself, which is why its gene carries no drug target.
    import tb_ontology

    targets = {g["name"]: g["drug_target"] for g in tb_ontology.genes}
    classes = {"fluoroquinolones": {"levofloxacin", "moxifloxacin"},
               "aminoglycosides": {"amikacin", "kanamycin", "capreomycin"}}
    mismatched = []

    for m in tb_ontology.mutations:
        if m.get("effect") == "compensatory":
            assert targets[m["gene"]] is None, m["id"]
            continue
        target = targets.get(m["gene"])
        allowed = classes.get(target, {target}) if target else set()
        if m["drug"] not in allowed:
            mismatched.append((m["id"], target, m["drug"]))

    assert not mismatched, mismatched


def test_seed_transmissions_run_forward_in_time():
    # A strain cannot transmit to one collected before it, and nothing else
    # compares the dates against the collection years.
    import tb_ontology

    year = {s["id"]: s["year"] for s in tb_ontology.strains}
    backward = [(t["source"], t["target"]) for t in tb_ontology.transmissions
                if year[t["target"]] < year[t["source"]]]
    early = [(i["patient"], i["strain"]) for i in tb_ontology.patient_infections
             if int(i["date"][:4]) < year[i["strain"]]]

    assert not backward, backward
    assert not early, early


def test_every_self_reference_resolves():
    # A missed call site passes lint and import and fails only when it runs.
    # The graph builders need a database, so nothing else reaches them.
    import ast

    unresolved = []
    for path in sorted(ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            names = {f.name for f in ast.walk(cls) if isinstance(f, ast.FunctionDef)}
            names |= {t.attr for n in ast.walk(cls) if isinstance(n, ast.Assign)
                      for t in n.targets if isinstance(t, ast.Attribute)
                      and isinstance(t.value, ast.Name) and t.value.id == "self"}
            unresolved += [f"{path.name}:{n.lineno} {cls.name}.{n.attr}"
                           for n in ast.walk(cls)
                           if isinstance(n, ast.Attribute)
                           and isinstance(n.value, ast.Name) and n.value.id == "self"
                           and n.attr not in names and not n.attr.startswith("__")]

    assert not unresolved, unresolved


def test_platt_recovers_a_planted_distortion():
    # A near-zero fitted slope on real data has to be readable as a finding
    # rather than a failed fit, which is what the two plants separate.
    import numpy as np
    from calibration import confidence_logit, fit_platt, platt_confidence

    rng = np.random.default_rng(0)
    raw = rng.uniform(0.02, 0.98, 60000)
    truth = platt_confidence(raw, 0.7, 1.2)
    labels = (rng.uniform(size=raw.size) < truth).astype(float)

    slope, intercept = fit_platt(raw, labels)
    assert abs(slope - 0.7) < 0.05 and abs(intercept - 1.2) < 0.05

    # a score carrying no signal must fit a slope of zero, not an arbitrary one
    noise = rng.uniform(0.02, 0.98, 60000)
    flat = (rng.uniform(size=noise.size) < 0.75).astype(float)
    slope, intercept = fit_platt(noise, flat)
    assert abs(slope) < 0.05
    assert abs(1 / (1 + np.exp(-intercept)) - flat.mean()) < 0.02
    assert fit_platt([], []) == (1.0, 0.0)
    assert confidence_logit([0.5])[0] == 0.0


def test_exclusions_name_drugs_not_classes():
    # A row labeled with the class should set the flag but not reach the
    # exclusion list, where it read as a drug in a clinical readout.
    import rule_engine

    labels = {label for label, _ in rule_engine.DRUG_CLASSES.values()}
    members = {d for _, drugs in rule_engine.DRUG_CLASSES.values() for d in drugs}
    assert not labels & members

    muts = [mutation("levofloxacin", "gyrA", "gyrA_D94G", 94),
            mutation("amikacin", "rrs", "rrs_a1401g", 1401)]
    excluded = {e["drug"] for e in evaluate(muts)["recommendations"]["exclusions"]}
    assert not excluded & labels
    assert "levofloxacin" in excluded and "moxifloxacin" in excluded
    assert rule_engine.DRUG_FLAG.get("fluoroquinolone") == "fluoroquinolone_resistance"


def test_answer_prompt_forbids_inferring_susceptibility():
    # The engine reaches about four fifths of measured resistance, so a drug
    # with no graded mutation is untested rather than susceptible.
    from nl_interface import NLInterface

    interface = NLInterface(FakeOntology([]), api_key="test-key")
    prompt = interface.answer_prompt("q", "MATCH (n) RETURN n", [{"a": 1}], None, None)

    assert "untested, not susceptible" in prompt
    assert "no emoji" in prompt


def test_formulary_is_narrower_than_the_cross_resistance_class():
    # The gap looks like drift, and closing it would let the app offer a drug no
    # longer recommended. config states why; this stops the repair.
    import tb_ontology
    from config import FLUOROQUINOLONES, GROUP_A_FLUOROQUINOLONES, INJECTABLES

    modeled = {d["name"] for d in tb_ontology.drugs}
    assert GROUP_A_FLUOROQUINOLONES < FLUOROQUINOLONES
    assert modeled & FLUOROQUINOLONES == GROUP_A_FLUOROQUINOLONES
    assert not {"ciprofloxacin", "ofloxacin"} & modeled

    # injectables carry no such split, so all three are modeled
    assert INJECTABLES <= modeled

def test_class_label_never_reaches_the_exclusion_list():
    # A source row may name the class rather than a member. It must set the flag
    # without appearing beside real drug names, since it is not prescribable.
    muts = [mutation("rifampin", "rpoB"), mutation("isoniazid", "katG"),
            mutation("fluoroquinolone", "gyrA")]
    out = evaluate(muts)["recommendations"]
    excluded = {e["drug"] for e in out["exclusions"]}

    assert classify(muts) == ["PreXDR"]
    assert not excluded & rule_engine.CLASS_LABELS
    assert {"levofloxacin", "moxifloxacin"} <= excluded


def test_regimen_ceiling_marginalizes_year():
    # Year drives the regimen but is drawn independently of every retrieved
    # feature, so conditioning on it would report a bound no scored predictor
    # can reach.
    from cbr_cases import PROFILE_TARGETS, REGIMEN_OPTIONS, YEAR_WEIGHTS, YEARS, regimen_ceiling

    year_aware = sum(
        share * year_weight * max(w for _, w in options) / sum(w for _, w in options)
        for profile, share in PROFILE_TARGETS.items()
        for year, year_weight in zip(YEARS, YEAR_WEIGHTS, strict=True)
        if (options := REGIMEN_OPTIONS[profile][str(year)])
    )
    assert regimen_ceiling() < year_aware
    assert regimen_ceiling() == 0.798


def test_regimen_ceiling_renormalizes_a_subset():
    from cbr_cases import regimen_ceiling

    assert regimen_ceiling(["Susceptible", "MonoResistant"]) == 1.0
    assert regimen_ceiling(["PolyResistant", "MDR", "PreXDR", "XDR"]) == 0.469


def test_auc_handles_ties_and_a_single_class():
    from metrics import auc, brier_constant

    assert auc([]) == 0.0
    assert auc([(0.6, True), (0.6, True)]) == 0.0
    assert auc([(0.9, True), (0.1, False)]) == 1.0
    assert auc([(0.5, True), (0.5, False)]) == 0.5
    assert brier_constant([(0.3, True), (0.9, False)]) == 0.25


def test_penalties_can_only_lower_the_success_rate():
    # The floor comment claims every penalty is at most 1.0, so only SUCCESS_FLOOR
    # can bind. One above 1.0 would silently make a risk factor protective.
    import cbr_cases

    named = ["HIV_PENALTY", "DIABETES_PENALTY", "PREV_TX_PENALTY", "MALE_PENALTY",
             "HIV_DIABETES_PENALTY", "PREV_TX_MAJOR_PENALTY"]
    banded = [cbr_cases.HIV_AGE_PENALTY[1], cbr_cases.DIABETES_AGE_PENALTY[1]]
    penalties = [getattr(cbr_cases, n) for n in named] + banded
    penalties += [p for _, p in cbr_cases.AGE_PENALTIES]

    assert all(0.0 < p <= 1.0 for p in penalties)


def test_age_penalty_takes_the_highest_band_cleared():
    from cbr_cases import AGE_PENALTIES, age_penalty

    (high_age, high), (low_age, low) = AGE_PENALTIES
    assert high_age > low_age and high < low
    assert age_penalty(high_age + 1) == high
    assert age_penalty(low_age + 1) == low
    assert age_penalty(low_age) == 1.0


def test_success_rate_never_leaves_the_floor_and_base():
    # Every modifier is a penalty, so the rate can only sit between the floor and
    # the profile-regimen base it starts from.
    from cbr_cases import BASE_SUCCESS, SUCCESS_FLOOR, CaseGenerator

    generator = CaseGenerator(seed=42)
    for case in case_base(500, seed=42):
        rate = generator.success_rate(case)
        assert SUCCESS_FLOOR <= rate <= BASE_SUCCESS[(case["profile"], case["regimen"])]


def test_retrieval_score_never_goes_negative():
    # retrieve() takes min_similarity as a parameter, so avg_sim can land below
    # the constant the score is normalized against.
    from cbr_engine import MIN_SIMILARITY, ConfidenceCalculator

    confidence = ConfidenceCalculator()
    assert confidence.retrieval_score(5, 0.06) >= 0.0
    assert confidence.retrieval_score(1, 0.0) >= 0.0
    assert confidence.retrieval_score(10, MIN_SIMILARITY) >= 0.0
    assert confidence.retrieval_score(10, 1.0) == 1.0


def test_unknown_profile_scores_as_susceptible():
    # Documented fallback, not an accident. An unrecognized profile ranks as the
    # least resistant tier, so a typo degrades toward Susceptible.
    from cbr_engine import PROFILE_RANK, UNKNOWN_PROFILE_RANK, SimilarityCalculator

    assert UNKNOWN_PROFILE_RANK == PROFILE_RANK["Susceptible"]
    cases = case_base(50, seed=42)
    calculator = SimilarityCalculator(cases)
    base = {"age": 40, "sex": "M", "hiv_status": "negative", "region": "African",
            "diabetes": False, "previous_treatment": False}

    susceptible = calculator.scores({**base, "profile": "Susceptible"})
    assert (calculator.scores({**base, "profile": "NotAProfile"}) == susceptible).all()
    assert (calculator.scores(base) == susceptible).all()


def test_gene_symbols_reached_from_two_loci():
    # mutation_id is built from the symbol, so these collapse to one node. Pinned
    # so the set cannot grow without a decision.
    from collections import Counter

    from who_catalog import GENE_LOCUS

    shared = {symbol for symbol, n in Counter(GENE_LOCUS.values()).items() if n > 1}
    assert shared == {"ndh", "alr", "pepQ"}