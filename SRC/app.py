import streamlit as st
from dataclasses import dataclass
from tb_ontology import TBOntology
from nl_interface import NLInterface, LLMUnavailable
from cbr_engine import CaseStore

st.set_page_config(
    page_title="TB Drug Resistance Hybrid AI System",
    layout="wide"
)


def init_system(api_key):
    ontology = TBOntology()
    nl_interface = NLInterface(ontology, api_key=api_key)
    return ontology, nl_interface


def show_database_stats(ontology):
    try:
        node_counts = ontology.count_nodes()
        for node_info in node_counts:
            st.metric(node_info['type'], node_info['count'])
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")


def cbr_status():
    nl_interface = st.session_state.get('nl_interface')
    if not nl_interface:
        return False, "System not initialized"

    if not nl_interface.cbr_engine:
        return False, "CBR not initialized"

    case_count = len(nl_interface.cbr_cases)
    if case_count == 0:
        return False, "No cases available"

    return True, f"Active with {case_count} cases"


def initialize_cbr():
    nl_interface = st.session_state.get('nl_interface')
    ontology = st.session_state.get('ontology')

    if not nl_interface or not ontology:
        st.error("System not initialized. Please check API key.")
        return False

    try:
        store = CaseStore(ontology)
        existing_cases = store.count_cases()

        if existing_cases == 0:
            st.info("Importing cases to database - first time only...")
            from cbr_engine import graph_cases
            graph_cases(n_cases=1000)

        case_count = nl_interface.init_cbr()

        if case_count > 0 and nl_interface.cbr_engine:
            st.success(f"Imported {case_count} cases")
            return True
        else:
            st.error("Failed to initialize CBR: no cases available")
            return False

    except Exception as e:
        st.error(f"CBR init error: {str(e)}")
        return False


def display_rule_output(rule_output):
    st.subheader("Expert System Analysis")
    _rule_metrics(rule_output)
    _rule_recommendations(rule_output['recommendations'])
    with st.expander("Rules Fired"):
        st.json(rule_output['rules_fired'])


def _rule_metrics(rule_output):
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Canonical Gene Fraction", f"{rule_output['canonical_gene_fraction']:.0%}")
        st.metric("Rules Applied", len(rule_output['rules_fired']))

    with col2:
        st.metric("Strain Analyzed", rule_output['strain'])
        classifications = rule_output['recommendations'].get('classifications', [])
        if classifications:
            st.metric("Classification", classifications[0]['type'])


def _rule_recommendations(recs):
    if recs.get('classifications'):
        st.write("**Classifications:**")
        for c in recs['classifications']:
            st.write(f"- {c['type']} (Rule: {c['rule']}, Source: {c['source']})")
    else:
        st.write("No MDR-class resistance detected (below-MDR).")

    if recs.get('regimens'):
        st.write("**Treatment Regimens:**")
        for r in recs['regimens']:
            st.write(f"- {r['name']}: {', '.join(r['drugs'])}")
            st.write(f"  Duration: {r['duration']} (Rule: {r['rule']})")

    if recs.get('exclusions'):
        st.write("**Drug Exclusions:**")
        for e in recs['exclusions']:
            st.write(f"- Exclude {e['drug']} (Reason: {e['reason']}, Rule: {e['rule']})")

    if recs.get('monitoring'):
        st.write("**Monitoring Required:**")
        for m in recs['monitoring']:
            st.write(f"- {m['parameter']}")
            if m.get('threshold'):
                st.write(f"  Threshold: {m['threshold']}")

    if recs.get('alerts'):
        st.write("**Clinical Alerts:**")
        for a in recs['alerts']:
            st.write(f"- {a['type']} (Rule: {a['rule']})")


def display_query_profile(cbr_output):
    qp = cbr_output.get('query_profile', {})
    if not qp:
        return

    st.write("**Patient Profile:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"Profile: **{qp.get('profile', 'N/A')}**")
        st.write(f"Age: {qp.get('age', 'N/A')}")
        st.write(f"Sex: {qp.get('sex', 'N/A')}")

    with col2:
        st.write(f"HIV: {qp.get('hiv_status', 'N/A')}")
        diabetes = "Yes" if qp.get('diabetes') else "No"
        st.write(f"Diabetes: {diabetes}")

    with col3:
        st.write(f"Region: {qp.get('region', 'N/A')}")
        prev_tx = "Yes" if qp.get('previous_treatment') else "No"
        st.write(f"Previous Tx: {prev_tx}")


def display_confidence(cbr_output):
    conf = cbr_output.get('confidence', {})
    if not conf:
        return

    level = conf.get('level', 'N/A')
    score = conf.get('score', 0)

    level_colors = {'high': 'green', 'moderate': 'orange', 'low': 'red'}
    color = level_colors.get(level, 'gray')

    st.write("**Confidence Assessment:**")
    outcome_prob = cbr_output.get('outcome_probability')
    if outcome_prob is not None:
        st.markdown(f"Estimated success probability: **{outcome_prob:.0%}**")
    st.markdown(f"Evidence: :{color}[**{level.upper()}**] (score: {score:.2f})")

    factors = conf.get('factors', {})
    if factors:
        for factor, data in factors.items():
            icon = "+" if data['score'] >= 0.6 else "~" if data['score'] >= 0.4 else "-"
            st.write(f"  {icon} {factor.title()}: {data['reason']}")

    interpretation = conf.get('interpretation', '')
    if interpretation:
        st.caption(interpretation)


def display_outcome_analysis(cbr_output):
    oa = cbr_output.get('outcome_analysis', {})
    if not oa:
        return

    st.write("**Outcome Analysis:**")

    dist = oa.get('distribution', {})
    if dist:
        total = sum(dist.values())
        parts = []
        for outcome, count in sorted(dist.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            parts.append(f"{outcome}: {count} ({pct:.0f}%)")
        st.write("Distribution: " + " | ".join(parts))

    weighted = oa.get('weighted_success_rate', 0)
    st.write(f"Weighted Success Rate: {weighted:.1%}")

    risk = oa.get('risk_factors', [])
    if risk:
        st.write(f"Risk Factors Present: {', '.join(risk)}")


def display_similar_case(exp_case):
    case_id = exp_case.get('case_id', 'Unknown')
    sim = exp_case.get('similarity', 0)
    outcome = exp_case.get('outcome', 'unknown')
    regimen = exp_case.get('regimen', 'Unknown')

    outcome_colors = {
        'success': 'green',
        'death': 'red',
        'failed': 'orange',
        'ltfu': 'gray',
        'not_evaluated': 'gray'
    }
    o_color = outcome_colors.get(outcome, 'gray')

    header = f"**{case_id}** | Similarity: {sim:.3f} | {regimen} | :{o_color}[{outcome}]"

    with st.expander(header):
        st.write("**Feature Breakdown:**")
        _case_feature_breakdown(exp_case.get('feature_breakdown', []))
        _case_matches(exp_case)


def _case_feature_breakdown(breakdown):
    for fb in breakdown:
        match = fb.get('match', 'unknown')
        icon = "+" if match in ['exact', 'close'] else "~" if match == 'partial' else "-"

        feature = fb.get('feature', '')
        q_val = fb.get('query_value', '')
        c_val = fb.get('case_value', '')
        contrib = fb.get('contribution', 0)

        if q_val == c_val:
            st.write(f"  {icon} {feature}: {q_val} = {c_val} (+{contrib:.3f})")
        else:
            st.write(f"  {icon} {feature}: {q_val} vs {c_val} (+{contrib:.3f})")


def _case_matches(exp_case):
    top = exp_case.get('top_matches', [])
    diffs = exp_case.get('key_differences', [])

    if top:
        st.write(f"Top Matches: {', '.join(top)}")
    if diffs:
        st.write(f"Key Differences: {', '.join(diffs)}")


def display_cbr_metrics(cbr_output):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Similar Cases", len(cbr_output.get('similar_cases', [])))
    with col2:
        st.metric("Success Rate", f"{cbr_output.get('success_rate', 0):.1%}")
    with col3:
        conf = cbr_output.get('confidence', {})
        level = conf.get('level', 'N/A')
        score = conf.get('score', 0)
        st.metric("Confidence", f"{level} ({score:.2f})")


def display_cbr_recommendations(cbr_output):
    recs = cbr_output.get('recommendations', [])
    if not recs:
        return
    st.write("**Recommended Regimens:**")
    for rec in recs:
        regimen = rec.get('regimen', 'Unknown')
        rate = rec.get('success_rate', 0)
        cases = rec.get('evidence_cases', 0)
        conf = rec.get('confidence', 'low')
        st.write(f"- {regimen}: {rate:.1%} success ({cases} cases) - {conf} confidence")


def display_cbr_similar(cbr_output):
    explained = cbr_output.get('explained_cases', [])
    if explained:
        st.write("**Similar Cases - with explanations:**")
        for exp_case in explained[:5]:
            display_similar_case(exp_case)
        return

    similar = cbr_output.get('similar_cases', [])
    if similar:
        with st.expander("View Similar Cases"):
            for similarity, case in similar[:10]:
                st.write(
                    f"Similarity: {similarity:.3f} | {case['case_id']} | "
                    f"{case['regimen']} | {case['outcome']}"
                )


def display_cbr_output(cbr_output):
    st.subheader("Case-Based Reasoning")
    display_query_profile(cbr_output)
    st.markdown("---")
    display_cbr_metrics(cbr_output)
    st.markdown("---")
    display_confidence(cbr_output)
    st.markdown("---")
    display_outcome_analysis(cbr_output)
    st.markdown("---")
    display_cbr_recommendations(cbr_output)
    st.markdown("---")
    display_cbr_similar(cbr_output)


def display_technical_details(user_question, cypher, results):
    st.subheader("Generated Cypher Query")
    st.code(cypher, language="cypher")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Results Found", len(results))
    with col2:
        if results:
            st.success("Query executed successfully")
        else:
            st.info("No results found")

    if results:
        with st.expander("View Raw Query Results"):
            st.json(results[:20] if len(results) > 20 else results)


@dataclass
class QueryOutcome:
    cypher: str = ""
    results: list = None
    error: str = None
    question_type: str = None
    rule_output: dict = None
    cbr_output: dict = None


def cypher_outcome(nl_interface, user_question):
    try:
        cypher = nl_interface.generate_cypher(user_question)
    except LLMUnavailable as e:
        return QueryOutcome(error=f"Language model unavailable: {str(e)}")

    is_valid, error = nl_interface.validate_cypher(cypher)

    if not is_valid:
        return QueryOutcome(cypher=cypher, error=error)
    if "UNANSWERABLE" in cypher:
        return QueryOutcome(cypher=cypher, error=cypher)

    try:
        results = nl_interface.execute_query(cypher)
    except Exception as e:
        return QueryOutcome(cypher=cypher, error=f"Error executing query: {str(e)}")

    return QueryOutcome(cypher=cypher, results=results)


def process_query(user_question):
    nl_interface = st.session_state['nl_interface']
    outcome = cypher_outcome(nl_interface, user_question)

    if outcome.error:
        return outcome

    nl_interface.last_question = user_question
    outcome.question_type = nl_interface.needs_rules(user_question)

    if outcome.question_type:
        outcome.rule_output = nl_interface.rule_recommend(outcome.results, outcome.question_type)

    if outcome.question_type == 'treatment' and nl_interface.cbr_engine:
        outcome.cbr_output = nl_interface.cbr_recommend(outcome.results)

    return outcome


def display_query_tabs(user_question, outcome, nl_interface):
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Query Results", "Expert System", "Case-Based Reasoning", "Technical Details"])

    with tab4:
        if outcome.error:
            st.error(outcome.error)
        else:
            display_technical_details(user_question, outcome.cypher, outcome.results)

    if outcome.error or not outcome.results:
        return

    with tab2:
        if outcome.rule_output:
            display_rule_output(outcome.rule_output)
        else:
            st.info("No expert system rules applicable for this query")

    with tab3:
        display_cbr_tab(outcome.cbr_output, nl_interface)

    with tab1:
        st.subheader("Answer")
        answer = nl_interface.format_results(
            user_question, outcome.cypher, outcome.results, outcome.rule_output, outcome.cbr_output)
        st.markdown(answer)


def display_cbr_tab(cbr_output, nl_interface):
    if cbr_output is None:
        st.error("Patient not found in database or CBR query failed")
        try:
            patient_count = nl_interface.ontology.query("MATCH (p:Patient) RETURN count(p) as count")
            if patient_count:
                st.info(f"Database contains {patient_count[0]['count']} patients")
        except Exception:
            pass
    elif cbr_output:
        display_cbr_output(cbr_output)
    else:
        st.info("CBR not initialized or not applicable for this query")


st.title("TB Drug Resistance Hybrid AI System")
st.markdown("Hybrid system: Knowledge Graph + Rule Engine + Case-Based Reasoning")

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Enter your Anthropic API key"
    )

    if api_key:
        st.success("API key provided")

        if 'nl_interface' not in st.session_state or st.session_state.get('api_key') != api_key:
            st.session_state['api_key'] = api_key
            ontology, nl_interface = init_system(api_key)
            st.session_state['ontology'] = ontology
            st.session_state['nl_interface'] = nl_interface
    else:
        st.warning("Please enter API key to continue")

    st.markdown("---")

    st.header("Database Statistics")
    if api_key and 'ontology' in st.session_state:
        show_database_stats(st.session_state['ontology'])

    st.markdown("---")

    st.header("CBR System")
    if api_key and st.button("Initialize CBR"):
        if initialize_cbr():
            st.rerun()

    is_active, status_msg = cbr_status()
    if is_active:
        st.success(status_msg)
    else:
        st.info(status_msg)

    st.markdown("---")
    st.caption("TB Expert System v4.0")
    st.caption("KG + Rules + CBR")

if not api_key:
    st.info("Enter your Anthropic API key in the sidebar to begin")
    st.stop()

if 'nl_interface' not in st.session_state:
    st.error("System not initialized. Please refresh the page.")
    st.stop()

nl_interface = st.session_state['nl_interface']
ontology = st.session_state['ontology']

user_question = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="e.g., What treatment should patient P003 receive?"
)

if st.button("Submit Query", type="primary", use_container_width=True):
    if not user_question:
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("Processing your question"):
        outcome = process_query(user_question)
        display_query_tabs(user_question, outcome, nl_interface)