# TB Drug-Resistance Decision Support System

[![tests](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml)
[![CI](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A hybrid decision-support prototype for *Mycobacterium tuberculosis* drug resistance. It combines a knowledge graph grounded in the World Health Organization (WHO) mutation catalog, a symbolic rule engine, case-based reasoning over synthetic patient cases, and a natural-language query layer driven by a large language model. The rule engine is validated against real-world resistance measurements from the Comprehensive Resistance Prediction for Tuberculosis international consortium, known as CRyPTIC.

## Objective

This system was developed as a graduate course project and serves as a portfolio piece. Its goal is to show how several methods work together as one pipeline. The knowledge graph supplies evidence-based structure, and the rule engine produces transparent classifications and treatment decisions. Case-based reasoning addresses cases that the rules cannot settle, and a natural-language layer translates questions into graph queries. Drug-resistant tuberculosis was chosen because the domain supplies what each of these methods needs. WHO publishes a graded mutation catalog and written resistance definitions, so the knowledge graph and the rule engine encode published evidence rather than structure invented for the exercise. Treatment selection is guideline-driven but stays underdetermined once an isolate is resistant to part of the recommended regimen, which is the gap the case-based layer fills.

Healthcare was selected because of its inherent challenges rather than despite them. The field requires systems to handle conflicting labels, incomplete outcomes, and evidence that is graded by confidence rather than definitiveness. Such imperfections are rarely encountered in clean benchmarks. Developing solutions here involved confronting these conditions directly instead of avoiding them.

Three constraints stem from this choice and apply to all the results below. The patient cases are synthetic because no open source links genotype, treatment, and outcome at the scale retrieval needs. The rule engine reaches only isolates carrying a graded mutation, which caps sensitivity short of 100%. The regimen layer is scored based on an exact match to a labeled regimen, penalizing it when it prioritizes optimizing treatment outcome instead. Each constraint is measured and shown next to the relevant figure, with Limitations offering a full explanation.

## Overview

Drug-resistant tuberculosis requires reasoning that is both auditable and grounded in current evidence. This system combines an explicit symbolic layer with a case-based layer applied to synthetic patient cases. In the symbolic layer, each classification is linked to a specific named rule, and each mutation reference corresponds to an entry in the WHO catalog. The case-based layer provides the decision on the regimen when rules alone are insufficient. A natural-language interface converts user questions into graph queries, with a read-only restriction to ensure safety, while a Streamlit front end visualizes the reasoning process.

An isolate is classified into four tiers. Below-MDR covers anything short of resistance to both first-line drugs; multidrug-resistant (MDR) signifies resistance to isoniazid and rifampin; pre-extensively drug-resistant (pre-XDR) includes resistance to one additional drug class; and extensively drug-resistant (XDR) denotes resistance to two additional classes.

The synthetic patient layer and the sensitivity ceiling of 79.7% are considered actual measured limits rather than hidden thresholds. This ceiling is determined by isolates at MDR or higher that do not possess any graded mutation. A brief demo video showcasing the front end and its reasoning trace is currently in development.

## Interactive demo

The system currently employs Streamlit for the front end. A clinical question drives the entire hybrid pipeline and returns an auditable recommendation with supporting reasoning.

![The app answering a treatment query for patient P003, showing the diagnosis, the XDR classification, and the contraindicated drugs beside the mutation or class rule that excludes each one](assets/query-results.png)

A question such as "What treatment should patient P003 receive" is answered across four tabs.

- Query Results carries the direct answer, the strain and its classification, the recommended regimen, and a table of contraindicated drugs tied to the mutations that rule them out.
- Expert System exposes the rule-engine trace, the canonical gene fraction, the rules that fired, and the regimen with its drug exclusions.
- Case-Based Reasoning retrieves the nearest matches from the 1,000 synthetic patient cases and reports a success rate and a confidence band.
- Technical Details shows the Cypher that the natural-language layer generated from the question, so the path from text to graph query stays visible.

The Expert System tab carries the symbolic trace. Strain TB011 classifies as XDR under rule RC002, which selects BPaL, and each excluded drug names the mutation that ruled it out. The tab applies five rules rather than six. A treatment question runs backward chaining, which reaches the XDR goal without visiting the pre-XDR rule.

![The Expert System tab for strain TB011, showing the XDR classification with its rule and source, the five rules applied, the BPaL regimen with its duration, and the bedaquiline and linezolid indications from rules TS004 and TS005](assets/expert-system.png)

The Case-Based Reasoning tab answers the same question from prior similar cases rather than from rules. Ten neighbors match this patient and three of them succeeded. The tab reports 33.3% rather than the raw 30% because the share is Laplace-smoothed, which the Calibration section describes. The confidence band reads moderate rather than high because the neighbors disagree on outcome.

![The Case-Based Reasoning tab for patient P003, showing the XDR patient profile, ten similar cases, a 33.3 percent success rate, and moderate confidence of 0.58](assets/cbr.png)

![The Technical Details tab showing the Cypher generated from the question, the confirmation that it ran, and the nine results it returned](assets/tech-details.png)

### Running the demo

Full setup is in [DEPLOYME.md](DEPLOYME.md). The short version, once dependencies and the
environment are in place, is three commands.

```bash
docker run -d -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph-mage:3.9.0
python SRC/tb_ontology.py
streamlit run SRC/app.py
```

Paste an Anthropic API key into the sidebar, click Initialize CBR to load the 1,000
synthetic cases, then ask a question such as "What treatment should patient P003 receive".
The seed strains and patients load whether or not the large datasets are present, so the
demo runs on the seed graph alone.

## Architecture

The design separates a durable, evidence-based platform from a swappable patient layer.

- Knowledge graph. A Memgraph database contains 1,295 mutation nodes sourced from the WHO mutation catalog. This catalog rates 48,152 variant and drug pairs on a scale from 1 to 5, covering 30,699 unique variants. The graph loads only the 1,383 pairs graded 1 or 2 that are associated with resistance, since higher grades indicate uncertain or no association. These pairs consolidate into 1,291 distinct nodes because nodes are identified by mutation, so a variant linked to multiple drugs merges into a single node. The remaining four nodes are seed mutations the catalog does not grade, since 19 of the 23 seed mutations share an identifier with a catalog entry and merge into it. Memgraph speaks the Bolt protocol, so the code reaches it through the neo4j Python driver, and the neo4j dependency in requirements.txt is that driver rather than a separate database.

- Rule engine. A symbolic engine using forward and backward chaining classifies isolates as MDR, pre-XDR, or XDR. It also applies whole-class cross-resistance and chooses between the BPaL and BPaLM regimens. Backward chaining aims for a specific goal, answering either a treatment or classification question. When targeting a treatment goal, both modes agree on classifications, exclusions, alerts, regimens, inclusions, and monitoring entries across all resistance flag combinations tested. For a classification goal, agreement includes classifications, exclusions, and alerts; the regimen and downstream elements are withheld since no treatment is requested. The list of rules fired sits outside that guarantee, because forward chaining reaches the pre-XDR rule on an XDR isolate while backward chaining stops earlier. This list appears in the Expert System tab as the trace, reflecting the mode used.

- Case-based reasoning. Retrieval over 1,000 synthetic patient cases that returns a regimen, a success rate, and a confidence band from the nearest neighbors. The model uses seven hand-set similarity weights rather than learned ones.

- Natural-language interface. An Anthropic large language model layer converts plain English into Cypher queries. It is protected by a read-only guard. The query executes in a read transaction that Memgraph rejects if any write is attempted, making the database the primary barrier. Additionally, a keyword pre-filter prevents obvious write attempts before the query executes.

The figure below follows a single strain through the graph, from its mutations to the genes and drugs they impact, and finally to its resistance profile. The rule engine retrieves mutation-to-drug associations directly from this structure without traversing it. Therefore, the figure illustrates the source of its data rather than a specific path taken.

![Strain TB011 traced through the knowledge graph, from its four mutations to the genes and drugs they affect and on to its XDR resistance profile](assets/knowledge_graph.png)

## Results

Every figure below is reproduced by the commands in [DEPLOYME.md](DEPLOYME.md) section 6 and
written to `Evaluation/validation_results.json`. Full scoring for each arm, with the
per-tier and per-drug breakdowns, sits in [EVALUATION.md](EVALUATION.md).

| Arm | Measure | Score | Floor | Ceiling |
| --- | --- | ---: | ---: | ---: |
| Tier classification | accuracy against phenotype | 83.4% | 73.3% majority | 83.5% catalog |
| Tier classification | balanced accuracy | 67.4% | 25.0% | 67.9% catalog |
| Regimen retrieval | exact match | 77.2% | 79.3% majority | 79.8% |
| Outcome probability | area under the curve | 0.562 | 0.500 | 0.668 |
| Query translation | execution match | 90.9% | | |

Two of those rows carry a measured ceiling rather than a perfect one, and the pair is the
most useful thing in the table. The regimen ceiling of 79.8% sits half a point above the
score and above the majority baseline, so that layer is bounded by the synthetic cohort.
The outcome ceiling of 0.668 sits far above the score, so that layer is bounded by the
neighborhood weighting instead. The same calculation separates them, and
[EVALUATION.md](EVALUATION.md) sets out both.

Three limits shape every number above. Sensitivity is capped at 79.7% because 3,562 of the
17,523 isolates measured at MDR or higher carry no mutation the catalog grades. Precision
on the top two tiers is held down because only 59.0% of the cohort was tested for both a
fluoroquinolone and an injectable, and the label reads an untested drug as susceptible.
The regimen layer is scored by exact match to a labeled regimen, which penalizes it for
optimizing treatment outcome instead. Limitations gives the full account.

![Grouped bar chart of per-tier sensitivity against measured phenotype, rule engine versus WHO catalog, across below-MDR, MDR, pre-XDR, and XDR. The two systems are within a point of each other on every tier, and sensitivity declines from 92% on below-MDR to 55% on XDR as the tiers grow rarer.](assets/cryptic_tier_sensitivity.png)

## Data

The platform is built on the [WHO mutation catalog, second edition](https://www.who.int/publications/i/item/9789240082410), supplied as the file WHO-UCN-TB-2023.7-eng.xlsx. Real-world validation uses [CRyPTIC release 3.4.0](https://doi.org/10.5281/zenodo.15680920), which pairs whole-genome variants graded against that catalog with laboratory resistance phenotypes. This release includes 53,897 samples with both sequencing and phenotype data, plus 11,945 samples with only phenotype data. The validation set consists of 65,588 samples with measured drug results, scored in their entirety rather than on a held-out subset.

The release carries two phenotype sources. Drug susceptibility testing, abbreviated DST, is the reference method run in clinical laboratories. The UKMYC plate is a broth microdilution assay read across a fixed drug panel. The two agree on 94.8% of the 21,568 isolates measured by both, which sets a label-noise floor beneath every accuracy figure reported above. They disagree on 1,117 isolates, and on all 1,117 the UKMYC profile is the less severe of the two, never the more severe. The exploratory analysis separates how much of that follows from the narrower UKMYC panel.

The catalog does not classify every call as resistant or susceptible. Some results are uncertain or failed, and both arms count these as not resistant, aligning with the approach where isolates without genotypic calls are considered below-MDR. The exposure is worth naming, since 31,517 isolates have at least one uncertain call, and 2,927 have one on rifampin or isoniazid, the two key drugs for defining MDR. If uncertain calls were categorized as resistant, all the figures in both arms would change.

The synthetic patient cases are transparent and deterministic under a fixed seed.

The actual datasets are not included in this repository due to their large size. To reproduce the results, download them into a `Datasets/` folder located at the project root. The catalog file WHO-UCN-TB-2023.7-eng.xlsx is from WHO. The CRyPTIC tables, including EFFECTS.parquet, PREDICTIONS.parquet, DST_MEASUREMENTS.parquet, UKMYC_PHENOTYPES.parquet, and the file DRUG_CODES.csv, originate from CRyPTIC release 3.4.0 on Zenodo. The synthetic patient cases are generated through code and do not require downloading. Accessing the CRyPTIC parquet tables requires the pyarrow engine, which is installed via `requirements.txt`.

The release also ships `DATA_SCHEMA.pdf`, which documents the full set of tables, and `MUTATIONS.parquet`, which this project retains but does not read. The exploratory analysis explains why the rule engine sources its genotypes from EFFECTS instead. A seventh file, `Datasets/cryptic_features.parquet`, is built on first use and cached. It rebuilds itself whenever a source table, `feature_engineering.py`, or `config.py` is newer than the cache, so replacing a table is enough.

## Evaluation

Scoring runs through two entry points, both detailed in [DEPLOYME.md](DEPLOYME.md) section 6.

```bash
python Evaluation/validation.py   # tier, expert-system, and case-based arms
python Evaluation/metrics.py      # per-drug scoring
pytest tests/test_core.py         # 127 tests, no database, API, or datasets
```

Every metric comes from `Evaluation/metrics.py`, so the tier arm and the per-drug arm
measure the same quantities. The reference arm differs between them. The tier arm reads
the catalog profile from `PREDICTIONS.parquet` and the per-drug arm reads it from
`EFFECTS.parquet`, so both of its columns come from one table. Compare within an arm
rather than across the two files.

The CRyPTIC, per-drug, and case-based arms are deterministic under a fixed seed of 42, and
repeated runs reproduce every digit in the two result files apart from the last decimal
place of one p-value, which moves with floating-point summation order. The expert-system
arm calls a live model and is the one figure expected to change, so it is reported beside
the model that generated it. Four runs returned ten of eleven three times and eleven of
eleven once.

The project builds up in three levels. The test suite needs only
`pip install -r requirements-dev.txt`. Adding Docker and an Anthropic API key activates the
demo and the expert-system arm on the seed graph, with no datasets downloaded. Adding the
`Datasets/` folder unlocks the CRyPTIC and per-drug arms.

### Exploratory analysis

[EDA/EDA.ipynb](EDA/EDA.ipynb) documents the data work behind the design, including the label-noise floor, the baselines the case-based layer must beat, the coverage gap between the PREDICTIONS and EFFECTS tables, and the seed graph's composition. It shares `baseline_accuracy` with `validation.py`, so the baselines shown there and the ones the system is scored against are the same function rather than two similar ones.

## Limitations

- The patient layer is synthetic because no open dataset links genotype, regimen, and outcome at the scale a case-based recommender needs. This data scarcity is a well-known challenge in healthcare machine learning, and it is the direct reason the rare resistant classes evaluate poorly.


- The case-based similarity weights are domain-informed priors set by hand, not values learned from data, and tuning them is future work. The region and outcome tables in the case generator follow the same pattern, since they carry real structure from the WHO regions while their magnitudes stay synthetic rather than transcribed from any WHO release.


- The regimen layer is scored by exact match to the labeled regimen, which penalizes it for optimizing treatment outcome instead, so part of the measured shortfall is a metric mismatch rather than a modeling error.


- CRyPTIC provides genotype and phenotype but not treatment outcomes, so it validates classification only and cannot validate the regimen and outcome layer.


- The rule engine implements a scoped pre-2021 XDR definition, documented as a deliberate choice rather than the current standard, which is built on the Group A drugs. The release does carry bedaquiline and linezolid phenotypes, but on few isolates and at genotypic sensitivity of 0.40 and 0.23. Those are the weakest of the four drugs the current definition reads, below levofloxacin at 0.66 and moxifloxacin at 0.70, so that definition would rest on the thinnest columns in the data.


- The engine anchors MDR to resistance against isoniazid and rifampin together. The current definition anchors instead to multidrug-resistant or rifampin-resistant tuberculosis, under which rifampin resistance alone qualifies. This follows from the pre-2021 scope described earlier and is applied consistently. The reference label reads the same anchor through `feature_engineering.profile()`, so both sides of the comparison are evaluated against one definition. The effect at that boundary is conservative, because a rifampin-monoresistant isolate is classified below MDR rather than above it, so the engine under-calls rather than over-calls.


- A regimen is a guideline recommendation, not a per-patient prescription. Where an isolate is resistant to a component drug, the engine keeps the regimen and names it in a contraindicated field rather than substituting, since choosing the replacement is a clinical decision the rule base does not model.


- Both inference paths return the same classification, exclusions, and alerts, and they return the same regimen, inclusions, and monitoring whenever the question asked is a treatment question. A classification question stops at the tier, so the regimen and everything downstream of it is withheld rather than computed. The list of rules fired can differ in one further way. Forward chaining evaluates every rule, so an XDR isolate also fires the pre-XDR rule whose criteria it meets, while backward chaining stops once the XDR goal is proved. The conclusions are identical either way, and the difference is what the two search strategies visit.


- Classification and treatment are keyed to different guideline revisions. An isolate is tiered by the pre-2021 definition and routed by WHO 2022 regimen rules, so an injectable-resistant isolate is labeled pre-XDR and still receives BPaLM. Each half is correct on its own terms. Under the current definition, injectable resistance does not reach pre-XDR, so the same isolate would be labeled multidrug-resistant and the pairing would raise no question at all. The mismatch follows from the tiering choice above, not from the regimen rules.


- The mono and poly split is performed by the labeling pipeline instead of the rule engine, and it counts every resistant drug rather than just the first-line set. This aligns with the terminology used by the Centers for Disease Control and Prevention and extends to second-line agents as WHO anticipated for surveillance. WHO's own definition is restricted to first-line drugs only, so this is a stated deviation. It alters no classification at MDR or above, which the test suite holds structurally.


- Precision is limited by phenotype coverage. The label identifies an untested drug as susceptible. Only 59.0% of the cohort has both a fluoroquinolone and an injectable result. Consequently, a correct genotypic call above MDR can be falsely scored as positive if the isolate was never tested. Among the measured isolates, pre-XDR precision improves from 0.483 to 0.616, and XDR from 0.482 to 0.629, while sensitivity remains unchanged, as reported under `second_line_covered`.


- Sensitivity is bounded by input coverage. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded mutation and cannot be reached by any rule, so every sensitivity figure should be read against the ceiling reported beside it rather than against 100%.


- Cross-resistance within the injectable class is modeled as a whole, and the per-drug scoring shows that it costs more precision than it gains in sensitivity. The behavior is observed and reported instead of being corrected. Narrowing it does not change the tier because the class rule only affects the exclusion list and not the facts the classifier uses. However, it shifts every value in the per-drug section, so it should be considered a rescoring rather than a patch. Refer to Future work.


- The rule engine does not model ethionamide, so the cross-resistance that links isoniazid and ethionamide through the enoyl reductase gene, written inhA, is out of scope. This is a named boundary rather than an oversight.


## Future work

Several directions would extend the work, and they fall into two groups, the data the system can reach and the way its layers are scored.

The most significant data gap is the synthetic case base. Validating the case-based layer with the TB Portals dataset, which includes actual treatment outcomes, would replace the cohort where real signals are most needed. Real data would strengthen the result without necessarily raising the accuracy, since resistant cases stay rare even in large collections. A trained model could push the results toward the other ceiling. Training such a model with the full genome-wide variant table and the minimum inhibitory concentration magnitudes would explore how much of the genotype-phenotype mismatch can be explained beyond the curated catalog.

The remaining directions concern how the system is evaluated and how it handles rare classes. The regimen layer is scored by exact match to the labeled regimen, which penalizes it for optimizing treatment outcome and guideline adherence instead, so part of the measured shortfall is a metric mismatch rather than a modeling error. Scoring against outcome directly is the simplest of these changes, since it needs no additional data and fixes the mismatch at its source. The outcome layer now carries a measured ceiling, and the gap beneath it is wide enough to make the similarity weights the first thing to tune, scored against that ceiling rather than against a perfect one. Confidence-gated deferral would extend the abstention the engine already performs when its neighbors carry nothing applicable. A sparse neighborhood would hand the case to the rule engine, and the arm would report coverage beside accuracy, which turns rare-class scarcity into calibrated behavior rather than silent error.

The whole-class injectable rule groups amikacin, kanamycin, and capreomycin together, and the per-drug scoring shows it over-calls all three against measured phenotype. The cost falls almost entirely on two of them. Amikacin and capreomycin lose 31.7 and 33.7 points of precision for gains of 2.2 and 4.2 in sensitivity, while kanamycin loses 3.9 for a gain of 0.2. That asymmetry is what the expansion would produce if kanamycin, the most frequently resistant of the three, is mainly the source of the added calls rather than their recipient. Tying cross-resistance to the gene instead, with rrs conferring resistance across the class and eis favoring kanamycin, would recover precision without moving any tier, since the class rule writes only into the exclusion list and never into the facts the classifier reads.



## References

### Case-Based Reasoning

1. Kolodner, J. L. (1992). An Introduction to Case-Based Reasoning. *Artificial Intelligence Review*, 6(1), 3-34.
2. Main, J., Dillon, T. S., & Shiu, S. C. K. (2001). A Tutorial on Case-Based Reasoning. *Soft Computing in Case Based Reasoning*, 1-28.
3. Goel, A. K., & Díaz-Agudo, B. (2017). What's Hot in Case-Based Reasoning. *Proceedings of AAAI-17*.
4. Das, R., Godbole, A., Dhuliawala, S., Zaheer, M., & McCallum, A. (2020). A Simple Approach to Case-Based Reasoning in Knowledge Bases. *Automated Knowledge Base Construction (AKBC)*.

### WHO Guidelines

5. World Health Organization. (2023). *Catalogue of mutations in Mycobacterium tuberculosis complex and their association with drug resistance* (2nd ed.).
6. World Health Organization. (2022). *WHO consolidated guidelines on tuberculosis. Module 4, drug-resistant tuberculosis treatment, 2022 update*. (the source the treatment rules cite)
7. World Health Organization. (2025). *WHO consolidated guidelines on tuberculosis. Module 4, treatment and care*.
8. Walker, T. M., et al. (2022). The 2021 WHO catalogue of Mycobacterium tuberculosis complex mutations. *The Lancet Microbe*, 3(4), e265-e273.

### Calibration

9. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 1321-1330.

### Treatment Evidence

10. Nyang'wa, B.-T., et al. (2022). A 24-Week, All-Oral Regimen for Rifampin-Resistant Tuberculosis. *New England Journal of Medicine*, 387(25), 2331-2343. (TB-PRACTECAL; BPaLM)
11. Conradie, F., et al. (2020). Treatment of Highly Drug-Resistant Pulmonary Tuberculosis. *New England Journal of Medicine*, 382(10), 893-902. (Nix-TB; BPaL)

### Datasets

12. The CRyPTIC Consortium. (2025). *CRyPTIC data release 3.4.0* [Data set]. Zenodo. doi:10.5281/zenodo.15680920. (the release the reported figures are computed on)
13. The CRyPTIC Consortium. (2022). A data compendium associating the genomes of 12,289 *Mycobacterium tuberculosis* isolates with quantitative resistance phenotypes to 13 antibiotics. *PLOS Biology*, 20(8), e3001721.
14. Rosenthal, A., et al. (2017). The TB Portals, an Open-Access, Web-Based Platform for Global Drug-Resistant-Tuberculosis Data Sharing and Analysis. *Journal of Clinical Microbiology*, 55(11). doi:10.1128/JCM.01013-17.

## License

Released under the MIT License.
