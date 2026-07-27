# TB Drug-Resistance Decision Support System

[![tests](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml)
[![CI](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A hybrid decision-support prototype for *Mycobacterium tuberculosis* drug resistance. It combines a WHO-grounded knowledge graph, a symbolic rule engine, case-based reasoning over synthetic patient cases, and an LLM-driven natural-language query layer. Its symbolic core is validated against real-world resistance measurements from the CRyPTIC consortium.

## Objective

This system was developed as a graduate course project and serves as a portfolio piece. Its goal is to show how several methods work together as one pipeline. The knowledge graph supplies evidence-based structure, and the rule engine produces transparent classifications and treatment decisions. Case-based reasoning addresses cases that the rules cannot settle, and a natural-language layer translates questions into graph queries. Drug-resistant tuberculosis was chosen because each method has a clear role here, and the reasoning must remain auditable rather than hidden.

Choosing healthcare also involved dealing with imperfect data. Resistance measurements are often incomplete and noisy, and sometimes the two phenotype assays disagree on the same isolate. Supplemental data that could fill these gaps was difficult to obtain. Records linking genotype, treatment, and outcome are rare, and the outcome data needed for the case base was not available at the required scale. The study reports these limitations, including genotype-phenotype discordance, assay disagreements that create a noise floor, and missing outcome data that necessitated a synthetic case base. The goal was to honestly acknowledge these data limitations rather than to develop a clinical tool.

## Overview

Drug-resistant tuberculosis requires reasoning that is both auditable and grounded in current evidence. This system combines an explicit symbolic layer, where each resistance classification links to a WHO catalog rule, with a case-based layer that leverages prior patient experience when guidelines are absent. It features a natural-language interface that converts questions into graph queries within a read-only environment, and a Streamlit front end that visualizes the complete reasoning process.

The synthetic patient layer and the genotype-phenotype prediction ceiling are treated as measured limits rather than hidden ones. A short demo video of the front end and its reasoning trace is in progress.

## Interactive demo

The Streamlit front end is the system in use. A plain-English clinical question drives the full hybrid pipeline and returns an auditable recommendation together with the reasoning behind it.

![The app answering a treatment query for patient P003, showing the diagnosis, the XDR classification, and the contraindicated drugs alongside the mutations that exclude them](assets/query-results.png)

A question such as "What treatment should patient P003 receive" is answered across four tabs.

- Query Results carries the direct answer, the strain and its classification, the recommended regimen, and a table of contraindicated drugs tied to the mutations that rule them out.
- Expert System exposes the rule-engine trace, the evidence confidence, the rules that fired, and the regimen with its drug exclusions.
- Case-Based Reasoning retrieves the nearest matches from the 1,000 synthetic patient cases and reports a success rate and a confidence band.
- Technical Details shows the Cypher that the natural-language layer generated from the question, so the path from text to graph query stays visible.

![The Technical Details tab showing the Cypher generated from the question together with the twelve graph results returned](assets/tech-details.png)

### Running the demo

After installing the dependencies and setting the environment (see [DEPLOYME.md](DEPLOYME.md)), bring the system up in this order.

1. Start a local Memgraph instance in Docker and leave it running in the background. If the container already exists from an earlier run, resume it with `docker start memgraph` instead.

    ```bash
    docker run -d -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph-mage:3.9.0
    ```

2. Build the knowledge graph. This clears the database, applies the schema, loads the seed strains and patients, and merges the WHO catalog, then prints `Database initialized successfully`.

    ```bash
    python SRC/tb_ontology.py
    ```

3. Launch the application.

    ```bash
    streamlit run SRC/app.py
    ```

4. Paste an Anthropic API key into the sidebar, since the natural-language layer calls the Anthropic API to turn questions into Cypher.

5. Click Initialize CBR in the sidebar to load the 1,000 synthetic cases. The control reads `Active with 1000 cases` once the case base is ready.

6. Ask a question such as "What treatment should patient P003 receive" and read the result across the four tabs.

The seed strains and patients load whether or not the large datasets are present, so the demo runs on the seed graph alone. The WHO catalog merge in step 2 is skipped with a printed note when the catalog file is absent.

## Architecture

The design separates a durable, evidence-grounded platform from a swappable patient layer.

- Knowledge graph. A Memgraph store holding 1,295 mutation nodes drawn from the WHO mutation catalog. The catalog grades all 48,152 of its variants from 1 to 5. The graph loads only the 1,383 rows graded 1 or 2, the variant-drug associations tied to resistance, since the higher groups carry uncertain or no association. Those rows collapse to 1,291 distinct nodes, because a node is keyed by its mutation identifier, so a variant graded against several drugs merges into one node. The four remaining nodes come from the seed strains loaded before the catalog merge. Memgraph speaks the Bolt protocol, so the code reaches it through the standard neo4j Python driver, and the neo4j dependency in requirements.txt is that driver rather than a separate database.

- Rule engine. A forward and backward chaining symbolic engine that classifies isolates as MDR, pre-XDR, or XDR, applies whole-class cross-resistance, and selects between the BPaL and BPaLM regimens. The two modes agree on every classification, exclusion, and alert across all resistance combinations the engine can distinguish, which the test suite checks exhaustively, so the mode the evaluation scores is the mode the application runs.

- Case-based reasoning. Retrieval of over 1,000 synthetic patient cases, used where the rules alone do not determine a regimen.

- Natural-language interface. An LLM layer that generates Cypher from plain English behind a read-only write guard. The query runs in a read transaction that Memgraph rejects on any write, so the database itself is the barrier, and a keyword pre-filter blocks an obvious write before the query runs.

The figure below traces one strain through the graph, from its mutations to the genes and drugs they affect and on to its resistance profile, which is the same path the rule engine walks to reach a classification.

![Strain TB011 traced through the knowledge graph, from its four mutations to the genes and drugs they affect and on to its XDR resistance profile](assets/knowledge_graph.png)

## Results

### Real-world validation of the symbolic core

The rule engine was validated on all 65,588 CRyPTIC isolates carrying a measured drug-susceptibility phenotype. It reproduces the WHO genotypic catalog on 99.8% of isolates, which confirms that the engine reimplements the catalog tiering it encodes rather than adding hidden logic. Measured against phenotype, the engine reaches 83.4% overall accuracy while the catalog reaches 83.5%.

Accuracy alone flatters an imbalanced set, since below-MDR cases make up 73.3% of the isolates and a system that always predicted below-MDR would reach that figure without reasoning at all. Balanced accuracy, the mean of the per-tier sensitivities, is 67.4% for the engine and 67.9% for the catalog, and macro-F1 is 0.662 against 0.665. Sensitivity falls from 91.6% on below-MDR to 61.9% on MDR, 61.5% on pre-XDR, and 54.7% on XDR, while specificity stays above 94% on every resistant tier.

![Grouped bar chart of per-tier sensitivity against measured phenotype, rule engine versus WHO catalog, across below-MDR, MDR, pre-XDR, and XDR. The two systems are within a point of each other on every tier, and sensitivity declines from 92% on below-MDR to 55% on XDR as the tiers grow rarer.](assets/cryptic_tier_sensitivity.png)

The bars frame the result. The engine reaches essentially the same per-tier sensitivity as the catalog it encodes, so the headroom that remains lives in the catalog and the data rather than in the implementation.

Of the 17,523 resistant isolates the engine and the catalog land the same way on all but 105.

| Of 17,523 resistant isolates | Isolates | Share | What it is |
| --- | ---: | ---: | --- |
| Both correct | 10,646 | 60.8% | engine and catalog both right |
| Both wrong | 6,772 | 38.6% | phenotypic resistance with no genotypic marker the catalog recognizes |
| Engine only wrong | 98 | 0.6% | 80 coverage gaps and 18 definitional cases |
| Catalog only wrong | 7 | 0.04% | resistance the catalog misses and the engine catches |

The 6,772 shared errors are a biological upper limit rather than a design flaw, since no genotype-based method can detect resistance that carries no recognized marker.

A paired McNemar test over the 105 discordant isolates gives $\chi^2 = 77.1$ and $p \approx 1.6 \times 10^{-18}$, with 98 falling on the engine side. That figure overstates the difference. Eighty of the 98 reached the engine with no graded mutation at all, because the engine reads EFFECTS while the catalog arm reads PREDICTIONS, and 194 more labeled isolates are absent from the first. Restricting the comparison to isolates the engine actually received gives 18 against 7, with $\chi^2 = 4.0$ and $p = 0.046$. The difference is real and far smaller than the pooled figure suggests. Every one of the 18 is an isolate the engine placed one tier too high, 13 raising MDR to pre-XDR on injectable resistance and 5 raising pre-XDR to XDR, which follows from the pre-2021 definition the engine implements.

### What bounds sensitivity

Sensitivity should be read against what the input permits rather than against 100%. The engine receives only mutations the catalog grades as resistance-conferring, so an isolate carrying no such row arrives with an empty genotype and is classified below the MDR threshold whatever its phenotype. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded row, which places a ceiling of 79.7% on overall sensitivity, 81.7% on MDR, 81.1% on pre-XDR, and 68.4% on XDR.

Against those ceilings the engine recovers 75.8% of what is reachable at MDR, 75.8% at pre-XDR, and 80.0% at XDR. Read raw, the engine appears to perform worst on the most resistant isolates. Read against what the input allows, it performs best there.

The lower XDR ceiling belongs to one collection rather than to resistance itself. Comparing coverage at MDR and at XDR inside each source, six of the seven sources show a median gap of negative 0.6 percentage points, meaning the tier makes no difference to coverage. SEQTREAT2020 shows a gap of 54.4 points, with MDR coverage at 75.6% and XDR at 21.2%. Excluding that one source drops the pooled gap from 17.4 points to 2.7.

### Per-drug resistance calls

Each drug is scored individually against the measured phenotype, with the WHO catalog as the reference arm. Macro-F1 across the 15 drugs is 0.588 for the engine and 0.611 for the catalog. Twelve of the 15 are identical to three decimal places, including both fluoroquinolones, since the catalog already grades levofloxacin and moxifloxacin from the same gyrA call and the class expansion adds nothing there.

The three injectables are the whole of the difference. The engine treats amikacin, kanamycin, and capreomycin as one class, so a mutation against any of them excludes all three. Cross-resistance among the injectables is only partial in practice, and the trade is unfavorable. Precision on amikacin falls from 0.834 to 0.518 and on capreomycin from 0.776 to 0.439, against sensitivity gains of 2.2 and 4.2 percentage points, so F1 falls on all three. The behavior is recorded as a measured property of the heuristic rather than left as an implicit assumption. The scoring runs through `python Evaluation/metrics.py`, which writes `Evaluation/per_drug_results.json`.

### Expert system

The natural-language layer is scored by execution match, pairing each question with a gold query and counting a generated query correct when it returns the same rows. The score is conditional on the model that wrote the Cypher and is reported alongside it. On claude-sonnet-4-6 the layer answers ten of eleven questions correctly, and because generation runs at temperature zero the same result reproduces exactly across runs, including the failure.

The single failure is a lookup where the generated query returned a relationship property without binding the relationship, which the database rejected as an unbound variable. That is an invalid query rather than a wrong answer, and the distinction matters more than the score. The deterministic parts of the layer are covered by the test suite, including the read-only guard, the routing, and the normalization that removes an order clause the database cannot satisfy after an aggregate.

### Case-based reasoning, the experimental component

Regimen accuracy is 77.2% with a bootstrap interval of 74.5% to 79.8%, against a majority-class baseline of 78.5% and a generator ceiling of 80.7%. Outcome accuracy is 74.8% against a baseline of 74.6%. Both arms sit at or below the trivial rule they are measured against.

The aggregate hides where the difficulty lies. Susceptible and MonoResistant each carry exactly one regimen in the generator, so those cases offer nothing to get wrong, and together they are 62% of the cohort. Restricting to the 38% where a choice exists, the baseline scores 43.4%, retrieval scores 40.5%, and the ceiling is 49.3%. The layer matches the trivial rule on the cases that pose no question and loses to it on the ones that do.

| Profile | Regimen accuracy | n |
| --- | ---: | ---: |
| Susceptible | 100.0% | 500 |
| MonoResistant | 98.3% | 120 |
| PolyResistant | 45.0% | 60 |
| MDR | 35.0% | 180 |
| PreXDR | 56.2% | 80 |
| XDR | 31.7% | 60 |

The synthetic cohort exists because no open dataset links genotype, patient profile, regimen, and observed outcome at the scale retrieval needs. Treatment-outcome records of that kind are scarce, held across institutions, and restricted for privacy, which is a common obstacle in clinical machine learning. Generating the cases keeps retrieval transparent and reproducible in the absence of that data, at the cost of measuring the layer against a process whose structure is already known.

### Calibration

Expected calibration error on the raw predicted success probability is 0.0912 and the Brier score is 0.1985.

Temperature scaling was tested and rejected on evidence, since it raised the calibration error to 0.1851. The failure is structural rather than a fitting problem. Dividing the logit can only pull scores toward one half, and the observed success rate is 0.746, so one parameter cannot hold a base rate away from the middle. Platt scaling adds an intercept and lowers the error to 0.0159.

That improvement is not evidence the outcome layer works. The fitted slope across the five folds runs from 0.009 to 0.059, so the calibrated predictor has stopped reading the score and returns close to a constant at the base rate, which is what makes calibration error small. Three further measurements agree. The area under the curve for the raw probability is 0.568, a constant at the base rate scores 0.1895 on Brier and beats both the raw and the scaled predictions, and outcome accuracy already falls inside its baseline interval. The predicted success probability carries almost no information about whether treatment succeeds, and reporting the fitted slope beside the calibration error is what keeps that visible.

## Data

The platform is based on the WHO mutation catalog, second edition, provided as the data file WHO-UCN-TB-2023.7-eng.xlsx. Real-world validation utilizes data from the CRyPTIC consortium release, which includes whole-genome variants graded against the catalog and associated drug-susceptibility phenotypes. The validation set consists of 65,588 isolates with measured phenotypes, scored in full rather than on a held-out split. The two phenotype assays in the release, DST and UKMYC, agree on 94.8% of the 21,568 jointly measured isolates, which sets a label-noise floor beneath the reported accuracy. Where they disagree, UKMYC is the more conservative of the two on every isolate, never the reverse, and the exploratory analysis separates how much of that follows from the smaller UKMYC panel. The synthetic patient cases are transparent and deterministic when using a fixed seed.

The actual datasets are not included in this repository due to their large size. To reproduce the results, download them into a `Datasets/` folder located at the project root. The catalog file WHO-UCN-TB-2023.7-eng.xlsx is from the World Health Organization. The CRyPTIC tables, including EFFECTS.parquet, PREDICTIONS.parquet, DST_MEASUREMENTS.parquet, UKMYC_PHENOTYPES.parquet, and the file DRUG_CODES.csv, originate from the CRyPTIC consortium release on Zenodo. The synthetic patient cases are generated through code and do not require downloading. Accessing the CRyPTIC parquet tables requires the pyarrow engine, which is installed via `requirements.txt`.

The release also ships `DATA_SCHEMA.pdf`, which documents the full set of tables, and `MUTATIONS.parquet`, which this project retains but does not read. The exploratory analysis explains why the rule engine sources its genotypes from EFFECTS instead. A seventh file, `Datasets/cryptic_features.parquet`, is built on first use and cached; delete it after replacing any source table or the scores will be computed from the old data.

## Evaluation

All scoring runs through a single entry point.

```bash
python Evaluation/validation.py
```

This clears and rebuilds the knowledge graph, then runs the expert-system and case-based reasoning validation against it, omitting that part and printing a note if the database or API is unavailable. It then performs the database-free CRyPTIC classification validation. Because it clears the graph first, run it before the demo rather than after, since it discards any case base the app has loaded.

Results are written to `Evaluation/validation_results.json`, replacing the committed reference run. A skipped arm keeps its previous result rather than being erased, and an `arms_this_run` field records which sections were actually recomputed.

The per-drug resistance scoring operates independently and writes `Evaluation/per_drug_results.json`.

```bash
python Evaluation/metrics.py
```

The shared scoring functions, including sensitivity, specificity, precision, F1, balanced accuracy, macro-F1, the McNemar test, and the Brier score, live in `Evaluation/metrics.py`, so the tier scoring in `validation.py` and the per-drug scoring use one definition and remain comparable.

A separate deterministic test suite of 77 tests verifies rule-engine classification, calibration math, the read-only query guard and routing, generator determinism, seed-graph integrity, and the agreement between the two inference modes. It requires no database, API, or datasets and runs from the project root.

```bash
pytest tests/test_core.py
```

The same suite runs in continuous integration on every push, across Python 3.10, 3.11, and 3.12.

### Reproducing

The project reproduces at three levels, each adding to the one before it. The test suite alone needs nothing beyond `pip install -r requirements-dev.txt`, and its 77 tests cover the rule engine, the calibration math, the query guard, and seed-graph integrity. Adding Docker and an Anthropic API key brings up the demo and the expert-system arm on the seed graph, without any dataset download. Adding the `Datasets/` folder unlocks the CRyPTIC and per-drug numbers reported above.

The CRyPTIC, per-drug, and case-based arms are deterministic and reproduce exactly, seeded at 42. The expert-system arm calls a live model and is reported alongside the model that produced it, so it is the one figure expected to move. [DEPLOYME.md](DEPLOYME.md) gives the full procedure.

### Exploratory analysis

[EDA/EDA.ipynb](EDA/EDA.ipynb) documents the data work behind the design, including the label-noise floor, the baselines the case-based layer has to beat, the coverage gap between the PREDICTIONS and EFFECTS tables, and the composition of the seed graph. It shares `baseline_accuracy` with `validation.py`, so the baselines shown there and the ones the system is scored against are the same function rather than two similar ones.

## Limitations

- The patient layer is synthetic because no open dataset links genotype, regimen, and outcome at the scale a case-based recommender needs. This data scarcity is a well-known challenge in healthcare machine learning, and it is the direct reason the rare resistant classes evaluate poorly.

- The case-based similarity weights are domain-informed priors set by hand, not values learned from data, and tuning them is future work. The region and outcome tables in the case generator follow the same pattern, since they carry real structure from the WHO regions while their magnitudes stay synthetic rather than transcribed from any WHO release.

- The regimen layer is scored by exact match to the labeled regimen, which penalizes it for optimizing treatment outcome instead, so part of the measured shortfall is a metric mismatch rather than a modeling error.

- CRyPTIC provides genotype and phenotype but not treatment outcomes, so it validates classification only and cannot validate the regimen and outcome layer.

- The rule engine implements a scoped pre-2021 XDR definition, documented as a deliberate choice rather than the current Group A based standard, because the release carries no bedaquiline or linezolid phenotype the current definition would need.

- The mono and poly split counts every resistant drug rather than the first-line set alone, which follows the CDC wording and the extension to second-line agents that WHO anticipated for surveillance. WHO's own definition still reads first-line only, so this is a stated deviation. It moves no tier at MDR or above, which the test suite holds structurally.

- Sensitivity is bounded by input coverage. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded mutation and cannot be reached by any rule, so every sensitivity figure should be read against the ceiling reported beside it rather than against 100%.

- Cross-resistance among the injectables is modeled as a whole class, which the per-drug scoring shows costs more precision than it gains sensitivity. The behavior is measured and reported rather than corrected, since narrowing it would change the tier definitions the engine is scored against.

- The rule engine does not model ethionamide, so the inhA cross-resistance that links isoniazid and ethionamide is out of scope. This is a named boundary rather than an oversight.

- Genotype-based resistance prediction is bounded by discordance. Of the 17,523 resistant isolates, 6,772 carry phenotypic resistance with no genotypic marker the catalog recognizes, so that share is not recoverable from the catalog by any rule-based method.

## Future work

Several directions would extend the work, and they fall into two groups, the data the system can reach and the way its layers are scored.

The most significant data gap is the synthetic case base. Validating the case-based layer with the TB Portals dataset, which includes actual treatment outcomes, would replace the cohort where real signals are most needed. Real data would strengthen the result without necessarily raising the accuracy, since resistant cases stay rare even in large collections. A trained model could push the results toward the other ceiling. Training such a model with the full genome-wide variant table and the minimum inhibitory concentration magnitudes would explore how much of the genotype-phenotype mismatch can be explained beyond the curated catalog.

The remaining instructions can enhance how the system is evaluated and how it manages rare classes. The regimen layer now receives a score based on an exact match to the labeled regimen, which penalizes it for emphasizing treatment outcomes and guideline adherence instead. This shift to the actual goal is the simplest change, as it requires no additional data and directly fixes the metric mismatch. Moreover, confidence-gated deferral enables a sparse retrieval neighborhood to defer to the rule engine and report coverage along with accuracy, turning rare-class scarcity into well-calibrated behavior. Lastly, the whole-class injectable rule groups amikacin, kanamycin, and capreomycin together, and the per-drug scoring shows it over-calls amikacin and capreomycin against measured phenotype, costing 31.6 and 33.7 points of precision for gains of 2.2 and 4.2 in sensitivity. Tying cross-resistance to the gene instead, with rrs conferring broad resistance and eis favoring kanamycin, would recover most of that precision without moving any tier.



## References

### Case-Based Reasoning

1. Kolodner, J. L. (1992). An Introduction to Case-Based Reasoning. *Artificial Intelligence Review*, 6(1), 3-34.
2. Main, J., Dillon, T. S., & Shiu, S. C. K. (2001). A Tutorial on Case-Based Reasoning. *Soft Computing in Case Based Reasoning*, 1-28.
3. Goel, A. K., & Díaz-Agudo, B. (2017). What's Hot in Case-Based Reasoning. *Proceedings of AAAI-17*.
4. Das, R., Godbole, A., Dhuliawala, S., Zaheer, M., & McCallum, A. (2020). A Simple Approach to Case-Based Reasoning in Knowledge Bases. *Automated Knowledge Base Construction (AKBC)*.

### WHO Guidelines

5. World Health Organization. (2023). *Catalogue of mutations in Mycobacterium tuberculosis complex and their association with drug resistance* (2nd ed.).
6. World Health Organization. (2025). *WHO consolidated guidelines on tuberculosis: Module 4: Treatment and care*.
7. Walker, T. M., et al. (2022). The 2021 WHO catalogue of Mycobacterium tuberculosis complex mutations. *The Lancet Microbe*, 3(4), e265-e273.

### Calibration

8. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 1321-1330.

### Treatment Evidence

9. Nyang'wa, B.-T., et al. (2022). A 24-Week, All-Oral Regimen for Rifampin-Resistant Tuberculosis. *New England Journal of Medicine*, 387(25), 2331-2343. (TB-PRACTECAL; BPaLM)
10. Conradie, F., et al. (2020). Treatment of Highly Drug-Resistant Pulmonary Tuberculosis. *New England Journal of Medicine*, 382(10), 893-902. (Nix-TB; BPaL)

### Datasets

11. The CRyPTIC Consortium. (2022). A data compendium associating the genomes of 12,289 *Mycobacterium tuberculosis* isolates with quantitative resistance phenotypes to 13 antibiotics. *PLOS Biology*, 20(8), e3001721.
12. Rosenthal, A., et al. (2017). The TB Portals: an Open-Access, Web-Based Platform for Global Drug-Resistant-Tuberculosis Data Sharing and Analysis. *Journal of Clinical Microbiology*, 55(11). doi:10.1128/JCM.01013-17.

## License

Released under the MIT License.
