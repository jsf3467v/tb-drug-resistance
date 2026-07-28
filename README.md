# TB Drug-Resistance Decision Support System

[![tests](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml)
[![CI](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A hybrid decision-support prototype for *Mycobacterium tuberculosis* drug resistance. It combines a WHO-grounded knowledge graph, a symbolic rule engine, case-based reasoning over synthetic patient cases, and an LLM-driven natural-language query layer. Its symbolic core is validated against real-world resistance measurements from the CRyPTIC consortium.

## Objective

This system was developed as a graduate course project and serves as a portfolio piece. Its goal is to show how several methods work together as one pipeline. The knowledge graph supplies evidence-based structure, and the rule engine produces transparent classifications and treatment decisions. Case-based reasoning addresses cases that the rules cannot settle, and a natural-language layer translates questions into graph queries. Drug-resistant tuberculosis was chosen because each method has a clear role here, and the reasoning must remain auditable rather than hidden.

Choosing healthcare also involved dealing with imperfect data. Resistance measurements are often incomplete and noisy, and sometimes the two phenotype assays disagree on the same isolate. Supplemental data that could fill these gaps was difficult to obtain. Records linking genotype, treatment, and outcome are rare, and the outcome data needed for the case base was not available at the required scale. That shortage is why the patient layer is synthetic. The goal was to measure these limitations and report each one next to the result it affects, rather than to develop a 
clinical tool. Problems like these are common in healthcare, where labels disagree and outcome records stay scattered across institutions. The project will keep developing as new methods, data, and research become available.

## Overview

Drug-resistant tuberculosis requires reasoning that is both auditable and grounded in current evidence. This system combines an explicit symbolic layer, where each resistance classification links to a WHO catalog rule, with a case-based layer that leverages prior patient experience when guidelines are absent. It features a natural-language interface that converts questions into graph queries within a read-only environment, and a Streamlit front end that visualizes the complete reasoning process.

The synthetic patient layer and the genotype-phenotype prediction ceiling are treated as measured limits rather than hidden ones. A short demo video of the front end and its reasoning trace is in progress.

## Interactive demo

The Streamlit front end is the system in use. A clinical question drives the full hybrid pipeline and returns an auditable recommendation together with the reasoning behind it.

![The app answering a treatment query for patient P003, showing the diagnosis, the XDR classification, and the contraindicated drugs alongside the mutations that exclude them](assets/query-results.png)

A question such as "What treatment should patient P003 receive" is answered across four tabs.

- Query Results carries the direct answer, the strain and its classification, the recommended regimen, and a table of contraindicated drugs tied to the mutations that rule them out.
- Expert System exposes the rule-engine trace, the canonical gene fraction, the rules that fired, and the regimen with its drug exclusions.
- Case-Based Reasoning retrieves the nearest matches from the 1,000 synthetic patient cases and reports a success rate and a confidence band.
- Technical Details shows the Cypher that the natural-language layer generated from the question, so the path from text to graph query stays visible.

The Expert System tab carries the symbolic trace. Strain TB011 classifies as XDR under rule RC002, which selects BPaL, and each excluded drug names the mutation that ruled it out.

![The Expert System tab for strain TB011, showing the XDR classification with its rule and source, the BPaL regimen and its duration, and levofloxacin excluded by the gyrA p.Asp94Gly mutation](assets/expert-system.png)

The Case-Based Reasoning tab answers the same question from prior cases rather than from rules. The ten neighbors match this patient, of which three succeeded, and the confidence band reports moderate rather than high because the neighbors disagree on outcome. The screenshot predates the smoothing described under Calibration, so the reported share now reads slightly higher than the image shows.

![The Case-Based Reasoning tab for patient P003, showing the XDR patient profile, ten similar cases, a 30 percent success rate, and moderate confidence of 0.58](assets/cbr.png)

![The Technical Details tab showing the Cypher generated from the question together with the nine graph results returned](assets/tech-details.png)

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

- Rule engine. A forward and backward chaining symbolic engine that classifies isolates as MDR, pre-XDR, or XDR, applies whole-class cross-resistance, and selects between the BPaL and BPaLM regimens. The two modes agree on every classification, exclusion, alert, regimen, inclusion, and monitoring entry, which the test suite checks over every combination of the resistance flags the engine can tell apart, so the mode the evaluation scores is the mode the application runs. The list of rules fired is the one field outside that guarantee, since forward chaining reaches the pre-XDR rule on an XDR isolate and backward chaining stops before it. That list is the trace the Expert System tab renders, so it reflects the mode that ran.

- Case-based reasoning. Retrieval over 1,000 synthetic patient cases, used where the rules alone do not determine a regimen.

- Natural-language interface. An LLM layer that generates Cypher from plain English behind a read-only write guard. The query runs in a read transaction that Memgraph rejects on any write, so the database itself is the barrier, and a keyword pre-filter blocks an obvious write before the query runs.

The figure below traces one strain through the graph, from its mutations to the genes and drugs they affect and on to its resistance profile, which is the same path the rule engine walks to reach a classification.

![Strain TB011 traced through the knowledge graph, from its four mutations to the genes and drugs they affect and on to its XDR resistance profile](assets/knowledge_graph.png)

## Results

### Real-world validation of the symbolic core

The rule engine was validated on all 65,588 CRyPTIC isolates with a measured drug-susceptibility phenotype. It reproduces the WHO genotypic catalog on 99.8% of isolates. That figure is close to definitional rather than independent evidence, since both arms grade the same variants against the same catalog. The remaining 0.2% comes from the two arms reading different tables, the engine from EFFECTS and the catalog arm from PREDICTIONS. Whole-class cross-resistance contributes nothing to this comparison, because it writes only into the exclusion list and never into the facts the classifier reads, so it is visible in the per-drug arm alone. Measured against phenotype, the engine achieves 83.4% overall accuracy, while the catalog reaches 83.5%.

Accuracy alone can be misleading for an imbalanced set, since below-MDR cases make up 73.3% of isolates and a system that always predicted below-MDR would reach that figure without reasoning. Balanced accuracy, the mean of the per-tier sensitivities, is 67.4% for the engine and 67.9% for the catalog, and macro-F1 is 0.662 against 0.665. Sensitivity falls from 91.6% on below-MDR to 61.9% on MDR, 61.5% on pre-XDR, and 54.7% on XDR, while specificity remains above 94% on every resistant tier.

![Grouped bar chart of per-tier sensitivity against measured phenotype, rule engine versus WHO catalog, across below-MDR, MDR, pre-XDR, and XDR. The two systems are within a point of each other on every tier, and sensitivity declines from 92% on below-MDR to 55% on XDR as the tiers grow rarer.](assets/cryptic_tier_sensitivity.png)

The bars outline the result. The engine attains nearly identical per-tier sensitivity as the catalog it encodes, meaning the remaining headroom resides in the catalog and data, not in the implementation. Out of 17,523 resistant isolates, the two arms differ in correctness on 105. They differ in the label they assign on about 123, since the remainder are isolates both arms place wrong under different names.

| Of 17,523 resistant isolates | Isolates | Share | What it is |
| --- | ---: | ---: | --- |
| Both correct | 10,646 | 60.8% | engine and catalog both right |
| Both wrong | 6,772 | 38.6% | phenotypic resistance with no genotypic marker the catalog recognizes |
| Engine only wrong | 98 | 0.6% | 80 coverage gaps and 18 definitional cases |
| Catalog only wrong | 7 | 0.04% | resistance the catalog misses and the engine catches |

The 6,772 shared errors are a biological upper limit rather than a design flaw, since no genotype-based method can detect resistance that carries no recognized marker.

A paired McNemar test over the resistant-truth isolates gives $\chi^2 = 77.1$ and $p \approx 1.6 \times 10^{-18}$, with 98 falling on the engine side. That figure overstates the difference. Eighty of the 98 reached the engine with no graded mutation at all, because the engine reads EFFECTS while the catalog arm reads PREDICTIONS, and 194 more labeled isolates are absent from the first. Restricting the comparison to isolates the engine actually received gives 18 against 7, with $\chi^2 = 4.0$ and $p = 0.046$. The difference is real and far smaller than the pooled figure suggests. The test conditions on resistant truth, so it reads 105 of the 124 total disagreements. The 91 isolates it separates the two arms by, 98 against 7, are the same 91 that separate the two overall accuracy figures, because the disagreements it sets aside fall on isolates both arms score wrong. Every one of the 18 is an isolate the engine placed one tier too high, 13 raising MDR to pre-XDR and 5 raising pre-XDR to XDR. All 18 carry a kanamycin call graded in EFFECTS and 17 carry amikacin as well, so these are the same table gap surfacing as a one-tier bump rather than a total miss. The tier definition is not the cause, since the label and the classification rules read the same drug-class sets from config.

### What bounds precision

The tier label reads an untested drug as susceptible. Pre-XDR and XDR are separated from MDR by fluoroquinolone and injectable results, and 38,674 isolates carry a result for both, which is 59.0% of the cohort. An isolate never tested on either class is capped at MDR by its label however resistant its genotype is, so a correct genotypic call above MDR is scored a false positive on an isolate that was never measured.

Scoring the same tiers on the isolates that were measured separates the two readings.

| Tier | Precision, full cohort | Precision, tested | Resistant, full | Resistant, tested |
| --- | ---: | ---: | ---: | ---: |
| MDR | 0.758 | 0.852 | 10,335 | 8,491 |
| pre-XDR | 0.483 | 0.616 | 4,725 | 4,605 |
| XDR | 0.482 | 0.629 | 2,463 | 2,463 |

The macro-F1 rises from 0.662 to 0.708 for the engine and from 0.665 to 0.712 for the catalog, and balanced accuracy from 67.4% to 68.9% and from 67.9% to 69.4%. Overall accuracy falls from 83.4% to 80.8%, because the restriction removes mostly below-MDR isolates and leaves a harder mix behind.

The gain's clarity comes from the restriction eliminating false positives rather than difficult positives. Each XDR isolate already contains both classes, since the label cannot reach XDR without them, so the count remains at 2,463 and sensitivity stays at 54.7%, while precision improves by 14.7 points. Pre-XDR retains 4,605 of its 4,725, with a sensitivity that barely shifts but a 13.3-point increase. The restriction on the 26,914 isolates removed errors contributed by those at the top two tiers without reducing correct calls.

The per-drug arm did not include this because it scores only drugs with results for the relevant isolates. The tier arm now records both readings, with the restricted one labeled under `second_line_covered`.

### What bounds sensitivity

The sensitivity should be assessed based on what the input permits, not against a perfect 100%. The engine only recognizes mutations that it categorizes as resistance-conferring. Consequently, an isolate without such mutations has an empty genotype and is considered below the MDR threshold, regardless of its phenotype. Of the 17,523 isolates identified as MDR or more resistant, 3,562 lack graded mutations, which sets an upper sensitivity limit of 79.7%. Specifically, the sensitivity is 81.7% for MDR, 81.1% for pre-XDR, and 68.4% for XDR. Within these limits, the engine detects 75.8% of the possible at MDR, 75.8% at pre-XDR, and 80.0% at XDR. Although the raw output suggests reduced effectiveness against highly resistant isolates, the relative performance under input constraints indicates it performs better in those cases.

### Per-drug resistance calls

Each drug is scored individually against the measured phenotype, with the WHO catalog as the reference arm. Every figure is computed on the isolates carrying a result for that drug, which ranges from 19,948 for capreomycin to 59,869 for rifampin, so each row reports its own denominator beside the resistant count rather than the 65,588 in the header. Macro-F1 across the 15 drugs is 0.588 for the engine and 0.611 for the catalog. Both arms score the same isolates for a given drug, so the difference carries a paired McNemar test rather than two separate intervals. Twelve of the 15 drugs show no discordant isolate at all, including both fluoroquinolones, since the catalog already grades levofloxacin and moxifloxacin from the same gyrA call and the class expansion adds nothing there. Those twelve agree isolate by isolate rather than only in aggregate.

The main distinction lies in the three injectables. The engine considers amikacin, kanamycin, and capreomycin as a single class, so a mutation affecting any one of them results in resistance to all three. In practice, cross-resistance among these injectables is only partial, making the trade-off generally disadvantageous. Precision on amikacin falls from 0.834 to 0.518 and on capreomycin from 0.776 to 0.439, against sensitivity gains of 2.2 and 4.2 percentage points, so F1 falls on all three. The paired test puts the discordance at 1,524 isolates for amikacin with $\chi^2 = 1280.6$, 1,343 for capreomycin with $\chi^2 = 1061.5$, and 149 for kanamycin with $\chi^2 = 116.9$, each at $p < 10^{-26}$. Read the ratios rather than the p-values. Because the engine calls a strict superset of the catalog, the test asks only whether the added calls are more often wrong than right, and at these cohort sizes the p-value reflects the denominator. The three tests are also driven by the same rrs and eis mutations in the same isolates, so they are one finding measured three ways rather than three confirmations. The behavior is recorded as a measured property of the heuristic rather than left as an implicit assumption. The scoring runs through `python Evaluation/metrics.py`, which writes `Evaluation/per_drug_results.json`.

### Expert system

The natural-language layer is evaluated based on execution match, where each question is paired with a gold query, and a generated query is deemed correct if it returns the same result set. The score depends on the model that produced the Cypher and is reported alongside it. On claude-sonnet-4-6, the layer correctly answers ten out of eleven questions, with a Wilson interval ranging from 62.3% to 98.4%, showing that eleven questions do not reflect the precision suggested by a simple percentage. Generation occurs at temperature zero, ensuring consistent scores and errors across runs, including the specific failed query and the error it raises. While the generated Cypher can vary slightly between runs, the reported arm includes its model rather than a fixed output.

The only failure involved a lookup where the generated query returned a relationship property without binding the relationship, leading to rejection as an unbound variable. This is an invalid query rather than a wrong answer, and this distinction is more important than the score itself. The deterministic elements of the layer, such as the read-only guard, routing, and normalization (which removes order clauses the database cannot satisfy after an aggregate but keeps them when a LIMIT depends on it), are verified by the test suite.

### Case-based reasoning, the experimental component

Regimen accuracy reaches 77.2%, with a bootstrap interval of 74.5% to 79.8%. This is just below the majority-class baseline of 79.3% and close to the ceiling of 79.8%. The ceiling excludes year because the generator selects regimens from profile and year, but year is independent of other features, making it unusable by predictors. Outcome accuracy is 74.8%, slightly above the baseline of 74.6%. Both metrics are at or below the baseline. 

The overall summary conceals the core challenge. Susceptible and MonoResistant cases each contribute exactly one regimen, so these cases can't be misclassified, making up 62% of the cohort. Focusing on the remaining 38% with multiple options, the baseline is 45.5%, retrieval scores 40.5%, and the ceiling is 46.9%. The model matches the trivial rule on cases with no challenge but underperforms on more complex cases. The baseline falls within the bootstrap interval for the entire cohort, and the restricted analysis, based on 380 cases, shows a small gap that is a point estimate rather than a statistically significant difference.

| Profile | Regimen accuracy | n |
| --- | ---: | ---: |
| Susceptible | 100.0% | 500 |
| MonoResistant | 98.3% | 120 |
| PolyResistant | 45.0% | 60 |
| MDR | 35.0% | 180 |
| PreXDR | 56.2% | 80 |
| XDR | 31.7% | 60 |

Retrieval often crosses resistance profiles for 15.4% of neighbors because the profile carries a weight of 0.32 when compared to an ordinal similarity, where an adjacent tier still scores 0.8. A neighbor one tier away loses 0.064, while a case with the same-profile but different previous treatment, HIV status, and region loses 0.41, since a region mismatch retains half its weight instead of losing all of it. Boundary crossings are common. When comparing to the per-profile baseline on identical folds, shipped retrieval is 59, while discordant cases are 78, at a significance level of $p = 0.12$. The generator selects the regimen based solely on profile and year; however, the year is not visible to retrieval. Therefore, the maximum achievable weighting of features that retrieval can apply on this cohort is represented by the per-profile lookup. This limits what the synthetic cases reveal about retrieval and doesn’t necessarily 
indicate a need for weighting adjustments.

### Calibration

The predicted probability is the share of the ten retrieved neighbors that succeeded, smoothed by adding one success and one failure. Smoothing is what makes the rest of this section readable. A raw share of ten neighbors lands on exactly one whenever all ten succeeded, which happened on about one case in thirteen, and the logit that both scaling methods fit is unbounded there. Those few saturated scores otherwise dominate the likelihood and drive the fit on their own.

Expected calibration error on the smoothed probability is 0.0942 and the Brier score is 0.1951. A constant prediction at the base rate scores 0.1895 on Brier and beats it. That constant is fit on the same outcomes it scores, which makes it the strongest baseline of its kind and the conservative direction for this comparison.

Temperature scaling fits at 1.113 on average, with the five folds running from 1.05 to 1.238, and moves the calibration error from 0.0942 to 0.103. It has close to nothing to correct because the problem is direction rather than spread. The middle bins are under-confident, where accuracy runs above the stated probability, while the top bin is over-confident. A single parameter that divides the logit moves every score the same way and cannot answer both.

Platt scaling adds an intercept, fits slopes from 0.252 to 0.517, and lowers the calibration error to 0.018 with a Brier score of 0.1891.

That improvement is not evidence the outcome layer works. Slopes well below one mean the fit is shrinking the score toward the base rate, and most of the gain in calibration error is that shrinkage rather than the score carrying more signal. Neither scaling changes the ranking, so the area under the curve of 0.562 is the figure that measures the signal, and it sits just above chance. The Platt Brier score of 0.1891 clears the constant at 0.1895 by a margin too small to rest anything on, and outcome accuracy already falls inside its baseline interval. The predicted success probability carries little information about whether treatment succeeds, and reporting the fitted slope beside the calibration error is what keeps that visible.


## Data

The platform is based on the WHO mutation catalog, second edition, provided as the data file WHO-UCN-TB-2023.7-eng.xlsx. Real-world validation utilizes data from the CRyPTIC consortium release, which includes whole-genome variants graded against the catalog and associated drug-susceptibility phenotypes. The validation set consists of 65,588 isolates with measured phenotypes, scored in full rather than on a held-out split. The two phenotype assays in the release, DST and UKMYC, agree on 94.8% of the 21,568 jointly measured isolates, which sets a label-noise floor beneath the reported accuracy. Where they disagree, UKMYC is the more conservative of the two on every isolate, never the reverse, and the exploratory analysis separates how much of that follows from the smaller UKMYC panel. The synthetic patient cases are transparent and deterministic when using a fixed seed.

The actual datasets are not included in this repository due to their large size. To reproduce the results, download them into a `Datasets/` folder located at the project root. The catalog file WHO-UCN-TB-2023.7-eng.xlsx is from the World Health Organization. The CRyPTIC tables, including EFFECTS.parquet, PREDICTIONS.parquet, DST_MEASUREMENTS.parquet, UKMYC_PHENOTYPES.parquet, and the file DRUG_CODES.csv, originate from the CRyPTIC consortium release on Zenodo. The synthetic patient cases are generated through code and do not require downloading. Accessing the CRyPTIC parquet tables requires the pyarrow engine, which is installed via `requirements.txt`.

The release also ships `DATA_SCHEMA.pdf`, which documents the full set of tables, and `MUTATIONS.parquet`, which this project retains but does not read. The exploratory analysis explains why the rule engine sources its genotypes from EFFECTS instead. A seventh file, `Datasets/cryptic_features.parquet`, is built on first use and cached. It rebuilds itself whenever a source table, `feature_engineering.py`, or `config.py` is newer than the cache, so replacing a table is enough.

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

The shared scoring functions, including sensitivity, specificity, precision, F1, balanced accuracy, macro-F1, the McNemar test, and the Brier score, live in `Evaluation/metrics.py`, so the tier scoring in `validation.py` and the per-drug scoring measure the same quantities the same way. The reference arm is not the same object in both. The tier arm reads the catalog profile from `PREDICTIONS.parquet`, while the per-drug arm reads it from `EFFECTS.parquet` so that both columns come from one table. Compare within an arm rather than across the two files.

A separate deterministic test suite of 121 tests verifies rule-engine classification, calibration math, the read-only query guard and routing, generator determinism, seed-graph integrity, and the agreement between the two inference modes. It requires no database, API, or datasets and runs from the project root.

```bash
pytest tests/test_core.py
```

The same suite runs in continuous integration on every push, across Python 3.10, 3.11, and 3.12.

### Reproducing

The project reproduces at three levels, each adding to the one before it. The test suite alone needs nothing beyond `pip install -r requirements-dev.txt`, and its 121 tests cover the rule engine, the calibration math, the query guard, and seed-graph integrity. Adding Docker and an Anthropic API key brings up the demo and the expert-system arm on the seed graph, without any dataset download. Adding the `Datasets/` folder unlocks the CRyPTIC and per-drug numbers reported above.

The CRyPTIC, per-drug, and case-based arms are deterministic and reproduce exactly, seeded at 42. The expert-system arm calls a live model and is reported alongside the model that produced it, so it is the one figure expected to move. [DEPLOYME.md](DEPLOYME.md) gives the full procedure.

### Exploratory analysis

[EDA/EDA.ipynb](EDA/EDA.ipynb) documents the data work behind the design, including the label-noise floor, the baselines the case-based layer has to beat, the coverage gap between the PREDICTIONS and EFFECTS tables, and the composition of the seed graph. It shares `baseline_accuracy` with `validation.py`, so the baselines shown there and the ones the system is scored against are the same function rather than two similar ones.

## Limitations

- The patient layer is synthetic because no open dataset links genotype, regimen, and outcome at the scale a case-based recommender needs. This data scarcity is a well-known challenge in healthcare machine learning, and it is the direct reason the rare resistant classes evaluate poorly.

- The case-based similarity weights are domain-informed priors set by hand, not values learned from data, and tuning them is future work. The region and outcome tables in the case generator follow the same pattern, since they carry real structure from the WHO regions while their magnitudes stay synthetic rather than transcribed from any WHO release.

- The regimen layer is scored by exact match to the labeled regimen, which penalizes it for optimizing treatment outcome instead, so part of the measured shortfall is a metric mismatch rather than a modeling error.

- CRyPTIC provides genotype and phenotype but not treatment outcomes, so it validates classification only and cannot validate the regimen and outcome layer.

- The rule engine implements a scoped pre-2021 XDR definition, documented as a deliberate choice rather than the current Group A based standard. The release does carry bedaquiline and linezolid phenotypes, but on few isolates and at genotypic sensitivity of 0.40 and 0.23. Those are the weakest of the four drugs the current definition reads, below levofloxacin at 0.66 and moxifloxacin at 0.70, so that definition would rest on the thinnest columns in the data.

- Classification and treatment are keyed to different guideline revisions. An isolate is tiered by the pre-2021 definition and routed by WHO 2022 regimen rules, so an injectable-resistant isolate is labeled pre-XDR and still receives BPaLM. Each half is correct on its own terms; the pairing follows from the tiering choice above.

- A regimen is a guideline recommendation, not a per-patient prescription. Where an isolate is resistant to a component drug, the engine keeps the regimen and names it in a contraindicated field rather than substituting, since choosing the replacement is a clinical decision the rule base does not model.

- Both inference paths return the same classification, regimen, exclusions, and monitoring. The list of rules fired can still differ on one point. Forward chaining evaluates every rule, so an XDR isolate also fires the pre-XDR rule whose criteria it meets, while backward chaining stops once the XDR goal is proved and never reaches that rule. The recommendations are identical either way, and the difference is what the two search strategies visit rather than what they conclude.

- The mono and poly split counts every resistant drug rather than the first-line set alone, which follows the CDC wording and the extension to second-line agents that WHO anticipated for surveillance. WHO's own definition still reads first-line only, so this is a stated deviation. It moves no tier at MDR or above, which the test suite holds structurally.

- Precision is bounded by phenotype coverage. The label reads an untested drug as susceptible, and only 59.0% of the cohort carries both a fluoroquinolone and an injectable result, so a correct genotypic call above MDR can be scored a false positive on an isolate that was never tested. On the measured isolates, pre-XDR precision moves from 0.483 to 0.616 and XDR from 0.482 to 0.629 with sensitivity flat, reported under `second_line_covered`.

- Sensitivity is bounded by input coverage. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded mutation and cannot be reached by any rule, so every sensitivity figure should be read against the ceiling reported beside it rather than against 100%.

- Cross-resistance among the injectables is modeled as a whole class, which the per-drug scoring shows costs more precision than it gains sensitivity. The behavior is measured and reported rather than corrected, since narrowing it would change the tier definitions the engine is scored against.

- The rule engine does not model ethionamide, so the inhA cross-resistance that links isoniazid and ethionamide is out of scope. This is a named boundary rather than an oversight.

- Genotype-based resistance prediction is bounded by discordance. Of the 17,523 resistant isolates, 6,772 carry phenotypic resistance with no genotypic marker the catalog recognizes, so that share is not recoverable from the catalog by any rule-based method.

## Future work

Several directions would extend the work, and they fall into two groups, the data the system can reach and the way its layers are scored.

The most significant data gap is the synthetic case base. Validating the case-based layer with the TB Portals dataset, which includes actual treatment outcomes, would replace the cohort where real signals are most needed. Real data would strengthen the result without necessarily raising the accuracy, since resistant cases stay rare even in large collections. A trained model could push the results toward the other ceiling. Training such a model with the full genome-wide variant table and the minimum inhibitory concentration magnitudes would explore how much of the genotype-phenotype mismatch can be explained beyond the curated catalog.

The remaining instructions can enhance how the system is evaluated and how it manages rare classes. The regimen layer now receives a score based on an exact match to the labeled regimen, which penalizes it for emphasizing treatment outcomes and guideline adherence instead. This shift to the actual goal is the simplest change, as it requires no additional data and directly fixes the metric mismatch. Moreover, confidence-gated deferral enables a sparse retrieval neighborhood to defer to the rule engine and report coverage along with accuracy, turning rare-class scarcity into well-calibrated behavior. Lastly, the whole-class injectable rule groups amikacin, kanamycin, and capreomycin together, and the per-drug scoring shows it over-calls amikacin and capreomycin against measured phenotype, costing 31.7 and 33.7 points of precision for gains of 2.2 and 4.2 in sensitivity. Tying cross-resistance to the gene instead, with rrs conferring broad resistance and eis favoring kanamycin, would recover most of that precision without moving any tier.



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
