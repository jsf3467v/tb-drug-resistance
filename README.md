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

The design separates a durable, evidence-based platform from a swappable patient layer.

- Knowledge graph. A Memgraph database contains 1,295 mutation nodes sourced from the WHO mutation catalog. This catalog rates 48,152 variant and drug pairs on a scale from 1 to 5, covering 30,699 unique variants. The graph loads only the 1,383 pairs graded 1 or 2 that are associated with resistance, since higher grades indicate uncertain or no association. These pairs consolidate into 1,291 distinct nodes because nodes are identified by mutation, so a variant linked to multiple drugs merges into a single node. The remaining four nodes are seed mutations the catalog does not grade, since 19 of the 23 seed mutations share an identifier with a catalog entry and merge into it. Memgraph speaks the Bolt protocol, so the code reaches it through the neo4j Python driver, and the neo4j dependency in requirements.txt is that driver rather than a separate database.

- Rule engine. A symbolic engine using forward and backward chaining classifies isolates as MDR, pre-XDR, or XDR. It also applies whole-class cross-resistance and chooses between the BPaL and BPaLM regimens. Backward chaining aims for a specific goal, answering either a treatment or classification question. When targeting a treatment goal, both modes agree on classifications, exclusions, alerts, regimens, inclusions, and monitoring entries across all resistance flag combinations tested. For a classification goal, agreement includes classifications, exclusions, and alerts; the regimen and downstream elements are withheld since no treatment is requested. The list of rules fired sits outside that guarantee, because forward chaining reaches the pre-XDR rule on an XDR isolate while backward chaining stops earlier. This list appears in the Expert System tab as the trace, reflecting the mode used.

- Case-based reasoning. Retrieval over 1,000 synthetic patient cases that returns a regimen, a success rate, and a confidence band from the nearest neighbors. The model uses seven hand-set similarity weights rather than learned ones.

- Natural-language interface. An Anthropic large language model layer converts plain English into Cypher queries. It is protected by a read-only guard. The query executes in a read transaction that Memgraph rejects if any write is attempted, making the database the primary barrier. Additionally, a keyword pre-filter prevents obvious write attempts before the query executes.

The figure below follows a single strain through the graph, from its mutations to the genes and drugs they impact, and finally to its resistance profile. The rule engine retrieves mutation-to-drug associations directly from this structure without traversing it. Therefore, the figure illustrates the source of its data rather than a specific path taken.

![Strain TB011 traced through the knowledge graph, from its four mutations to the genes and drugs they affect and on to its XDR resistance profile](assets/knowledge_graph.png)

## Results

### Real-world validation of the rule engine

The rule engine was validated on all 65,588 CRyPTIC isolates carrying a measured drug-susceptibility phenotype. Both arms read the EFFECTS and PREDICTIONS tables directly. The knowledge graph holds the same catalog in a second representation but does not appear in any figure reported here, as it serves the seed strains and the natural-language interface instead.

The engine reproduces the WHO genotypic catalog on 99.8% of isolates. That figure is closer to definitional than to independent evidence, since both arms grade the same variants against the same catalog and apply the same tier definitions. The remaining 0.2% difference arises because one method reads from the EFFECTS table and the other from PREDICTIONS. Both tables contain precomputed catalog verdicts and differ mainly in detail. The EFFECTS table provides a per-variant grade that the engine then aggregates into a drug-level call, while PREDICTIONS already includes that aggregation. Therefore, the engine's contribution here is limited to the aggregation process, not the resistance grading or tier definitions, which are consistent across both methods. The whole-class cross-resistance affects only the exclusion list and does not influence the classifier's core facts, so it appears only in the per-drug analysis. When measured against phenotype, the engine achieves 83.4% overall accuracy, while the catalog reaches 83.5%.

Relying solely on accuracy can be misleading for an imbalanced dataset, as below-MDR cases comprise 73.3% of isolates. A model that always predicts below-MDR would achieve this proportion without actual reasoning. Balanced accuracy, which averages the per-tier sensitivities, is 67.4% for the engine and 67.9% for the catalog, with macro-F1 scores of 0.662 and 0.665, respectively. Sensitivity drops from 91.6% for below-MDR to 61.9% for MDR, 61.5% for pre-XDR, and 54.7% for XDR, while specificity remains above 94% across all resistance tiers.

![Grouped bar chart of per-tier sensitivity against measured phenotype, rule engine versus WHO catalog, across below-MDR, MDR, pre-XDR, and XDR. The two systems are within a point of each other on every tier, and sensitivity declines from 92% on below-MDR to 55% on XDR as the tiers grow rarer.](assets/cryptic_tier_sensitivity.png)

The bars illustrate the results. The engine reaches almost the same sensitivity per tier as the catalog it encodes, suggesting that most of the remaining potential is in the catalog and data rather than the implementation. The two arms assign different labels to 124 isolates throughout the cohort. Among the 17,523 isolates labeled as resistant truth, they disagree on 105 cases; most other disagreements happen on isolates both arms mislabel, but under different names.

| Of 17,523 resistant isolates | Isolates | Share | What it is |
| --- | ---: | ---: | --- |
| Both correct | 10,646 | 60.8% | engine and catalog both right |
| Both wrong | 6,772 | 38.6% | both arms assign a tier and both are wrong |
| Engine only wrong | 98 | 0.6% | 80 coverage gaps and 18 one-tier over-calls |
| Catalog only wrong | 7 | 0.04% | resistance the catalog misses and the engine catches |

The 6,772 cover two different failures. The first is resistance the genotype does not show, which no method can reach. The engine places 4,314 isolates with resistant truth below MDR, and 3,562 of those carry no graded mutation at all. That count spans the whole cohort rather than this one row, since 80 of the 4,314 are isolates the catalog places correctly and so belong to the engine-only row above. The second failure is isolates that both arms call resistant and both assign to the wrong tier, which is a limit of the tiering rather than of the catalog.

A paired McNemar test over the resistant-truth isolates gives $\chi^2 = 77.1$ and $p \approx 1.6 \times 10^{-18}$, with 98 falling on the engine side. That number overstates the difference. The engine reads EFFECTS while the catalog arm reads PREDICTIONS, and 97 isolates carry a catalog call above below-MDR with no graded mutation reaching the engine at all. Eighty of the 97 are placed correctly by the catalog, which accounts for most of the 98. The other 17 are isolates that both arms misclassify. Restricting the comparison to isolates the engine actually received yields 18 against 7, with $\chi^2 = 4.0$ and $p = 0.046$. The difference is real and far smaller than the pooled figure suggests. The test conditions on resistant truth, so it reads 105 of the 124 total disagreements. The gap of 91 isolates between 98 and 7 is the same gap that separates the two overall accuracy figures, because the disagreements the test sets aside fall on isolates where both arms score wrong. Every one of the 18 is an isolate the engine placed one tier too high, 13 raising MDR to pre-XDR and 5 raising pre-XDR to XDR. All 18 carry a kanamycin call graded in EFFECTS and 17 carry amikacin as well. So these are the same table gap surfacing as a one-tier bump rather than a total miss. The tier definition is not the cause, since the label and the classification rules read the same drug-class sets from config.

### What bounds precision

The tier label reads an untested drug as susceptible. Pre-XDR and XDR are separated from MDR by fluoroquinolone and injectable results, and 38,674 isolates carry a result for both, which is 59.0% of the cohort. An isolate never tested on either class is capped at MDR by its label however resistant its genotype is, so a correct genotypic call above MDR is scored a false positive on an isolate that was never measured.

Scoring the same tiers on the isolates that were measured separates the two readings.

| Tier | Precision, full cohort | Precision, tested | Resistant, full | Resistant, tested |
| --- | ---: | ---: | ---: | ---: |
| MDR | 0.758 | 0.852 | 10,335 | 8,491 |
| pre-XDR | 0.483 | 0.616 | 4,725 | 4,605 |
| XDR | 0.482 | 0.629 | 2,463 | 2,463 |

The macro-F1 increases from 0.662 to 0.708 for the engine and from 0.665 to 0.712 for the catalog, and balanced accuracy from 67.4% to 68.9% and from 67.9% to 69.4%. Overall accuracy falls from 83.4% to 80.8%, because the restriction removes mostly below-MDR isolates and leaves a harder mix behind.

The increase results mainly from removing false positives, not hard positives. An isolate cannot be labeled XDR without results for both classes, so the restriction cannot eliminate an XDR isolate. Its count stays at 2,463 and sensitivity remains at 54.7% by design, with only the 14.7 percentage point increase in precision conveying new information. Pre-XDR provides the evidence, as it retains 4,605 of 4,725 isolates, loses 53 correct calls, and gains 13.3 points of precision. Removing the 26,914 untested isolates thus reduces false positives in the top tiers with minimal loss of correct calls.

The per-drug arm did not include this because it scores only drugs with results for the relevant isolates. The tier arm now records both readings, with the restricted one labeled under `second_line_covered`.

### What bounds sensitivity

Sensitivity should be read against what the input permits rather than against a perfect 100%. The engine recognizes only mutations the catalog grades as resistance-conferring, so an isolate carrying none of them arrives with an empty genotype and falls below the MDR threshold, whatever its phenotype. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded mutation, which puts the upper limit on sensitivity at 79.7%. The limit differs by tier, at 81.7% for MDR, 81.1% for pre-XDR, and 68.4% for XDR. Against those limits, the engine finds 75.8% of what is available at MDR, 75.8% at pre-XDR, and 80.0% at XDR. The raw figures fall as the tiers grow more resistant while the reachable share does not, though the four-point spread rests on 2,463 isolates and carries no interval, so it is a direction rather than a result.

### Per-drug resistance calls

Each drug is evaluated individually against the measured phenotype, using the WHO catalog as the reference. Calculations are based on isolates with results for each drug, ranging from 19,948 for capreomycin to 59,869 for rifampin, so each row shows its own denominator alongside the resistant count, instead of the 65,588 total in the header. The macro-F1 score is 0.588 for the engine and 0.611 for the catalog across the 15 drugs. Since both reference arms assess the same isolates per drug, the comparison uses a paired McNemar test rather than separate confidence intervals. Twelve out of 15 drugs show no discordant isolates, including both fluoroquinolones, since the catalog already grades levofloxacin and moxifloxacin from the same call on the DNA gyrase subunit A gene (gyrA), so the class expansion adds nothing there. These twelve drugs align on an isolate-by-isolate level rather than just overall.

The main distinction lies in the three injectables. The engine considers amikacin, kanamycin, and capreomycin as a single class, so a mutation affecting any one of them results in resistance to all three. In practice, cross-resistance among these injectables is only partial, making the trade-off generally disadvantageous. The precision on amikacin falls from 0.834 to 0.518 and on capreomycin from 0.776 to 0.439, against sensitivity gains of 2.2 and 4.2 percentage points. F1 falls on all three. The paired test puts the discordance at 1,524 isolates for amikacin with $\chi^2 = 1280.6$, 1,343 for capreomycin with $\chi^2 = 1061.5$, and 149 for kanamycin with $\chi^2 = 116.9$, each at $p < 10^{-26}$. Read the ratios rather than the p-values. Because the engine calls a strict superset of the catalog, the test asks only whether the added calls are more often wrong than right, and at these cohort sizes the p-value reflects the denominator. The three tests are also driven by the same mutations in the same isolates, falling in the 16S ribosomal RNA gene, written rrs, and the enhanced intracellular survival gene, written eis, so they are one finding measured three ways rather than three confirmations.

The behavior is documented as a quantifiable characteristic of the heuristic, rather than an implicit assumption. The score assigned to the arm does not directly reflect the rule's intent, as the expansion only adds entries to the exclusion list. Its goal is to prevent a likely-failing drug from being included in a regimen, rather than to predict a phenotype. The scoring runs through `python Evaluation/metrics.py`, which writes `Evaluation/per_drug_results.json`.

### Expert system

The natural-language layer is assessed through execution match, pairing each question with a gold query. A generated query is deemed correct if it produces the same result set. On claude-sonnet-4-6, the latest run answered ten out of eleven questions, achieving a correctness rate of 90.9%, with a Wilson interval of 62.3% to 98.4%. It is better to interpret the interval instead of the single-point estimate, as eleven questions are insufficient to draw conclusions about translation quality in either direction.

Only nine questions are scored this way. The remaining two require information that the graph cannot provide and are scored on refusal, which passes when the generated text either states the question is unanswerable or fails the read-only guard. An explicit refusal and an attempted write count alike, and the result file notes the generated query only when an item fails, so the two passing refusals leave no trace for review.

Generation runs at temperature zero, which does not make the arm reproducible. Four runs returned ten of eleven three times and eleven of eleven once. Every failing run failed the same lookup, where the query collects a relationship property without binding the relationship. Memgraph rejects it as an unbound variable. One unstable question describes the arm better than a drifting score, and an invalid query is a better failure than a plausible wrong answer.

The test suite verifies the deterministic components of the layer, including the read-only guard, routing, and normalization (which removes order clauses the database cannot satisfy after an aggregate but retains them when a LIMIT depends on it).

### Case-based reasoning, the experimental component

Regimen accuracy is 77.2%, with a bootstrap interval from 74.5% to 79.8%. It is slightly below the majority-class baseline of 79.3% and near the ceiling of 79.8%. The ceiling marginalizes the year, since the generator selects the regimen from profile and year while the year is independent of every feature retrieval matches on. Outcome accuracy stands at 74.8%, marginally above its baseline of 74.6%.

The engine refrains from assigning a regimen when none of the retrieved neighbors provide a suitable option supported by at least two neighbors. This occurred six times in a thousand, leading to a coverage rate of 99.4%. An abstention indicates no regimen is assigned and is viewed as an error, making these figures conservative estimates. When selecting the most common regimen among the retrieved neighbors, regardless of outcome, the accuracy is 76.8%, with a confidence interval from 74.2% to 79.5%. This suggests that outcome weighting has minimal influence and that the main limitation arises from the retrieval neighborhood rather than the scoring method.

The overall summary hides the main challenge. Susceptible and MonoResistant cases each correspond to a single regimen and constitute 62% of the cohort. Consequently, a predictor identifying their profile nearly always succeeds, with 618 out of 620 cases correctly classified. For the remaining 38% with multiple options, the baseline accuracy is 45.5%, the retrieval score is 40.5%, and the upper limit reaches 46.9%. The model performs well on straightforward cases but faces difficulties with more complex ones. The baseline falls within the bootstrap interval for the entire cohort, and a smaller subset of 380 cases shows a slight gap that is a point estimate rather than a statistically significant difference.

| Profile | Regimen accuracy | n |
| --- | ---: | ---: |
| Susceptible | 100.0% | 500 |
| MonoResistant | 98.3% | 120 |
| PolyResistant | 45.0% | 60 |
| MDR | 35.0% | 180 |
| PreXDR | 56.2% | 80 |
| XDR | 31.7% | 60 |

Retrieval crosses resistance profiles on 15.4% of neighbors, because the profile carries a weight of 0.32 against an ordinal similarity where an adjacent tier still scores 0.8. A neighbor one tier away loses 0.064, while a case with the same profile but a different treatment history, HIV status, and region loses 0.41, since a region mismatch keeps half its weight instead of losing all of it. Boundary crossings are therefore common. Compared with the per-profile baseline on identical folds, retrieval wins 59 cases and loses 80, which is 139 discordant cases at $p = 0.090$.

The generator draws the regimen from the profile and the year. The year never enters the query, so retrieval cannot match on it, although the ranking does discount older cases and that discount is worth 0.4 points of regimen accuracy. What retrieval can reach on this cohort is therefore bounded by the per-profile lookup, whatever weights it carries. The limit belongs to the synthetic cases rather than to the weighting, so a weaker score here is not on its own an argument for retuning.

### Calibration

The predicted probability represents the proportion of the ten retrieved neighbors that succeeded, smoothed by adding one success and one failure. This smoothing is important. Without it, a raw share would be exactly one if all ten neighbors succeeded, which occurred in 77 of the 1,000 cases. In those cases the logit that both scaling methods fit is unbounded, and these few scores could otherwise drive the fit on their own.

The expected calibration error for the smoothed probability is 0.0942, and the Brier score is 0.1951. A constant prediction based on the base rate achieves a score of 0.1895, outperforming it. This constant is fitted to the outcomes it predicts, making it the most robust baseline of its kind and serving as a conservative point of comparison.

Temperature scaling averages at 1.113, with fold values from 1.05 to 1.238, but it increases the calibration error from 0.0942 to 0.103, indicating an issue with direction rather than spread. The middle bins are under-confident, with accuracy above their stated probabilities, while the top two bins are over-confident. A single parameter dividing the logit shifts every score the same way and cannot answer both. Platt scaling introduces an intercept, adjusts slopes from 0.252 to 0.517, and reduces the calibration error to 0.018, with a Brier score of 0.1891. Both methods are trained on four folds and applied to the held-out fold, ensuring no data leakage.

This improvement does not prove the outcome layer is effective. Slopes well below one reduce the score toward the base rate, and most of the apparent gain results from this shrinkage rather than actual added signal. Since neither scaling alters the ranking, the area under the curve of 0.562 reflects the true signal, which is just above chance. The Platt Brier score surpasses the constant by 0.0004, and the outcome accuracy of 74.8%, with a confidence interval from 72.1% to 77.4%, includes the 74.6% baseline. The predicted probability offers little insight into treatment success, and presenting the fitted slope alongside the calibration error keeps this point clear.


## Data

The platform is built on the [WHO mutation catalog, second edition](https://www.who.int/publications/i/item/9789240082410), supplied as the file WHO-UCN-TB-2023.7-eng.xlsx. Real-world validation uses [CRyPTIC release 3.4.0](https://doi.org/10.5281/zenodo.15680920), which pairs whole-genome variants graded against that catalog with laboratory resistance phenotypes. This release includes 53,897 samples with both sequencing and phenotype data, plus 11,945 samples with only phenotype data. The validation set consists of 65,588 samples with measured drug results, scored in their entirety rather than on a held-out subset.

The release carries two phenotype sources. Drug susceptibility testing, abbreviated DST, is the reference method run in clinical laboratories. The UKMYC plate is a broth microdilution assay read across a fixed drug panel. The two agree on 94.8% of the 21,568 isolates measured by both, which sets a label-noise floor beneath every accuracy figure reported above. They disagree on 1,117 isolates, and on all 1,117 the UKMYC profile is the less severe of the two, never the more severe. The exploratory analysis separates how much of that follows from the narrower UKMYC panel.

The catalog does not classify every call as resistant or susceptible. Some results are uncertain or failed, and both arms count these as not resistant, aligning with the approach where isolates without genotypic calls are considered below-MDR. The exposure is worth naming, since 31,517 isolates have at least one uncertain call, and 2,927 have one on rifampin or isoniazid, the two key drugs for defining MDR. If uncertain calls were categorized as resistant, all the figures in both arms would change.

The synthetic patient cases are transparent and deterministic under a fixed seed.

The actual datasets are not included in this repository due to their large size. To reproduce the results, download them into a `Datasets/` folder located at the project root. The catalog file WHO-UCN-TB-2023.7-eng.xlsx is from WHO. The CRyPTIC tables, including EFFECTS.parquet, PREDICTIONS.parquet, DST_MEASUREMENTS.parquet, UKMYC_PHENOTYPES.parquet, and the file DRUG_CODES.csv, originate from CRyPTIC release 3.4.0 on Zenodo. The synthetic patient cases are generated through code and do not require downloading. Accessing the CRyPTIC parquet tables requires the pyarrow engine, which is installed via `requirements.txt`.

The release also ships `DATA_SCHEMA.pdf`, which documents the full set of tables, and `MUTATIONS.parquet`, which this project retains but does not read. The exploratory analysis explains why the rule engine sources its genotypes from EFFECTS instead. A seventh file, `Datasets/cryptic_features.parquet`, is built on first use and cached. It rebuilds itself whenever a source table, `feature_engineering.py`, or `config.py` is newer than the cache, so replacing a table is enough.

## Evaluation

All scoring runs through a single entry point.

```bash
python Evaluation/validation.py
```

This clears and rebuilds the knowledge graph, then runs the expert-system and case-based reasoning validation against it. If the database or API is unavailable, it omits that part and prints a note. The next step then performs the database-free CRyPTIC classification validation. Because it clears the graph first, run it before the demo rather than after, since it discards any case base the app has loaded.

Results are written to `Evaluation/validation_results.json`, replacing the committed reference run. A skipped arm keeps its previous result rather than being erased, and an `arms_this_run` field records which sections were actually recomputed.

The per-drug resistance scoring operates independently and writes `Evaluation/per_drug_results.json`.

```bash
python Evaluation/metrics.py
```

The shared scoring functions, such as sensitivity, specificity, precision, F1, balanced accuracy, macro-F1, the McNemar test, and the Brier score, are located in `Evaluation/metrics.py`. This ensures that the tier scoring in `validation.py` and the per-drug scoring measure the same metrics consistently. However, the reference arm differs between the two; the tier arm fetches the catalog profile from `PREDICTIONS.parquet`, while the per-drug arm retrieves it from `EFFECTS.parquet` so that both columns come from one table. It is advisable to compare within the same arm rather than across different files.

A standalone deterministic test suite with 124 tests verifies rule-engine classification, calibration calculations, the read-only query guard and routing, generator determinism, seed-graph integrity, and consistency between the two inference modes. It operates without needing a database, API, or datasets and runs from the project root.

```bash
pytest tests/test_core.py
```

The same suite executes in continuous integration for every push to `main` and each pull request, covering Python versions 3.10, 3.11, and 3.12.

### Reproducing

The project builds up in three levels, each adding to the previous one. The test suite only requires running `pip install -r requirements-dev.txt` and includes 124 tests that verify the rule engine, calibration calculations, query guard, and seed-graph integrity. Incorporating Docker and an Anthropic API key activates the demo and the expert-system component on the seed graph, all without downloading any datasets. Including the `Datasets/` folder enables access to the CRyPTIC data and per-drug metrics mentioned earlier.

The CRyPTIC, per-drug, and case-based arms are deterministic, seeded at 42. Repeated runs consistently reproduce every digit in the two result files, with only the last decimal place of one p-value varying due to differences in floating-point summation order. The expert-system arm calls a live model, and four runs yielded ten out of eleven three times and eleven out of eleven once; this is the only figure expected to change and is reported alongside its generating model. The full procedure is detailed in [DEPLOYME.md](DEPLOYME.md).

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

The remaining directions concern how the system is evaluated and how it handles rare classes. The regimen layer is scored by exact match to the labeled regimen, which penalizes it for optimizing treatment outcome and guideline adherence instead, so part of the measured shortfall is a metric mismatch rather than a modeling error. Scoring against outcome directly is the simplest of these changes, since it needs no additional data and fixes the mismatch at its source. Confidence-gated deferral would extend the abstention the engine already performs when its neighbors carry nothing applicable. A sparse neighborhood would hand the case to the rule engine, and the arm would report coverage beside accuracy, which turns rare-class scarcity into calibrated behavior rather than silent error.

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
