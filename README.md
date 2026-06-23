# TB Drug-Resistance Decision Support System

![tests](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/tests.yml/badge.svg)

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

- Rule engine. A forward and backward chaining symbolic engine that classifies isolates as MDR, pre-XDR, or XDR, applies whole-class cross-resistance, and selects between the BPaL and BPaLM regimens.

- Case-based reasoning. Retrieval of over 1,000 synthetic patient cases, used where the rules alone do not determine a regimen.

- Natural-language interface. An LLM layer that generates Cypher from plain English behind a read-only write guard. The query runs in a read transaction that Memgraph rejects on any write, so the database itself is the barrier, and a keyword pre-filter blocks an obvious write before the query runs.

The figure below traces one strain through the graph, from its mutations to the genes and drugs they affect and on to its resistance profile, which is the same path the rule engine walks to reach a classification.

![Strain TB011 traced through the knowledge graph, from its four mutations to the genes and drugs they affect and on to its XDR resistance profile](assets/knowledge_graph.png)

## Results

### Real-world validation of the symbolic core

The rule engine was validated on all 65,588 CRyPTIC isolates with a measured drug-susceptibility phenotype. It reproduces the WHO genotypic catalog on 99.8% of isolates, confirming that the engine faithfully reimplements the catalog tiering it encodes rather than adding hidden logic. Measured against phenotype, the engine achieves 83.4% overall accuracy, while the WHO catalog achieves 83.5%. The two are close but not identical. A paired McNemar test on the 105 isolates where they disagree yields $\chi^2 = 77.1$ and $p \approx 1.6 \times 10^{-18}$, and 98 of those 105 disagreements are engine-side, all accounted for below. The gap is small, real, and explained.

Relying solely on accuracy favors an imbalanced dataset, where below-MDR cases make up 73.3% of the isolates, meaning a model that always predicts below-MDR would already achieve that baseline 73.3%. Balanced accuracy, calculated as the average of sensitivities across different tiers, is 67.4% for the engine compared to 67.9% for the catalog. The macro-F1 scores are 0.662 versus 0.666. Sensitivity per tier ranges from 91.6% on below-MDR, down to 61.9% on MDR, 61.5% on pre-XDR, and 54.7% on XDR. Specificity remains above 94% across all resistant tiers.

![Grouped bar chart of per-tier sensitivity against measured phenotype, rule engine versus WHO catalog, across below-MDR, MDR, pre-XDR, and XDR. The two systems are within a point of each other on every tier, and sensitivity declines from 92% on below-MDR to 55% on XDR as the tiers grow rarer.](assets/cryptic_tier_sensitivity.png)

The bars also frame the real result. The engine is not the bottleneck. It reaches essentially the same per-tier sensitivity as the catalog it encodes, so the headroom that remains lives in the catalog and the data, not in the implementation. The error analysis below quantifies exactly that.

The error analysis is the main finding. Of the 17,523 resistant isolates, the engine and catalog land the same way on all but 105 of them.

| Of 17,523 resistant isolates | Isolates | Share | What it is |
| --- | ---: | ---: | --- |
| Both correct | 10,646 | 60.8% | engine and catalog both right |
| Both wrong | 6,772 | 38.6% | genotype-phenotype discordance, resistance on phenotype with no genotypic marker the catalog recognizes |
| Engine only wrong | 98 | 0.6% | 80 data-coverage gaps plus 18 documented definitional cases |
| Catalog only wrong | 7 | 0.04% | resistance the catalog misses but the engine catches |

The 6,772 shared errors represent a biological upper limit, not a flaw in the design. No genotype-based method can detect resistance that lacks a recognized genotypic marker. The 98 engine-only cases also do not indicate a logical error. Eighty cases are coverage gaps, where the catalog labels an isolate as resistant, but no graded mutation is detected by the engine, resulting in a resistance below-MDR classification. The remaining eighteen are due to a pre-2021 definitional choice, where injectable-based escalation assigns an isolate to a higher resistance tier than the 2021 catalog. Thirteen of these elevate MDR to pre-XDR on injectable resistance; five raise pre-XDR to XDR due to fluoroquinolone-plus-injectable resistance. Overall, the actionable error accounts for a mere 0.6%.

### Per-drug resistance calls

The tier validation groups categorized drugs into four resistance groups. Each drug is individually assessed for resistance or susceptibility based on the DST phenotype, using the WHO catalog as a reference. The engine and the catalog agree on 12 of 15 drugs, including both fluoroquinolones, since the catalog groups levofloxacin and moxifloxacin under a single gyrA call. The only discrepancies are with the three injectable drugs. In these cases, the engine assumes whole-class cross-resistance any mutation in this class indicates resistance to amikacin, kanamycin, and capreomycin collectively. This approach slightly increases sensitivity and decreases specificity, as injectables only partially exhibit cross-resistance in practice.

For example, amikacin's precision drops from 0.834 in the catalog to 0.518 in the engine, and capreomycin's from 0.776 to 0.439. This tradeoff is recorded as a property of the heuristic rather than an implicit assumption. The scoring runs through `python Evaluation/metrics.py`, which generates `per_drug_results.json`.

### Expert system

The natural-language layer's performance is assessed based on execution match, where each question is matched with a gold-standard query. A generated query is considered correct if it returns the same entities. Since the layer depends on live model generation, its score varies with the model used, rather than being a fixed measure. For example, on claude-sonnet-4-6, it correctly answers ten out of eleven queries and maintains that accuracy across different runs, as it generates responses at temperature zero. The only failure occurs during a lookup, where the generator returns more information than requested, specifically extracting a relationship property without binding the relationship, leading the database to reject it as an unbound variable. The deterministic components of the layer remain stable. The read-only write guard and query routing are controlled by the test suite, and normalization removes an unsupported order clause after an aggregate. The remaining issue is a generation error, not a flaw in the layer itself.

### Case-based reasoning, the experimental component

The regimen accuracy stands at 67.4%, compared to an 81.0% baseline for the majority class, while outcome accuracy reaches 74.5%, slightly above the 73.8% baseline. The shortfall in regimen performance mainly results from approximately 7.5 points of objective mismatch and about 6 points of retrieval starvation in the rare resistant classes, where neighbor-based retrieval inherently has limited data. This reflects the observed behavior of the experimental layer, not the overall result.

Regimen accuracy also changes significantly depending on the resistance profile, revealing the effects of retrieval starvation.

| Profile | Regimen accuracy | n |
| --- | ---: | ---: |
| Susceptible | 99.0% | 500 |
| MonoResistant | 55.0% | 120 |
| PolyResistant | 18.3% | 60 |
| MDR | 37.8% | 180 |
| PreXDR | 26.3% | 80 |
| XDR | 21.7% | 60 |

The synthetic cohort was created intentionally to address a significant data gap. A case-based regimen recommender requires data that connects genotype, patient profile, treatment regimen, and observed outcome. No publicly available dataset offers this complete chain at the necessary scale. Such treatment-outcome data is rare, scattered across different institutions, and often kept confidential for privacy reasons—an issue common in clinical machine learning. Creating the synthetic cases guarantees that the retrieval process stays transparent and reproducible despite this data gap.

The weaker numbers in the rare resistant classes follow from the same scarcity. XDR and pre-XDR are uncommon by definition, so even a large cohort holds few neighbors for them, and neighbor-based retrieval degrades wherever a class is thin. The shortfall is therefore a measured consequence of too little data per class rather than a defect in the retrieval method, and it mirrors what learned models face on the same rare-disease data.

### Calibration

Expected calibration error is 0.075 on the raw predicted success probability, and the Brier score is 0.192. Post-hoc temperature scaling was tested and rejected on evidence, since it raised the calibration error to 0.177, a classic mismatch between negative log-likelihood and calibration error.

## Data

The platform is based on the WHO mutation catalog, second edition, provided as the data file WHO-UCN-TB-2023.7-eng.xlsx. Real-world validation utilizes data from the CRyPTIC consortium release, which includes whole-genome variants graded against the catalog and associated drug-susceptibility phenotypes. The validation set consists of 65,588 isolates with measured phenotypes, scored in full rather than on a held-out split. The two phenotype assays in the release, DST and UKMYC, agree on 95.6% of the jointly measured isolates, establishing a label-noise floor below the reported accuracy. The synthetic patient cases are transparent and deterministic when using a fixed seed.

The actual datasets are not included in this repository due to their large size. To reproduce the results, download them into a `Datasets/` folder located at the project root. The catalog file WHO-UCN-TB-2023.7-eng.xlsx is from the World Health Organization. The CRyPTIC tables, including EFFECTS.parquet, PREDICTIONS.parquet, DST_MEASUREMENTS.parquet, UKMYC_PHENOTYPES.parquet, and the file DRUG_CODES.csv, originate from the CRyPTIC consortium release on Zenodo. The synthetic patient cases are generated through code and do not require downloading. Accessing the CRyPTIC parquet tables requires the pyarrow engine, which is installed via `requirements.txt`.

## Evaluation

All scoring runs through a single entry point.

```bash
python Evaluation/validation.py
```

This executes the expert-system and case-based reasoning validation on the live graph, omitting that part and printing a note if the database or API is unavailable. It then performs the database-free CRyPTIC classification validation. The results are saved to `validation_results.json`.

The per-drug resistance scoring operates independently and saves its output in `per_drug_results.json`.

```bash
python Evaluation/metrics.py
```

The shared scoring functions, including sensitivity, specificity, precision, balanced accuracy, macro-F1, the McNemar test, and the Brier score, live in `Evaluation/metrics.py`, so the tier scoring in `validation.py` and the per-drug scoring use one definition and remain comparable.

A separate deterministic test suite of 44 tests verifies rule-engine classification, calibration math, the read-only query guard and routing, generator determinism, and seed-graph integrity. It requires no database, API, or datasets and runs from the project root.

```bash
pytest tests/test_core.py
```

The same suite runs in continuous integration on every push, across Python 3.10, 3.11, and 3.12.

## Limitations

- The patient layer is synthetic because no open dataset links genotype, regimen, and outcome at the scale a case-based recommender needs. This data scarcity is a well-known challenge in healthcare machine learning, and it is the direct reason the rare resistant classes evaluate poorly.

- The case-based similarity weights are domain-informed priors set by hand, not values learned from data, and tuning them is future work. The region and outcome tables in the case generator follow the same pattern, since they carry real structure from the WHO regions while their magnitudes stay synthetic rather than transcribed from any WHO release.

- The regimen layer is scored by exact match to the labeled regimen, which penalizes it for optimizing treatment outcome instead, so part of the measured shortfall is a metric mismatch rather than a modeling error.

- CRyPTIC provides genotype and phenotype but not treatment outcomes, so it validates classification only and cannot validate the regimen and outcome layer.

- The rule engine implements a scoped pre-2021 XDR definition, documented as a deliberate choice rather than the current Group A based standard.

- The rule engine does not model ethionamide, so the inhA cross-resistance that links isoniazid and ethionamide is out of scope. This is a named boundary rather than an oversight.

- Genotype-based resistance prediction is bounded by discordance. Of the 17,523 resistant isolates, 6,772 carry phenotypic resistance with no genotypic marker the catalog recognizes, so that share is not recoverable from the catalog by any rule-based method.

## Future work

Several directions would extend the work, and they fall into two groups, the data the system can reach and the way its layers are scored.

The most significant data gap is the synthetic case base. Validating the case-based layer with the TB Portals dataset, which includes actual treatment outcomes, would replace the cohort where real signals are most needed. Using real data enhances the credibility of the results, though it doesn't ensure higher accuracy, as resistant cases remain rare even in large real datasets. A trained model could push the results toward the other ceiling. Training such a model with the full genome-wide variant table and the minimum inhibitory concentration magnitudes would explore how much of the genotype-phenotype mismatch can be explained beyond the curated catalog.

The remaining instructions can enhance how the system is evaluated and how it manages rare classes. The regimen layer now receives a score based on an exact match to the labeled regimen, which penalizes it for emphasizing treatment outcomes and guideline adherence instead. This shift to the actual goal is the simplest change, as it requires no additional data and directly fixes the metric mismatch. Moreover, confidence-gated deferral enables a sparse retrieval neighborhood to defer to the rule engine and report coverage along with accuracy, turning rare-class scarcity into well-calibrated behavior. Lastly, the injectable rule's all-class form groups amikacin, kanamycin, and capreomycin, but the per-drug table indicates over-calls for amikacin and capreomycin against measured DST. Connecting cross-resistance to the gene, with rrs causing broad resistance and favoring kanamycin, could restore lost precision without changing tier results.



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
