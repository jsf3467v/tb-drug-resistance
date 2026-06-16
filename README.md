[![CI](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml/badge.svg)](https://github.com/jsf3467v/tb-drug-resistance/actions/workflows/ci.yml)

# TB Drug-Resistance Decision Support

A neuro-symbolic clinical reasoning system for *Mycobacterium tuberculosis*. The system combines a WHO-grounded knowledge graph, a symbolic rule engine, case-based reasoning over patient cases, and a natural-language query layer. Its symbolic core has been validated against real-world resistance measurements from the CRyPTIC consortium.

## Overview

Drug-resistant tuberculosis demands reasoning that is both auditable and grounded in current evidence. This system pairs an explicit symbolic layer, where every resistance classification can be traced to a WHO catalog rule, with a case-based layer that draws on prior patient experience where guidelines are silent. A natural-language interface translates plain-English questions into graph queries under a read-only guard, and a Streamlit front end exposes the full reasoning trace.

The project is presented as a portfolio piece rather than a deployable clinical tool. Its thesis is honest, rigorously evaluated engineering, with the synthetic patient layer and the genotype-phenotype prediction ceiling treated as measured limits rather than hidden ones.

## Architecture

The design separates a durable, evidence-grounded platform from a swappable patient layer.

- Knowledge graph. A Memgraph store holding 30,699 mutation nodes derived from 48,152 rows of the WHO mutation catalog.
- Rule engine. A forward and backward chaining symbolic engine that classifies isolates as MDR, pre-XDR, or XDR, applies whole-class cross-resistance, and selects between the BPaL and BPaLM regimens.
- Case-based reasoning. Retrieval over 1,000 synthetic patient cases, used where the rules alone do not determine a regimen.
- Natural-language interface. An LLM layer that generates Cypher from plain English behind a read-only write guard.

## Results

### Real-world validation of the symbolic core

The rule engine was validated on 13,118 CRyPTIC isolates carrying measured drug-susceptibility phenotypes. On those isolates it reproduces the WHO genotypic catalog for 99.9% of predictions, confirming that the symbolic reasoning is faithful to the gold-standard source it encodes. Measured against phenotype, the engine reaches 84.0% overall accuracy, statistically identical to the WHO catalog itself at 84.0%.

Per-tier accuracy runs from 92.0% on the below-MDR class to 63.0% on MDR, 63.0% on pre-XDR, and 52.9% on XDR.

The error analysis is the central finding. Of 3,505 resistant isolates, 2,158 are classified correctly by both the engine and the catalog, 1,335 are misclassified by both, and only 12 are unique to the engine. The 1,335 shared errors are genotype-phenotype discordance, namely resistance that is present on phenotypic testing but carries no genotypic marker the WHO catalog recognizes. No genotype-based method can recover these, so the residual error is a measured biological ceiling rather than a defect in the implementation. The 12 engine-only cases reduce to 11 data-coverage artifacts and one instance of a documented, scoped definitional choice.

### Expert system

The natural-language query layer reached 100% on the twelve scored queries, with one ambiguous query reported as measured rather than pass-or-fail gated.

### Case-based reasoning, the experimental component

Regimen accuracy is 67.4% against an 81.0% majority-class baseline, and outcome accuracy is 74.5% against a 73.8% baseline. The regimen shortfall decomposes into roughly 7.5 points of objective mismatch and 6 points of retrieval starvation in the rare resistant classes, where neighbor-based retrieval is data-poor by construction. This result is reported as the measured behavior of the experimental layer, not as the headline.

### Calibration

Expected calibration error is 0.075 on the raw predicted success probability. Post-hoc temperature scaling was tested and rejected on evidence, since it raised the error to 0.177, a classic mismatch between negative log-likelihood and calibration error.

## Data

The platform is grounded in the WHO mutation catalog (catalog name WHO-UCN-GTB-PCI-2023.5). Real-world validation draws on the CRyPTIC consortium release, which provides whole-genome variants graded against the catalog together with measured drug-susceptibility phenotypes. Of 65,588 isolates with a measured phenotype, 13,118 form the held-out validation set. The two phenotype assays in the release, DST and UKMYC, agree on 95.6% of jointly measured isolates, which sets a label-noise floor beneath the accuracy figures above. The synthetic patient cases are transparent and deterministic given a fixed seed.

## Evaluation

All scoring runs through a single entry point.

```bash
python validation.py
```

This runs the expert-system and case-based-reasoning validation against the live graph, skipping that arm with a printed note if the database or API is unavailable, and then runs the database-free CRyPTIC classification validation. Results are written to `validation_results.json`. A separate deterministic test suite of 32 tests locks in rule-engine classification, calibration math, generator determinism, and seed-graph integrity, with no database or API required.

## Limitations

- The patient layer is synthetic. No real treatment-outcome dataset with the required structure was available, so case retrieval is demonstrated on generated cases.
- CRyPTIC provides genotype and phenotype but not treatment outcomes, so it validates classification only and cannot validate the regimen and outcome layer.
- The rule engine implements a scoped pre-2021 XDR definition, documented as a deliberate choice rather than the current Group A based standard.
- Genotype-based resistance prediction is bounded by discordance, so roughly 16% of measured resistance is not recoverable from the catalog by any rule-based method.

## Future work

- A learned model trained on the full genome-wide variant table and minimum-inhibitory-concentration magnitudes, to probe how much of the genotype-phenotype discordance ceiling can be recovered beyond the curated catalog.
- Outcome validation of the case-based layer against the TB Portals dataset, which carries real treatment outcomes.

## Setup

Run a local Memgraph instance, then start the application.

```bash
pip install -r requirements.txt
python validation.py
streamlit run app.py
```

Graph credentials are read from the environment, defaulting to a local no-auth instance.

## References

### Case-Based Reasoning

1. Kolodner, J. L. (1992). An Introduction to Case-Based Reasoning. *Artificial Intelligence Review*, 6(1), 3–34.
2. Main, J., Dillon, T. S., & Shiu, S. C. K. (2001). A Tutorial on Case-Based Reasoning. *Soft Computing in Case Based Reasoning*, 1–28.
3. Goel, A. K., & Díaz-Agudo, B. (2017). What's Hot in Case-Based Reasoning. *Proceedings of AAAI-17*.
4. Das, R., Godbole, A., Dhuliawala, S., Zaheer, M., & McCallum, A. (2020). A Simple Approach to Case-Based Reasoning in Knowledge Bases. *Automated Knowledge Base Construction (AKBC)*.

### WHO Guidelines

5. World Health Organization. (2023). *Catalogue of mutations in Mycobacterium tuberculosis complex and their association with drug resistance* (2nd ed.).
6. World Health Organization. (2025). *WHO consolidated guidelines on tuberculosis: Module 4: Treatment and care*.
7. Walker, T. M., et al. (2022). The 2021 WHO catalogue of Mycobacterium tuberculosis complex mutations. *The Lancet Microbe*, 3(4), e265–e273.

### Calibration

8. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern
   Neural Networks. *Proceedings of the 34th International Conference on Machine Learning
   (ICML)*, PMLR 70, 1321–1330.

### Treatment Evidence

9. Nyang'wa, B.-T., et al. (2022). A 24-Week, All-Oral Regimen for Rifampin-Resistant
   Tuberculosis. *New England Journal of Medicine*, 387(25), 2331–2343. (TB-PRACTECAL; BPaLM)
10. Conradie, F., et al. (2020). Treatment of Highly Drug-Resistant Pulmonary Tuberculosis.
    *New England Journal of Medicine*, 382(10), 893–902. (Nix-TB; BPaL)

### Datasets

11. The CRyPTIC Consortium. (2022). A data compendium associating the genomes of 12,289
    *Mycobacterium tuberculosis* isolates with quantitative resistance phenotypes to 13
    antibiotics. *PLOS Biology*, 20(8), e3001721.
12. Rosenthal, A., et al. (2017). The TB Portals: an Open-Access, Web-Based Platform for
    Global Drug-Resistant-Tuberculosis Data Sharing and Analysis. *Journal of Clinical
    Microbiology*, 55(11). doi:10.1128/JCM.01013-17.

## License

Released under the MIT License.
