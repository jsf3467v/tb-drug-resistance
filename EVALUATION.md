# Evaluation detail

Full scoring for every arm. The [README](README.md) carries the headline figures and the
design rationale. Numbers here come from `Evaluation/validation_results.json` and
`Evaluation/per_drug_results.json`, both written by the commands in
[DEPLOYME.md](DEPLOYME.md) section 6.

## Contents

- [Real-world validation of the rule engine](#real-world-validation-of-the-rule-engine)
- [What bounds precision](#what-bounds-precision)
- [What bounds sensitivity](#what-bounds-sensitivity)
- [Per-drug resistance calls](#per-drug-resistance-calls)
- [Query translation](#query-translation)
- [Case-based reasoning, the experimental component](#case-based-reasoning-the-experimental-component)
- [Calibration](#calibration)

## Real-world validation of the rule engine

The rule engine was validated on all 65,588 CRyPTIC isolates carrying a measured drug-susceptibility phenotype. Both arms read the EFFECTS and PREDICTIONS tables directly. The knowledge graph holds the same catalog in a second representation but appears in no figure here, since it serves the seed strains and the natural-language interface instead.

The engine reproduces the WHO genotypic catalog on 99.8 percent of isolates. That figure is closer to definitional than to independent evidence, since both arms grade the same variants against the same catalog and apply the same tier definitions. The remaining 0.2 percent arises because one method reads EFFECTS and the other PREDICTIONS. Both tables carry precomputed catalog verdicts and differ mainly in detail, since EFFECTS supplies a per-variant grade that the engine aggregates into a drug-level call while PREDICTIONS already includes that aggregation. The engine's contribution here is therefore the aggregation alone, not the grading or the tier definitions. Whole-class cross-resistance writes only into the exclusion list and never into the classifier's facts, so it appears only in the per-drug analysis.

Measured against phenotype, the engine reaches 83.4 percent overall accuracy and the catalog 83.5 percent. Accuracy alone misleads on an imbalanced set, since below-MDR holds 73.3 percent of isolates and a model that always predicted it would score that much without reasoning. Balanced accuracy, the mean of the per-tier sensitivities, is 67.4 percent for the engine and 67.9 percent for the catalog, with macro-F1 of 0.662 and 0.665. Sensitivity falls from 91.6 percent on below-MDR to 61.9 percent on MDR, 61.5 percent on pre-XDR, and 54.7 percent on XDR, while specificity stays above 94 percent on every resistance tier.

![Grouped bar chart of per-tier sensitivity against measured phenotype, rule engine versus WHO catalog, across below-MDR, MDR, pre-XDR, and XDR. The two systems are within a point of each other on every tier, and sensitivity declines from 92% on below-MDR to 55% on XDR as the tiers grow rarer.](assets/cryptic_tier_sensitivity.png)

The engine reaches almost the same sensitivity per tier as the catalog it encodes, which suggests most of the remaining headroom lies in the catalog and the data rather than in the implementation.

The two arms assign different labels to 124 isolates across the cohort, and 123 of those carry resistant truth. On 105 of the 123 the arms also differ in which one is correct, and that smaller count is what the paired test reads. The other 18 are isolates both arms mislabel under different names.

| Of 17,523 resistant isolates | Isolates | Share | What it is |
| --- | ---: | ---: | --- |
| Both correct | 10,646 | 60.8% | engine and catalog both right |
| Both wrong | 6,772 | 38.6% | both assign a tier and both are wrong |
| Engine only wrong | 98 | 0.6% | 80 coverage gaps and 18 one-tier over-calls |
| Catalog only wrong | 7 | 0.04% | resistance the catalog misses and the engine catches |

The 6,772 cover two failures. The first is resistance the genotype does not show, which no method can reach. The engine places 4,314 isolates with resistant truth below MDR, and 3,562 of those carry no graded mutation at all. That count spans the whole cohort rather than this one row, since 80 of the 4,314 are isolates the catalog places correctly and so belong to the engine-only row above. The second failure is isolates both arms call resistant and both assign to the wrong tier, which is a limit of the tiering rather than of the catalog.

A paired McNemar test over the resistant-truth isolates gives $\chi^2 = 77.1$ and $p \approx 1.6 \times 10^{-18}$, with 98 falling on the engine side. That figure overstates the difference. The engine reads EFFECTS while the catalog arm reads PREDICTIONS, and 97 isolates carry a catalog call above below-MDR with no graded mutation reaching the engine at all. Eighty of the 97 are placed correctly by the catalog, which accounts for most of the 98, and the other 17 are isolates both arms misclassify. Restricting the comparison to isolates the engine actually received gives 18 against 7, with $\chi^2 = 4.0$ and $p = 0.046$. The difference is real and far smaller than the pooled figure suggests. The test conditions on resistant truth and on the arms differing in correctness, so it reads 105 of the 124 total disagreements, and the gap of 91 between 98 and 7 is the same gap that separates the two overall accuracy figures, because the disagreements it sets aside fall where both arms score wrong.

Every one of the 18 is an isolate the engine placed one tier too high, 13 raising MDR to pre-XDR and 5 raising pre-XDR to XDR. All 18 carry a kanamycin call graded in EFFECTS and 17 carry amikacin as well, so these are the same table gap surfacing as a one-tier bump rather than as a total miss. The tier definition is not the cause, since the label and the classification rules read the same drug-class sets from config.

## What bounds precision

The tier label reads an untested drug as susceptible. Pre-XDR and XDR are separated from MDR by fluoroquinolone and injectable results, and 38,674 isolates carry a result for both, which is 59.0 percent of the cohort. An isolate never tested on either class is capped at MDR by its label however resistant its genotype is, so a correct genotypic call above MDR is scored a false positive on an isolate that was never measured.

Scoring the same tiers on the isolates that were measured separates the two readings.

| Tier | Precision, full cohort | Precision, tested | Resistant, full | Resistant, tested |
| --- | ---: | ---: | ---: | ---: |
| MDR | 0.758 | 0.852 | 10,335 | 8,491 |
| pre-XDR | 0.483 | 0.616 | 4,725 | 4,605 |
| XDR | 0.482 | 0.629 | 2,463 | 2,463 |

Macro-F1 rises from 0.662 to 0.708 for the engine and from 0.665 to 0.712 for the catalog, and balanced accuracy from 67.4 to 68.9 percent and from 67.9 to 69.4 percent. Overall accuracy falls from 83.4 to 80.8 percent, because the restriction removes mostly below-MDR isolates and leaves a harder mix behind.

The gain comes from removing false positives rather than hard positives. No isolate can be labeled XDR without results for both classes, so the restriction cannot remove an XDR isolate. Its count stays at 2,463 and its sensitivity at 54.7 percent by design, with only the 14.7 point rise in precision carrying new information. Pre-XDR supplies the evidence, retaining 4,605 of 4,725 isolates, losing 53 correct calls, and gaining 13.3 points of precision. Removing the 26,914 untested isolates therefore cuts false positives in the top tiers at little cost in correct calls. The restricted reading is written under `second_line_covered`.

The per-drug arm carries no equivalent, since it already scores each drug only on the isolates tested for it.

## What bounds sensitivity

Sensitivity should be read against what the input permits rather than against a perfect 100 percent. The engine recognizes only mutations the catalog grades as resistance-conferring, so an isolate carrying none arrives with an empty genotype and falls below the MDR threshold whatever its phenotype. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded mutation, which puts the ceiling at 79.7 percent.

The limit differs by tier, at 81.7 percent for MDR, 81.1 percent for pre-XDR, and 68.4 percent for XDR. Against those limits the engine finds 75.8 percent of what is available at MDR, 75.8 percent at pre-XDR, and 80.0 percent at XDR. The raw figures fall as the tiers grow more resistant while the reachable share does not, though the four-point spread rests on 2,463 isolates and carries no interval, so it is a direction rather than a result.

Most of the gap is provenance rather than resistance. Of the 3,562 unreachable isolates, 3,182 were never sequenced at all and only 380 were sequenced and carry nothing the catalog grades. On the 53,735 sequenced isolates the ceiling rises to 97.3 percent, overall accuracy to 85.6 percent, and balanced accuracy to 78.8 percent. The exploratory analysis traces the tier spread to one pooled collection that sequenced its XDR isolates far less completely than its MDR ones. Both readings are written under `sensitivity_ceiling` and `genotype_covered`.

## Per-drug resistance calls

Each drug is evaluated individually against the measured phenotype with the WHO catalog as reference. Calculations use the isolates carrying a result for that drug, from 19,948 for capreomycin to 59,869 for rifampin, so each row shows its own denominator rather than the 65,588 in the header. Macro-F1 is 0.588 for the engine and 0.611 for the catalog across the 15 drugs. Both arms score the same isolates per drug, so the comparison uses a paired McNemar test rather than separate intervals.

Twelve of the 15 drugs show no discordant isolates, including both fluoroquinolones, since the catalog already grades levofloxacin and moxifloxacin from the same call on the DNA gyrase subunit A gene, written gyrA, so the class expansion adds nothing there. Those twelve agree isolate by isolate rather than merely in aggregate.

The difference lies in the three injectables. The engine treats amikacin, kanamycin, and capreomycin as one class, so a mutation affecting any one confers resistance to all three. Cross-resistance among them is only partial in practice, which makes the trade generally unfavorable. Precision on amikacin falls from 0.834 to 0.518 and on capreomycin from 0.776 to 0.439, against sensitivity gains of 2.2 and 4.2 points. F1 falls on all three. The paired test puts the discordance at 1,524 isolates for amikacin with $\chi^2 = 1280.6$, 1,343 for capreomycin with $\chi^2 = 1061.5$, and 149 for kanamycin with $\chi^2 = 116.9$, each at $p < 10^{-26}$.

Read the ratios rather than the p-values. Because the engine calls a strict superset of the catalog, the test asks only whether the added calls are more often wrong than right, and at these cohort sizes the p-value reflects the denominator. The three tests are also driven by the same mutations in the same isolates, falling in the 16S ribosomal RNA gene, written rrs, and the enhanced intracellular survival gene, written eis, so they are one finding measured three ways rather than three confirmations.

The behavior is documented as a measured property of the heuristic rather than an implicit assumption. The score does not reflect the rule's intent, since the expansion only adds entries to the exclusion list and its goal is to keep a likely-failing drug out of a regimen rather than to predict a phenotype.

## Query translation

The natural-language layer is scored by execution match against a gold query, where a generated query passes if it returns the same result set. On claude-sonnet-4-6, the latest run answered ten of eleven questions for 90.9 percent, with a Wilson interval of 62.3 to 98.4 percent. Read the interval rather than the point estimate, since eleven questions cannot support a conclusion about translation quality in either direction.

Only nine questions are scored by execution match. The remaining two cannot be answered by a read query, one because it requests a write and one because it asks for a field the graph does not hold. Both are scored on refusal, which passes when the generated text either declares the question unanswerable or fails the read-only guard. An explicit refusal and an attempted write count alike, and the result file records the generated query only on failure, so the two passing refusals leave no trace for review.

Generation runs at temperature zero, which does not make the arm reproducible. Six runs returned ten of eleven four times and eleven of eleven twice. Every failing run failed the same lookup, where the query collects a relationship property without binding the relationship and Memgraph rejects it as an unbound variable. One unstable question describes the arm better than a drifting score would, and an invalid query is a better failure than a plausible wrong answer.

The test suite covers the deterministic parts of the layer, including the read-only guard, routing, and normalization, which removes order clauses the database cannot satisfy after an aggregate while keeping them where a limit depends on the order.

## Case-based reasoning, the experimental component

Regimen accuracy is 77.2 percent, with a bootstrap interval from 74.5 to 79.8 percent. It sits slightly below the majority-class baseline of 79.3 percent and 2.6 points under the ceiling of 79.8 percent. That ceiling marginalizes the year, since the generator selects the regimen from profile and year while the year is independent of every feature retrieval matches on. Outcome accuracy is 74.8 percent, marginally above its baseline of 74.6 percent and below its own ceiling of 75.5 percent, computed the same way.

The engine assigns no regimen when none of the retrieved neighbors offers a suitable option backed by at least two of them. That happened six times in a thousand, for a coverage rate of 99.4 percent. An abstention is scored as an error, so the figures are conservative. Taking the most common regimen among the neighbors regardless of outcome scores 76.8 percent, with an interval from 74.2 to 79.5 percent, which suggests outcome weighting has little influence and the binding constraint is the retrieval neighborhood rather than the scoring.

The overall figure hides the real difficulty. Susceptible and MonoResistant cases each map to a single regimen and together hold 62 percent of the cohort, so identifying the profile almost always suffices and 618 of those 620 cases are correct. On the remaining 38 percent, where two or three regimens are available, the baseline reaches 45.5 percent, retrieval 40.5 percent, and the ceiling 46.9 percent. The model performs well on the easy cases and struggles on the rest. The baseline falls inside the bootstrap interval for the whole cohort, and the gap on the 380-case subset is a point estimate rather than a significant difference.

| Profile | Regimen accuracy | n |
| --- | ---: | ---: |
| Susceptible | 100.0% | 500 |
| MonoResistant | 98.3% | 120 |
| PolyResistant | 45.0% | 60 |
| MDR | 35.0% | 180 |
| PreXDR | 56.2% | 80 |
| XDR | 31.7% | 60 |

Retrieval crosses resistance profiles on 15.4 percent of neighbors, because the profile carries a weight of 0.32 against an ordinal similarity where an adjacent tier still scores 0.8. A neighbor one tier away loses 0.064, while a case with the same profile but a different treatment history, HIV status, and region loses 0.41, since a region mismatch keeps half its weight instead of losing all of it. Boundary crossings are therefore common. Against the per-profile baseline on identical folds, retrieval wins 59 cases and loses 80, which is 139 discordant cases at $p = 0.090$.

The generator draws the regimen from the profile and the year. The year never enters the query, so retrieval cannot match on it, although the ranking does discount older cases and that discount is worth 0.4 points of regimen accuracy. What retrieval can reach on this cohort is therefore bounded by the per-profile lookup whatever weights it carries, and the limit belongs to the synthetic cases rather than to the weighting, so a weaker score here is not on its own an argument for retuning. The outcome layer measures the other way, and Calibration sets out why.

## Calibration

The predicted probability is the share of the ten retrieved neighbors that succeeded, smoothed by adding one success and one failure. The smoothing matters. Without it a raw share would be exactly one whenever all ten neighbors succeeded, which happened in 77 of the 1,000 cases, and the logit both scaling methods fit would be unbounded there, letting those few scores drive the fit.

The expected calibration error of the smoothed probability is 0.0942 and the Brier score 0.1951. A constant prediction at the base rate scores 0.1895, beating it. That constant is fitted to the outcomes it predicts, which makes it the strongest baseline of its kind and a conservative comparison.

Temperature scaling averages 1.113, with fold values from 1.05 to 1.238, and raises the calibration error from 0.0942 to 0.103, which indicates a problem of direction rather than of spread. The middle bins are under-confident, with accuracy above their stated probabilities, while the top two are over-confident. One parameter dividing the logit shifts every score the same way and cannot answer both. Platt scaling adds an intercept, fits slopes from 0.252 to 0.517, and cuts the calibration error to 0.018 with a Brier score of 0.1891. Both methods are fit on four folds and applied to the held-out fold, so no case is scored by a fit that saw it.

That improvement does not show the outcome layer works. Slopes well below one pull every score toward the base rate, so most of the lower calibration error is shrinkage rather than added signal. Neither scaling changes the ranking, which leaves the area under the curve of 0.562 as the measure of discrimination. The Platt Brier score passes the constant by 0.0004, and the outcome accuracy of 74.8 percent, with an interval from 72.1 to 77.4 percent, contains the 74.6 percent baseline.

The generator settles what those figures could have been. Success is drawn against a probability the generator computes from the profile, the regimen, and the patient risk factors. Marginalizing the year and the regimen, neither of which retrieval observes, gives the best score any predictor reading the seven retrieved features could reach. That ceiling is an area under the curve of 0.668, a Brier score of 0.176, and an accuracy of 75.5 percent, against 0.562, 0.1951, and 74.8 percent for retrieval. It is computed by `outcome_ceiling` in `validation.py` and scored by the same functions the arm is scored with, so the two sit on one scale.

The gap is the finding. Discrimination available above chance is 0.168 and retrieval holds 0.062 of it, so close to two thirds of the reachable signal is lost. Accuracy barely separates the three figures because the base rate sits near three quarters and a threshold of one half leaves the reported probability almost constant, falling below that threshold on 34 of the 1,000 cases. This is the reverse of the regimen layer, where the ceiling and the baseline sat within half a point of each other and the cohort was the binding constraint. Here the case base carries signal the neighborhood weighting does not recover, which makes the weights the part worth revisiting rather than the data.
