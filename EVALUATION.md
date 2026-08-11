# Evaluation detail

Full scoring for every arm of the system. The [README](README.md) carries the headline
figures and the design rationale. Numbers here come from `Evaluation/validation_results.json`
and `Evaluation/per_drug_results.json`, both written by the commands in
[DEPLOYME.md](DEPLOYME.md) section 6.

## Contents

- [Real-world validation of the rule engine](#real-world-validation-of-the-rule-engine)
- [What bounds precision](#what-bounds-precision)
- [What bounds sensitivity](#what-bounds-sensitivity)
- [Per-drug resistance calls](#per-drug-resistance-calls)
- [Expert system](#expert-system)
- [Case-based reasoning, the experimental component](#case-based-reasoning-the-experimental-component)
- [Calibration](#calibration)

## Real-world validation of the rule engine

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

## What bounds precision

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

## What bounds sensitivity

Sensitivity should be read against what the input permits rather than against a perfect 100%. The engine recognizes only mutations the catalog grades as resistance-conferring, so an isolate carrying none of them arrives with an empty genotype and falls below the MDR threshold, whatever its phenotype. Of the 17,523 isolates measured at MDR or above, 3,562 carry no graded mutation, which puts the upper limit on sensitivity at 79.7%. The limit differs by tier, at 81.7% for MDR, 81.1% for pre-XDR, and 68.4% for XDR. Against those limits, the engine finds 75.8% of what is available at MDR, 75.8% at pre-XDR, and 80.0% at XDR. The raw figures fall as the tiers grow more resistant while the reachable share does not, though the four-point spread rests on 2,463 isolates and carries no interval, so it is a direction rather than a result.

## Per-drug resistance calls

Each drug is evaluated individually against the measured phenotype, using the WHO catalog as the reference. Calculations are based on isolates with results for each drug, ranging from 19,948 for capreomycin to 59,869 for rifampin, so each row shows its own denominator alongside the resistant count, instead of the 65,588 total in the header. The macro-F1 score is 0.588 for the engine and 0.611 for the catalog across the 15 drugs. Since both reference arms assess the same isolates per drug, the comparison uses a paired McNemar test rather than separate confidence intervals. Twelve out of 15 drugs show no discordant isolates, including both fluoroquinolones, since the catalog already grades levofloxacin and moxifloxacin from the same call on the DNA gyrase subunit A gene (gyrA), so the class expansion adds nothing there. These twelve drugs align on an isolate-by-isolate level rather than just overall.

The main distinction lies in the three injectables. The engine considers amikacin, kanamycin, and capreomycin as a single class, so a mutation affecting any one of them results in resistance to all three. In practice, cross-resistance among these injectables is only partial, making the trade-off generally disadvantageous. The precision on amikacin falls from 0.834 to 0.518 and on capreomycin from 0.776 to 0.439, against sensitivity gains of 2.2 and 4.2 percentage points. F1 falls on all three. The paired test puts the discordance at 1,524 isolates for amikacin with $\chi^2 = 1280.6$, 1,343 for capreomycin with $\chi^2 = 1061.5$, and 149 for kanamycin with $\chi^2 = 116.9$, each at $p < 10^{-26}$. Read the ratios rather than the p-values. Because the engine calls a strict superset of the catalog, the test asks only whether the added calls are more often wrong than right, and at these cohort sizes the p-value reflects the denominator. The three tests are also driven by the same mutations in the same isolates, falling in the 16S ribosomal RNA gene, written rrs, and the enhanced intracellular survival gene, written eis, so they are one finding measured three ways rather than three confirmations.

The behavior is documented as a quantifiable characteristic of the heuristic, rather than an implicit assumption. The score assigned to the arm does not directly reflect the rule's intent, as the expansion only adds entries to the exclusion list. Its goal is to prevent a likely-failing drug from being included in a regimen, rather than to predict a phenotype. The scoring runs through `python Evaluation/metrics.py`, which writes `Evaluation/per_drug_results.json`.

## Expert system

The natural-language layer is assessed through execution match, pairing each question with a gold query. A generated query is deemed correct if it produces the same result set. On claude-sonnet-4-6, the latest run answered ten out of eleven questions, achieving a correctness rate of 90.9%, with a Wilson interval of 62.3% to 98.4%. It is better to interpret the interval instead of the single-point estimate, as eleven questions are insufficient to draw conclusions about translation quality in either direction.

Only nine questions are scored this way. The remaining two require information that the graph cannot provide and are scored on refusal, which passes when the generated text either states the question is unanswerable or fails the read-only guard. An explicit refusal and an attempted write count alike, and the result file notes the generated query only when an item fails, so the two passing refusals leave no trace for review.

Generation runs at temperature zero, which does not make the arm reproducible. Four runs returned ten of eleven three times and eleven of eleven once. Every failing run failed the same lookup, where the query collects a relationship property without binding the relationship. Memgraph rejects it as an unbound variable. One unstable question describes the arm better than a drifting score, and an invalid query is a better failure than a plausible wrong answer.

The test suite verifies the deterministic components of the layer, including the read-only guard, routing, and normalization (which removes order clauses the database cannot satisfy after an aggregate but retains them when a LIMIT depends on it).

## Case-based reasoning, the experimental component

Regimen accuracy is 77.2%, with a bootstrap interval from 74.5% to 79.8%. It is slightly below the majority-class baseline of 79.3% and near the ceiling of 79.8%. The ceiling marginalizes the year, since the generator selects the regimen from profile and year while the year is independent of every feature retrieval matches on. Outcome accuracy stands at 74.8%, marginally above its baseline of 74.6% and below its own ceiling of 75.5%, which is computed the same way. Calibration reports that comparison in full, since accuracy is the weakest of the three measures at this base rate.

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

The generator draws the regimen from the profile and the year. The year never enters the query, so retrieval cannot match on it, although the ranking does discount older cases and that discount is worth 0.4 points of regimen accuracy. What retrieval can reach on this cohort is therefore bounded by the per-profile lookup, whatever weights it carries. The limit belongs to the synthetic cases rather than to the weighting, so a weaker score here is not on its own an argument for retuning. The outcome layer measures the other way, and Calibration sets out why.

## Calibration

The predicted probability represents the proportion of the ten retrieved neighbors that succeeded, smoothed by adding one success and one failure. This smoothing is important. Without it, a raw share would be exactly one if all ten neighbors succeeded, which occurred in 77 of the 1,000 cases. In those cases the logit that both scaling methods fit is unbounded, and these few scores could otherwise drive the fit on their own.

The expected calibration error for the smoothed probability is 0.0942, and the Brier score is 0.1951. A constant prediction based on the base rate achieves a score of 0.1895, outperforming it. This constant is fitted to the outcomes it predicts, making it the most robust baseline of its kind and serving as a conservative point of comparison.

Temperature scaling averages at 1.113, with fold values from 1.05 to 1.238, but it increases the calibration error from 0.0942 to 0.103, indicating an issue with direction rather than spread. The middle bins are under-confident, with accuracy above their stated probabilities, while the top two bins are over-confident. A single parameter dividing the logit shifts every score the same way and cannot answer both. Platt scaling introduces an intercept, adjusts slopes from 0.252 to 0.517, and reduces the calibration error to 0.018, with a Brier score of 0.1891. Both methods are trained on four folds and applied to the held-out fold, ensuring no data leakage.

This improvement does not show that the outcome layer works. Slopes well below one pull every score toward the base rate, so most of the lower calibration error is shrinkage rather than added signal. Neither scaling changes the ranking, which leaves the area under the curve of 0.562 as the measure of discrimination. The Platt Brier score passes the constant by 0.0004, and the outcome accuracy of 74.8%, with an interval from 72.1% to 77.4%, contains the 74.6% baseline.

The generator settles what those figures could have been. Success is drawn against a probability the generator computes from the profile, the regimen, and the patient risk factors. Marginalizing the year and the regimen, neither of which retrieval observes, gives the best score any predictor reading the seven retrieved features could reach. That ceiling is an area under the curve of 0.668, a Brier score of 0.176, and an accuracy of 75.5%, against 0.562, 0.1951, and 74.8% for retrieval. It is computed by `outcome_ceiling` in `validation.py` and scored by the same functions the arm is scored with, so the two sit on one scale.

The gap is the finding. Discrimination available above chance is 0.168 and retrieval holds 0.062 of it, so close to two thirds of the reachable signal is lost. Accuracy barely separates the three figures because the base rate sits near three quarters and a threshold of one half leaves the reported probability almost constant, falling below that threshold on 34 of the 1,000 cases. This is the reverse of what the regimen layer showed. There the ceiling and the baseline sat within half a point of the measured score, so the cohort was the binding constraint. Here the case base carries signal the neighborhood weighting does not recover, which makes the weights the part worth revisiting rather than the data.
