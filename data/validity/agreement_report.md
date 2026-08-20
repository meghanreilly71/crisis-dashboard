# Human vs LLM agreement — benchmark (n=100 articles)

Primary comparison run: **Run 1** for both variants, fixed before results were inspected. Runs 2 and 3 appear only as a robustness spread.

No interpretation, ranking, or preferred variant is expressed here.

## Per-label, primary run (Run 1)

`prev` columns are yes-rates. `±` is the min–max spread of that statistic across runs 1–3. `skew` marks a pooled base rate at or beyond 85% in one direction, where kappa is unreliable and raw agreement should be read alongside it.

| label | scope | n | prev H | prev A | prev B | raw A | raw B | κ A | κ B | α A | α B | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conflict | shared | 100 | 0.61 | 0.80 | 0.83 | 0.790 | 0.780 | 0.516 | 0.485 | 0.498 | 0.457 |  |
| human_interest | shared | 100 | 0.29 | 0.41 | 0.37 | 0.840 | 0.820 | 0.654 | 0.596 | 0.650 | 0.595 |  |
| economic | shared | 100 | 0.43 | 0.53 | 0.48 | 0.880 | 0.950 | 0.762 | 0.899 | 0.761 | 0.900 |  |
| deservingness | shared | 100 | 0.36 | 0.40 | 0.35 | 0.840 | 0.810 | 0.661 | 0.585 | 0.662 | 0.587 |  |
| responsibility | shared | 100 | 0.81 | 0.88 | 0.91 | 0.890 | 0.840 | 0.584 | 0.349 | 0.582 | 0.339 | yes |
| securitization | shared | 100 | 0.12 | 0.28 | 0.26 | 0.820 | 0.840 | 0.459 | 0.496 | 0.440 | 0.483 |  |
| othering | shared | 100 | 0.19 | 0.62 | 0.71 | 0.550 | 0.460 | 0.217 | 0.143 | 0.071 | -0.085 |  |
| agency | shared | 100 | 0.93 | 0.96 | 1.00 | 0.950 | 0.930 | 0.521 | 0.000 | 0.521 | -0.031 | yes |
| humanitarian | migration-only | 75 | 0.43 | 0.43 | 0.53 | 0.867 | 0.813 | 0.727 | 0.630 | 0.729 | 0.629 |  |
| security | migration-only | 75 | 0.25 | 0.41 | 0.40 | 0.787 | 0.827 | 0.533 | 0.615 | 0.523 | 0.609 |  |
| policy | migration-only | 75 | 0.77 | 0.83 | 0.88 | 0.920 | 0.893 | 0.751 | 0.635 | 0.752 | 0.630 |  |
| scientific | climate-only | 26 | 0.65 | 0.69 | 0.81 | 0.808 | 0.769 | 0.564 | 0.431 | 0.571 | 0.425 |  |
| crisis | climate-only | 26 | 0.65 | 0.73 | 0.69 | 0.846 | 0.885 | 0.641 | 0.738 | 0.646 | 0.743 |  |
| solutions | climate-only | 26 | 0.73 | 0.85 | 0.77 | 0.808 | 0.731 | 0.435 | 0.283 | 0.435 | 0.296 |  |
| victim | climate-only | 26 | 0.19 | 0.38 | 0.31 | 0.808 | 0.885 | 0.552 | 0.698 | 0.541 | 0.698 |  |
| skepticism | climate-only | 26 | 0.08 | 0.42 | 0.27 | 0.654 | 0.808 | 0.204 | 0.369 | 0.095 | 0.341 |  |

## Across-run spread (runs 1–3)

| label | κ A run1 | κ A range | κ B run1 | κ B range | raw A range | raw B range |
|---|---|---|---|---|---|---|
| conflict | 0.516 | 0.491–0.516 | 0.485 | 0.465–0.562 | 0.780–0.790 | 0.770–0.810 |
| human_interest | 0.654 | 0.654–0.672 | 0.596 | 0.580–0.596 | 0.840–0.850 | 0.820–0.820 |
| economic | 0.762 | 0.722–0.762 | 0.899 | 0.840–0.899 | 0.860–0.880 | 0.920–0.950 |
| deservingness | 0.661 | 0.657–0.680 | 0.585 | 0.566–0.595 | 0.840–0.850 | 0.800–0.810 |
| responsibility | 0.584 | 0.557–0.630 | 0.349 | 0.349–0.405 | 0.880–0.900 | 0.840–0.850 |
| securitization | 0.459 | 0.459–0.459 | 0.496 | 0.416–0.496 | 0.820–0.820 | 0.810–0.840 |
| othering | 0.217 | 0.208–0.217 | 0.143 | 0.143–0.167 | 0.540–0.550 | 0.460–0.470 |
| agency | 0.521 | 0.521–0.582 | 0.000 | 0.000–0.000 | 0.950–0.960 | 0.930–0.930 |
| humanitarian | 0.727 | 0.676–0.727 | 0.630 | 0.630–0.686 | 0.840–0.867 | 0.813–0.840 |
| security | 0.533 | 0.511–0.533 | 0.615 | 0.569–0.615 | 0.773–0.787 | 0.800–0.827 |
| policy | 0.751 | 0.618–0.751 | 0.635 | 0.579–0.635 | 0.880–0.920 | 0.880–0.893 |
| scientific | 0.564 | 0.564–0.564 | 0.431 | 0.241–0.539 | 0.808–0.808 | 0.692–0.808 |
| crisis | 0.641 | 0.641–0.641 | 0.738 | 0.283–0.738 | 0.846–0.846 | 0.692–0.885 |
| solutions | 0.435 | 0.435–0.435 | 0.283 | 0.283–0.570 | 0.808–0.808 | 0.731–0.846 |
| victim | 0.552 | 0.552–0.552 | 0.698 | 0.698–0.698 | 0.808–0.808 | 0.885–0.885 |
| skepticism | 0.204 | 0.204–0.272 | 0.369 | 0.369–0.435 | 0.654–0.731 | 0.808–0.846 |

## Aggregate

| variant | run | primary | cells | pooled raw | pooled κ | pooled α | macro κ | macro α |
|---|---|---|---|---|---|---|---|---|
| A | 1 | * | 1155 | 0.823 | 0.651 | 0.645 | 0.549 | 0.530 |
| A | 2 |  | 1155 | 0.822 | 0.647 | 0.642 | 0.547 | 0.531 |
| A | 3 |  | 1155 | 0.819 | 0.643 | 0.636 | 0.542 | 0.524 |
| B | 1 | * | 1155 | 0.813 | 0.631 | 0.624 | 0.497 | 0.476 |
| B | 2 |  | 1155 | 0.804 | 0.614 | 0.606 | 0.476 | 0.455 |
| B | 3 |  | 1155 | 0.816 | 0.638 | 0.631 | 0.505 | 0.485 |

## Excluded cells

445 cell(s) dropped as non-comparable (one side blank). Full ledger in `agreement_exclusions.csv`.

| idx | corpus | label | human | llm | reason |
|---|---|---|---|---|---|
| 72 | migration | crisis_present | no | None | human annotated, LLM pass not run for this corpus |
| 21 | climate | humanitarian_present | yes | None | human annotated, LLM pass not run for this corpus |
| 23 | climate | humanitarian_present | yes | None | human annotated, LLM pass not run for this corpus |
| 21 | climate | policy_present | yes | None | human annotated, LLM pass not run for this corpus |
| 23 | climate | policy_present | yes | None | human annotated, LLM pass not run for this corpus |
| 72 | migration | scientific_present | no | None | human annotated, LLM pass not run for this corpus |
| 21 | climate | security_present | no | None | human annotated, LLM pass not run for this corpus |
| 23 | climate | security_present | yes | None | human annotated, LLM pass not run for this corpus |
| 72 | migration | skepticism_present | no | None | human annotated, LLM pass not run for this corpus |
| 72 | migration | solutions_present | yes | None | human annotated, LLM pass not run for this corpus |
| 72 | migration | victim_present | no | None | human annotated, LLM pass not run for this corpus |
