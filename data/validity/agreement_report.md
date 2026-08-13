# Human vs LLM agreement — benchmark (n=100 articles)

Primary comparison run: **Run 1** for both variants, fixed before results were inspected. Runs 2 and 3 appear only as a robustness spread.

No interpretation, ranking, or preferred variant is expressed here.

## Per-label, primary run (Run 1)

`prev` columns are yes-rates. `±` is the min–max spread of that statistic across runs 1–3. `skew` marks a pooled base rate at or beyond 85% in one direction, where kappa is unreliable and raw agreement should be read alongside it.

| label | scope | n | prev H | prev A | prev B | raw A | raw B | κ A | κ B | α A | α B | skew |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| conflict | shared | 100 | 0.61 | 0.80 | 0.84 | 0.770 | 0.730 | 0.470 | 0.365 | 0.450 | 0.326 |  |
| human_interest | shared | 100 | 0.29 | 0.40 | 0.37 | 0.850 | 0.820 | 0.672 | 0.596 | 0.670 | 0.595 |  |
| economic | shared | 100 | 0.43 | 0.52 | 0.47 | 0.870 | 0.940 | 0.741 | 0.879 | 0.741 | 0.879 |  |
| deservingness | shared | 100 | 0.36 | 0.39 | 0.35 | 0.790 | 0.790 | 0.552 | 0.541 | 0.554 | 0.544 |  |
| responsibility | shared | 100 | 0.81 | 0.90 | 0.93 | 0.870 | 0.820 | 0.484 | 0.229 | 0.478 | 0.208 | yes |
| securitization | shared | 100 | 0.12 | 0.31 | 0.29 | 0.790 | 0.810 | 0.409 | 0.442 | 0.381 | 0.420 |  |
| othering | shared | 100 | 0.19 | 0.60 | 0.68 | 0.570 | 0.490 | 0.235 | 0.166 | 0.105 | -0.032 |  |
| agency | shared | 100 | 0.93 | 0.97 | 1.00 | 0.920 | 0.930 | 0.165 | 0.000 | 0.162 | -0.031 | yes |
| humanitarian | migration-only | 73 | 0.41 | 0.44 | 0.53 | 0.836 | 0.795 | 0.664 | 0.594 | 0.666 | 0.591 |  |
| security | migration-only | 73 | 0.26 | 0.41 | 0.40 | 0.740 | 0.781 | 0.431 | 0.514 | 0.420 | 0.507 |  |
| policy | migration-only | 73 | 0.77 | 0.84 | 0.89 | 0.904 | 0.877 | 0.701 | 0.577 | 0.701 | 0.569 |  |
| scientific | climate-only | 26 | 0.65 | 0.73 | 0.85 | 0.769 | 0.731 | 0.462 | 0.316 | 0.469 | 0.296 |  |
| crisis | climate-only | 26 | 0.65 | 0.73 | 0.69 | 0.846 | 0.885 | 0.641 | 0.738 | 0.646 | 0.743 |  |
| solutions | climate-only | 26 | 0.73 | 0.88 | 0.81 | 0.769 | 0.769 | 0.284 | 0.355 | 0.271 | 0.362 |  |
| victim | climate-only | 26 | 0.19 | 0.38 | 0.31 | 0.808 | 0.885 | 0.552 | 0.698 | 0.541 | 0.698 |  |
| skepticism | climate-only | 26 | 0.08 | 0.42 | 0.27 | 0.654 | 0.808 | 0.204 | 0.369 | 0.095 | 0.341 |  |

## Across-run spread (runs 1–3)

| label | κ A run1 | κ A range | κ B run1 | κ B range | raw A range | raw B range |
|---|---|---|---|---|---|---|
| conflict | 0.470 | 0.398–0.470 | 0.365 | 0.365–0.444 | 0.740–0.770 | 0.730–0.760 |
| human_interest | 0.672 | 0.654–0.691 | 0.596 | 0.580–0.596 | 0.840–0.860 | 0.820–0.820 |
| economic | 0.741 | 0.702–0.741 | 0.879 | 0.780–0.879 | 0.850–0.870 | 0.890–0.940 |
| deservingness | 0.552 | 0.547–0.571 | 0.541 | 0.523–0.552 | 0.790–0.800 | 0.780–0.790 |
| responsibility | 0.484 | 0.458–0.535 | 0.229 | 0.229–0.290 | 0.860–0.880 | 0.820–0.830 |
| securitization | 0.409 | 0.409–0.409 | 0.442 | 0.368–0.442 | 0.790–0.790 | 0.780–0.810 |
| othering | 0.235 | 0.226–0.235 | 0.166 | 0.166–0.191 | 0.560–0.570 | 0.490–0.500 |
| agency | 0.165 | 0.165–0.197 | 0.000 | 0.000–0.000 | 0.920–0.930 | 0.930–0.930 |
| humanitarian | 0.664 | 0.612–0.664 | 0.594 | 0.594–0.651 | 0.808–0.836 | 0.795–0.822 |
| security | 0.431 | 0.409–0.431 | 0.514 | 0.468–0.514 | 0.726–0.740 | 0.753–0.781 |
| policy | 0.701 | 0.597–0.701 | 0.577 | 0.518–0.577 | 0.877–0.904 | 0.863–0.877 |
| scientific | 0.462 | 0.462–0.462 | 0.316 | 0.120–0.431 | 0.769–0.769 | 0.654–0.769 |
| crisis | 0.641 | 0.641–0.641 | 0.738 | 0.283–0.738 | 0.846–0.846 | 0.692–0.885 |
| solutions | 0.284 | 0.284–0.284 | 0.355 | 0.355–0.435 | 0.769–0.769 | 0.769–0.808 |
| victim | 0.552 | 0.552–0.552 | 0.698 | 0.698–0.698 | 0.808–0.808 | 0.885–0.885 |
| skepticism | 0.204 | 0.204–0.272 | 0.369 | 0.369–0.435 | 0.654–0.731 | 0.808–0.846 |

## Aggregate

| variant | run | primary | cells | pooled raw | pooled κ | pooled α | macro κ | macro α |
|---|---|---|---|---|---|---|---|---|
| A | 1 | * | 1149 | 0.804 | 0.613 | 0.606 | 0.479 | 0.459 |
| A | 2 |  | 1149 | 0.803 | 0.612 | 0.605 | 0.478 | 0.460 |
| A | 3 |  | 1149 | 0.798 | 0.602 | 0.594 | 0.469 | 0.449 |
| B | 1 | * | 1149 | 0.799 | 0.603 | 0.595 | 0.461 | 0.438 |
| B | 2 |  | 1149 | 0.789 | 0.583 | 0.574 | 0.428 | 0.405 |
| B | 3 |  | 1149 | 0.799 | 0.604 | 0.595 | 0.454 | 0.434 |

## Excluded cells

494 cell(s) dropped as non-comparable (one side blank). Full ledger in `agreement_exclusions.csv`.

| idx | corpus | label | human | llm | reason |
|---|---|---|---|---|---|
| 14 | nan | agency_present | nan | no | human blank / non-binary, LLM annotated |
| 28 | nan | agency_present | nan | yes | human blank / non-binary, LLM annotated |
| 57 | nan | agency_present | nan | yes | human blank / non-binary, LLM annotated |
| 14 | nan | conflict_present | nan | yes | human blank / non-binary, LLM annotated |
| 28 | nan | conflict_present | nan | yes | human blank / non-binary, LLM annotated |
| 57 | nan | conflict_present | nan | no | human blank / non-binary, LLM annotated |
| 14 | nan | crisis_present | nan | yes | human blank / non-binary, LLM annotated |
| 25 | migration | crisis_present | None | yes | human blank / non-binary, LLM annotated |
| 72 | migration | crisis_present | no | None | human annotated, LLM pass not run for this corpus |
| 14 | nan | deservingness_present | nan | yes | human blank / non-binary, LLM annotated |
| 28 | nan | deservingness_present | nan | no | human blank / non-binary, LLM annotated |
| 57 | nan | deservingness_present | nan | no | human blank / non-binary, LLM annotated |
| 14 | nan | economic_present | nan | yes | human blank / non-binary, LLM annotated |
| 28 | nan | economic_present | nan | yes | human blank / non-binary, LLM annotated |
| 57 | nan | economic_present | nan | yes | human blank / non-binary, LLM annotated |
| 14 | nan | human_interest_present | nan | yes | human blank / non-binary, LLM annotated |
| 28 | nan | human_interest_present | nan | no | human blank / non-binary, LLM annotated |
| 57 | nan | human_interest_present | nan | no | human blank / non-binary, LLM annotated |
| 12 | climate | humanitarian_present | yes | None | human annotated, LLM pass not run for this corpus |
| 13 | climate | humanitarian_present | None | yes | human blank / non-binary, LLM annotated |
| 21 | climate | humanitarian_present | yes | None | human annotated, LLM pass not run for this corpus |
| 23 | climate | humanitarian_present | yes | None | human annotated, LLM pass not run for this corpus |
| 25 | migration | humanitarian_present | yes | None | human annotated, LLM pass not run for this corpus |
| 28 | nan | humanitarian_present | nan | no | human blank / non-binary, LLM annotated |
| 57 | nan | humanitarian_present | nan | no | human blank / non-binary, LLM annotated |
| 14 | nan | othering_present | nan | yes | human blank / non-binary, LLM annotated |
| 28 | nan | othering_present | nan | yes | human blank / non-binary, LLM annotated |
| 57 | nan | othering_present | nan | yes | human blank / non-binary, LLM annotated |
| 12 | climate | policy_present | yes | None | human annotated, LLM pass not run for this corpus |
| 13 | climate | policy_present | None | yes | human blank / non-binary, LLM annotated |
| 21 | climate | policy_present | yes | None | human annotated, LLM pass not run for this corpus |
| 23 | climate | policy_present | yes | None | human annotated, LLM pass not run for this corpus |
| 25 | migration | policy_present | yes | None | human annotated, LLM pass not run for this corpus |
| 28 | nan | policy_present | nan | no | human blank / non-binary, LLM annotated |
| 57 | nan | policy_present | nan | yes | human blank / non-binary, LLM annotated |
| 14 | nan | responsibility_present | nan | no | human blank / non-binary, LLM annotated |
| 28 | nan | responsibility_present | nan | no | human blank / non-binary, LLM annotated |
| 57 | nan | responsibility_present | nan | yes | human blank / non-binary, LLM annotated |
| 14 | nan | scientific_present | nan | no | human blank / non-binary, LLM annotated |
| 25 | migration | scientific_present | None | no | human blank / non-binary, LLM annotated |
| 72 | migration | scientific_present | no | None | human annotated, LLM pass not run for this corpus |
| 14 | nan | securitization_present | nan | no | human blank / non-binary, LLM annotated |
| 28 | nan | securitization_present | nan | no | human blank / non-binary, LLM annotated |
| 57 | nan | securitization_present | nan | no | human blank / non-binary, LLM annotated |
| 12 | climate | security_present | no | None | human annotated, LLM pass not run for this corpus |
| 13 | climate | security_present | None | no | human blank / non-binary, LLM annotated |
| 21 | climate | security_present | no | None | human annotated, LLM pass not run for this corpus |
| 23 | climate | security_present | yes | None | human annotated, LLM pass not run for this corpus |
| 25 | migration | security_present | no | None | human annotated, LLM pass not run for this corpus |
| 28 | nan | security_present | nan | yes | human blank / non-binary, LLM annotated |
| 57 | nan | security_present | nan | no | human blank / non-binary, LLM annotated |
| 14 | nan | skepticism_present | nan | no | human blank / non-binary, LLM annotated |
| 25 | migration | skepticism_present | None | no | human blank / non-binary, LLM annotated |
| 72 | migration | skepticism_present | no | None | human annotated, LLM pass not run for this corpus |
| 14 | nan | solutions_present | nan | no | human blank / non-binary, LLM annotated |
| 25 | migration | solutions_present | None | yes | human blank / non-binary, LLM annotated |
| 72 | migration | solutions_present | yes | None | human annotated, LLM pass not run for this corpus |
| 14 | nan | victim_present | nan | no | human blank / non-binary, LLM annotated |
| 25 | migration | victim_present | None | yes | human blank / non-binary, LLM annotated |
| 72 | migration | victim_present | no | None | human annotated, LLM pass not run for this corpus |
