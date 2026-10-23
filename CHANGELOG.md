<a id="v0.4.0"></a>
# [v0.4.0](https://github.com/jordandeklerk/moderndid/releases/tag/v0.4.0) - 2026-10-23

<!-- Release notes generated using configuration in .github/release.yml at v0.4.0 -->

## What's Changed
* DOC: add runnable Colab and molab notebooks for the examples by [@jordandeklerk](https://github.com/jordandeklerk) in [#382](https://github.com/jordandeklerk/moderndid/pull/382)
* BUG: fix the R comparison notebooks in Colab and drop their saved outputs by [@jordandeklerk](https://github.com/jordandeklerk) in [#383](https://github.com/jordandeklerk/moderndid/pull/383)
* MAINT: test installs without the extras in CI by [@jordandeklerk](https://github.com/jordandeklerk) in [#384](https://github.com/jordandeklerk/moderndid/pull/384)
* MAINT: rename `dropna` to `na_rm` in `agg_ddd` and `compute_aggte` by [@jordandeklerk](https://github.com/jordandeklerk) in [#385](https://github.com/jordandeklerk/moderndid/pull/385)
* DOC: reorganize the index pages and merge the Development and Contributing tabs by [@jordandeklerk](https://github.com/jordandeklerk) in [#386](https://github.com/jordandeklerk/moderndid/pull/386)
* DOC: reorganize the API reference and redesign the landing page by [@jordandeklerk](https://github.com/jordandeklerk) in [#387](https://github.com/jordandeklerk/moderndid/pull/387)
* DOC: cap the prose width and tidy callouts, equations, and tables by [@jordandeklerk](https://github.com/jordandeklerk) in [#388](https://github.com/jordandeklerk/moderndid/pull/388)
* DOC: trim the guide pages, restore full-width text, and center the landing logo by [@jordandeklerk](https://github.com/jordandeklerk) in [#389](https://github.com/jordandeklerk/moderndid/pull/389)
* DOC: shorten the FAQ answers and keep sidebar sections open by [@jordandeklerk](https://github.com/jordandeklerk) in [#390](https://github.com/jordandeklerk/moderndid/pull/390)
* DOC: trim the DiD example pages and their check outputs by [@jordandeklerk](https://github.com/jordandeklerk) in [#393](https://github.com/jordandeklerk/moderndid/pull/393)
* DOC: landing page redesign by [@jordandeklerk](https://github.com/jordandeklerk) in [#394](https://github.com/jordandeklerk/moderndid/pull/394)
* DOC: set the footer wordmark in the block letters by [@jordandeklerk](https://github.com/jordandeklerk) in [#395](https://github.com/jordandeklerk/moderndid/pull/395)
* DOC: trim the continuous treatment, fracking, dynamic balancing, and NPIV examples by [@jordandeklerk](https://github.com/jordandeklerk) in [#396](https://github.com/jordandeklerk/moderndid/pull/396)
* DOC: draw figures at the column width by [@jordandeklerk](https://github.com/jordandeklerk) in [#397](https://github.com/jordandeklerk/moderndid/pull/397)
* DOC: drop the window frame and search prompt from the landing page by [@jordandeklerk](https://github.com/jordandeklerk) in [#398](https://github.com/jordandeklerk/moderndid/pull/398)
* ENH: support didml results in to_df and the plots by [@jordandeklerk](https://github.com/jordandeklerk) in [#399](https://github.com/jordandeklerk/moderndid/pull/399)
* ENH: run att_gt on Spark DataFrames by [@jordandeklerk](https://github.com/jordandeklerk) in [#400](https://github.com/jordandeklerk/moderndid/pull/400)
* DOC: split the Spark guide in two and rewrite the shard engine section by [@jordandeklerk](https://github.com/jordandeklerk) in [#401](https://github.com/jordandeklerk/moderndid/pull/401)
* TEST: ignore pyfixest's pandas 3.1 deprecations in the ETWFE warning tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#402](https://github.com/jordandeklerk/moderndid/pull/402)
* BUG: read decimal value columns as doubles in every estimator and fix pandas version by [@jordandeklerk](https://github.com/jordandeklerk) in [#403](https://github.com/jordandeklerk/moderndid/pull/403)
* REL: prepare release v0.4.0 by [@jordandeklerk](https://github.com/jordandeklerk) in [#404](https://github.com/jordandeklerk/moderndid/pull/404)


**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.3.1...v0.4.0

[Changes][v0.4.0]


<a id="v0.3.1"></a>
# [v0.3.1](https://github.com/jordandeklerk/moderndid/releases/tag/v0.3.1) - 2026-10-16

<!-- Release notes generated using configuration in .github/release.yml at v0.3.1 -->



**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.3.1...v0.3.1

<!-- Release notes generated using configuration in .github/release.yml at v0.3.1 -->

## What's Changed
* BUG: load plots without scikit-learn and drop ECOS from the extras by [@jordandeklerk](https://github.com/jordandeklerk) in [#379](https://github.com/jordandeklerk/moderndid/pull/379)
* REL: prepare release v0.3.1 by [@jordandeklerk](https://github.com/jordandeklerk) in [#380](https://github.com/jordandeklerk/moderndid/pull/380)


**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.3.0...v0.3.1

[Changes][v0.3.1]


<a id="v0.3.0"></a>
# [v0.3.0](https://github.com/jordandeklerk/moderndid/releases/tag/v0.3.0) - 2026-10-16

## What's Changed
* Update changelog for v0.2.0 by [@github-actions](https://github.com/github-actions)[bot] in [#246](https://github.com/jordandeklerk/moderndid/pull/246)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.6 by [@renovate](https://github.com/renovate)[bot] in [#245](https://github.com/jordandeklerk/moderndid/pull/245)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#244](https://github.com/jordandeklerk/moderndid/pull/244)
* Update changelog for v0.2.0 by [@github-actions](https://github.com/github-actions)[bot] in [#247](https://github.com/jordandeklerk/moderndid/pull/247)
* FEAT: enhance `att_gt()` dask and spark methods  by [@jordandeklerk](https://github.com/jordandeklerk) in [#248](https://github.com/jordandeklerk/moderndid/pull/248)
* REF: make `npiv` its own sub-package by [@jordandeklerk](https://github.com/jordandeklerk) in [#249](https://github.com/jordandeklerk/moderndid/pull/249)
* FEAT: add the Wooldridge extended two-way fixed effects estimator by [@jordandeklerk](https://github.com/jordandeklerk) in [#250](https://github.com/jordandeklerk/moderndid/pull/250)
* CHG: update pyproject.toml to include 'etwfe' in the all dependencies by [@jordandeklerk](https://github.com/jordandeklerk) in [#251](https://github.com/jordandeklerk/moderndid/pull/251)
* MAINT: fix the failing API docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#252](https://github.com/jordandeklerk/moderndid/pull/252)
* DOC: update pyproject.toml to include additional dependencies by [@jordandeklerk](https://github.com/jordandeklerk) in [#253](https://github.com/jordandeklerk/moderndid/pull/253)
* FEAT: add R validation tests for ETWFE by [@jordandeklerk](https://github.com/jordandeklerk) in [#254](https://github.com/jordandeklerk/moderndid/pull/254)
* DOC: add more documentation for ETWFE and related functionalities by [@jordandeklerk](https://github.com/jordandeklerk) in [#255](https://github.com/jordandeklerk/moderndid/pull/255)
* DEP: update pyproject.toml to include formulaic dependency for ETWFE by [@jordandeklerk](https://github.com/jordandeklerk) in [#256](https://github.com/jordandeklerk/moderndid/pull/256)
* DEP: add more dependencies for `etwfe` by [@jordandeklerk](https://github.com/jordandeklerk) in [#257](https://github.com/jordandeklerk/moderndid/pull/257)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.7 by [@renovate](https://github.com/renovate)[bot] in [#259](https://github.com/jordandeklerk/moderndid/pull/259)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#258](https://github.com/jordandeklerk/moderndid/pull/258)
* DOC: better documentation for installation and intro by [@jordandeklerk](https://github.com/jordandeklerk) in [#260](https://github.com/jordandeklerk/moderndid/pull/260)
* DEP: restrict pyfixest version in dependencies by [@jordandeklerk](https://github.com/jordandeklerk) in [#261](https://github.com/jordandeklerk/moderndid/pull/261)
* DOC: enhancements to background and user guide docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#262](https://github.com/jordandeklerk/moderndid/pull/262)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#265](https://github.com/jordandeklerk/moderndid/pull/265)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#264](https://github.com/jordandeklerk/moderndid/pull/264)
* Pin pypa/gh-action-pypi-publish action to ed0c539 by [@renovate](https://github.com/renovate)[bot] in [#263](https://github.com/jordandeklerk/moderndid/pull/263)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.9 by [@renovate](https://github.com/renovate)[bot] in [#266](https://github.com/jordandeklerk/moderndid/pull/266)
* Update pypa/gh-action-pypi-publish digest to cef2210 by [@renovate](https://github.com/renovate)[bot] in [#268](https://github.com/jordandeklerk/moderndid/pull/268)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.10 by [@renovate](https://github.com/renovate)[bot] in [#269](https://github.com/jordandeklerk/moderndid/pull/269)
* FEAT: add dynamic balancing weights DiD estimator by [@jordandeklerk](https://github.com/jordandeklerk) in [#270](https://github.com/jordandeklerk/moderndid/pull/270)
* Update github/codeql-action action to v4.35.2 by [@renovate](https://github.com/renovate)[bot] in [#271](https://github.com/jordandeklerk/moderndid/pull/271)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#273](https://github.com/jordandeklerk/moderndid/pull/273)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.11 by [@renovate](https://github.com/renovate)[bot] in [#272](https://github.com/jordandeklerk/moderndid/pull/272)
* DOC: update plotting documentation by [@jordandeklerk](https://github.com/jordandeklerk) in [#274](https://github.com/jordandeklerk/moderndid/pull/274)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.12 by [@renovate](https://github.com/renovate)[bot] in [#275](https://github.com/jordandeklerk/moderndid/pull/275)
* Update codecov/codecov-action action to v6 by [@renovate](https://github.com/renovate)[bot] in [#276](https://github.com/jordandeklerk/moderndid/pull/276)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#277](https://github.com/jordandeklerk/moderndid/pull/277)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#278](https://github.com/jordandeklerk/moderndid/pull/278)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#279](https://github.com/jordandeklerk/moderndid/pull/279)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.13 by [@renovate](https://github.com/renovate)[bot] in [#280](https://github.com/jordandeklerk/moderndid/pull/280)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#281](https://github.com/jordandeklerk/moderndid/pull/281)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#284](https://github.com/jordandeklerk/moderndid/pull/284)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.14 by [@renovate](https://github.com/renovate)[bot] in [#283](https://github.com/jordandeklerk/moderndid/pull/283)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#282](https://github.com/jordandeklerk/moderndid/pull/282)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#285](https://github.com/jordandeklerk/moderndid/pull/285)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#286](https://github.com/jordandeklerk/moderndid/pull/286)
* Update pre-commit hooks to v0.15.16 by [@renovate](https://github.com/renovate)[bot] in [#287](https://github.com/jordandeklerk/moderndid/pull/287)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#288](https://github.com/jordandeklerk/moderndid/pull/288)
* Update pre-commit hooks by [@renovate](https://github.com/renovate)[bot] in [#289](https://github.com/jordandeklerk/moderndid/pull/289)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#291](https://github.com/jordandeklerk/moderndid/pull/291)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#292](https://github.com/jordandeklerk/moderndid/pull/292)
* Update GitHub Actions to v7 by [@renovate](https://github.com/renovate)[bot] in [#293](https://github.com/jordandeklerk/moderndid/pull/293)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#294](https://github.com/jordandeklerk/moderndid/pull/294)
* FEAT: initial implementation for MLDiD by [@jordandeklerk](https://github.com/jordandeklerk) in [#295](https://github.com/jordandeklerk/moderndid/pull/295)
* REF: Remove Dask test jobs from CI by [@jordandeklerk](https://github.com/jordandeklerk) in [#296](https://github.com/jordandeklerk/moderndid/pull/296)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#299](https://github.com/jordandeklerk/moderndid/pull/299)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#298](https://github.com/jordandeklerk/moderndid/pull/298)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.21 - autoclosed by [@renovate](https://github.com/renovate)[bot] in [#297](https://github.com/jordandeklerk/moderndid/pull/297)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#302](https://github.com/jordandeklerk/moderndid/pull/302)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.22 by [@renovate](https://github.com/renovate)[bot] in [#301](https://github.com/jordandeklerk/moderndid/pull/301)
* Update pypa/gh-action-pypi-publish digest to ba38be9 by [@renovate](https://github.com/renovate)[bot] in [#300](https://github.com/jordandeklerk/moderndid/pull/300)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#305](https://github.com/jordandeklerk/moderndid/pull/305)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.0 by [@renovate](https://github.com/renovate)[bot] in [#304](https://github.com/jordandeklerk/moderndid/pull/304)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#303](https://github.com/jordandeklerk/moderndid/pull/303)
* Update pypa/gh-action-pypi-publish digest to dc37677 by [@renovate](https://github.com/renovate)[bot] in [#306](https://github.com/jordandeklerk/moderndid/pull/306)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.2 by [@renovate](https://github.com/renovate)[bot] in [#307](https://github.com/jordandeklerk/moderndid/pull/307)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#308](https://github.com/jordandeklerk/moderndid/pull/308)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#309](https://github.com/jordandeklerk/moderndid/pull/309)
* Update dependency flit_core to v4 by [@renovate](https://github.com/renovate)[bot] in [#310](https://github.com/jordandeklerk/moderndid/pull/310)
* Update github/codeql-action action to v4.37.7 by [@renovate](https://github.com/renovate)[bot] in [#311](https://github.com/jordandeklerk/moderndid/pull/311)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#313](https://github.com/jordandeklerk/moderndid/pull/313)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.4 by [@renovate](https://github.com/renovate)[bot] in [#312](https://github.com/jordandeklerk/moderndid/pull/312)
* Update github/codeql-action action to v4.37.8 by [@renovate](https://github.com/renovate)[bot] in [#314](https://github.com/jordandeklerk/moderndid/pull/314)
* Update GitHub Actions (major) by [@renovate](https://github.com/renovate)[bot] in [#315](https://github.com/jordandeklerk/moderndid/pull/315)
* Update github/codeql-action action to v4.37.9 by [@renovate](https://github.com/renovate)[bot] in [#316](https://github.com/jordandeklerk/moderndid/pull/316)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.5 by [@renovate](https://github.com/renovate)[bot] in [#317](https://github.com/jordandeklerk/moderndid/pull/317)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#318](https://github.com/jordandeklerk/moderndid/pull/318)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#319](https://github.com/jordandeklerk/moderndid/pull/319)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#320](https://github.com/jordandeklerk/moderndid/pull/320)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.7 by [@renovate](https://github.com/renovate)[bot] in [#321](https://github.com/jordandeklerk/moderndid/pull/321)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#324](https://github.com/jordandeklerk/moderndid/pull/324)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.8 by [@renovate](https://github.com/renovate)[bot] in [#323](https://github.com/jordandeklerk/moderndid/pull/323)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#322](https://github.com/jordandeklerk/moderndid/pull/322)
* Update github/codeql-action action to v4.38.2 by [@renovate](https://github.com/renovate)[bot] in [#325](https://github.com/jordandeklerk/moderndid/pull/325)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.16.9 by [@renovate](https://github.com/renovate)[bot] in [#326](https://github.com/jordandeklerk/moderndid/pull/326)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#327](https://github.com/jordandeklerk/moderndid/pull/327)
* MAINT: update tooling, CI, and environment configuration by [@jordandeklerk](https://github.com/jordandeklerk) in [#331](https://github.com/jordandeklerk/moderndid/pull/331)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#330](https://github.com/jordandeklerk/moderndid/pull/330)
* MAINT: remove the Spark, Dask, and distributed backends by [@jordandeklerk](https://github.com/jordandeklerk) in [#332](https://github.com/jordandeklerk/moderndid/pull/332)
* ENH: update core preprocessing, reports, converters, and plots by [@jordandeklerk](https://github.com/jordandeklerk) in [#333](https://github.com/jordandeklerk/moderndid/pull/333)
* ENH: update att_gt and aggte and their tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#334](https://github.com/jordandeklerk/moderndid/pull/334)
* ENH: update the two-period estimators and their tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#335](https://github.com/jordandeklerk/moderndid/pull/335)
* ENH: update ddd and agg_ddd and their tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#336](https://github.com/jordandeklerk/moderndid/pull/336)
* ENH: fix `did_multiplegt` controls and inference by [@jordandeklerk](https://github.com/jordandeklerk) in [#337](https://github.com/jordandeklerk/moderndid/pull/337)
* ENH: tighten cont_did options, windows, and the pte pipeline by [@jordandeklerk](https://github.com/jordandeklerk) in [#340](https://github.com/jordandeklerk/moderndid/pull/340)
* BUG: fix dyn_balancing variance, weights, and missing data by [@jordandeklerk](https://github.com/jordandeklerk) in [#341](https://github.com/jordandeklerk/moderndid/pull/341)
* BUG: fix honest_did confidence sets, restrictions, and clustering by [@jordandeklerk](https://github.com/jordandeklerk) in [#342](https://github.com/jordandeklerk/moderndid/pull/342)
* BUG: fix etwfe reference cohorts, term order, and input checks by [@jordandeklerk](https://github.com/jordandeklerk) in [#343](https://github.com/jordandeklerk/moderndid/pull/343)
* BUG: fix npiv basis matching, dimension selection, and bands by [@jordandeklerk](https://github.com/jordandeklerk) in [#345](https://github.com/jordandeklerk/moderndid/pull/345)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#344](https://github.com/jordandeklerk/moderndid/pull/344)
* ENH: fix the didml estimand and bootstrap its standard errors by [@jordandeklerk](https://github.com/jordandeklerk) in [#346](https://github.com/jordandeklerk/moderndid/pull/346)
* TEST: compare did, drdid, and ddd with their reference packages by [@jordandeklerk](https://github.com/jordandeklerk) in [#347](https://github.com/jordandeklerk/moderndid/pull/347)
* TEST: compare the other estimators with their reference packages by [@jordandeklerk](https://github.com/jordandeklerk) in [#348](https://github.com/jordandeklerk/moderndid/pull/348)
* BUG: require scikit-learn 1.7 and pass the lasso grid as alphas by [@jordandeklerk](https://github.com/jordandeklerk) in [#349](https://github.com/jordandeklerk/moderndid/pull/349)
* MAINT: replace benchmark/ with an ASV suite and spin compare by [@jordandeklerk](https://github.com/jordandeklerk) in [#350](https://github.com/jordandeklerk/moderndid/pull/350)
* DOC: add the docs theme, configuration, and API pages by [@jordandeklerk](https://github.com/jordandeklerk) in [#351](https://github.com/jordandeklerk/moderndid/pull/351)
* MAINT: fix logo sizing by [@jordandeklerk](https://github.com/jordandeklerk) in [#352](https://github.com/jordandeklerk/moderndid/pull/352)
* DOC: rewrite the did, drdid, and triple-difference background by [@jordandeklerk](https://github.com/jordandeklerk) in [#353](https://github.com/jordandeklerk/moderndid/pull/353)
* DOC: rewrite the didcont, didinter, and diddynamic background by [@jordandeklerk](https://github.com/jordandeklerk) in [#354](https://github.com/jordandeklerk/moderndid/pull/354)
* DOC: rewrite the didhonest, etwfe, and npiv background by [@jordandeklerk](https://github.com/jordandeklerk) in [#355](https://github.com/jordandeklerk/moderndid/pull/355)
* DOC: rewrite the getting started and overview guides by [@jordandeklerk](https://github.com/jordandeklerk) in [#356](https://github.com/jordandeklerk/moderndid/pull/356)
* DOC: rewrite the data, results, plotting, and tables guides by [@jordandeklerk](https://github.com/jordandeklerk) in [#357](https://github.com/jordandeklerk/moderndid/pull/357)
* DOC: rewrite the staggered, triple, and continuous examples by [@jordandeklerk](https://github.com/jordandeklerk) in [#358](https://github.com/jordandeklerk/moderndid/pull/358)
* DOC: rewrite the remaining example guides by [@jordandeklerk](https://github.com/jordandeklerk) in [#359](https://github.com/jordandeklerk/moderndid/pull/359)
* DOC: add Comparisons with R and its first four notebooks by [@jordandeklerk](https://github.com/jordandeklerk) in [#360](https://github.com/jordandeklerk/moderndid/pull/360)
* DOC: add the remaining Comparisons with R notebooks by [@jordandeklerk](https://github.com/jordandeklerk) in [#361](https://github.com/jordandeklerk/moderndid/pull/361)
* DOC: rewrite the contributing and developer pages by [@jordandeklerk](https://github.com/jordandeklerk) in [#362](https://github.com/jordandeklerk/moderndid/pull/362)
* BUG: handle zero-variance statistics in honest_did's ARP test by [@jordandeklerk](https://github.com/jordandeklerk) in [#363](https://github.com/jordandeklerk/moderndid/pull/363)
* MAINT: run the test suite in parallel and install it with uv by [@jordandeklerk](https://github.com/jordandeklerk) in [#364](https://github.com/jordandeklerk/moderndid/pull/364)
* TEST: speed up the slowest tests and switch CI coverage to sysmon by [@jordandeklerk](https://github.com/jordandeklerk) in [#365](https://github.com/jordandeklerk/moderndid/pull/365)
* DOC: shorten the did, drdid, and triple-difference docstrings by [@jordandeklerk](https://github.com/jordandeklerk) in [#366](https://github.com/jordandeklerk/moderndid/pull/366)
* DOC: shorten the continuous, intertemporal, balancing, and ML docstrings by [@jordandeklerk](https://github.com/jordandeklerk) in [#367](https://github.com/jordandeklerk/moderndid/pull/367)
* DOC: shorten the honest_did, etwfe, emfx, and npiv docstrings by [@jordandeklerk](https://github.com/jordandeklerk) in [#368](https://github.com/jordandeklerk/moderndid/pull/368)
* DOC: fold and simplify code cells on the comparison and guide pages by [@jordandeklerk](https://github.com/jordandeklerk) in [#369](https://github.com/jordandeklerk/moderndid/pull/369)
* BUG: fix repeated cross-section DR standard errors and CCK draws by [@jordandeklerk](https://github.com/jordandeklerk) in [#370](https://github.com/jordandeklerk/moderndid/pull/370)
* ENH: speed up the bootstraps and honest_did's conditional tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#371](https://github.com/jordandeklerk/moderndid/pull/371)
* MAINT: remove Numba  by [@jordandeklerk](https://github.com/jordandeklerk) in [#372](https://github.com/jordandeklerk/moderndid/pull/372)
* BUG: fix the CuPy backend's crashes and test it on a real GPU by [@jordandeklerk](https://github.com/jordandeklerk) in [#373](https://github.com/jordandeklerk/moderndid/pull/373)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#338](https://github.com/jordandeklerk/moderndid/pull/338)
* DOC: add an influence function background section by [@jordandeklerk](https://github.com/jordandeklerk) in [#374](https://github.com/jordandeklerk/moderndid/pull/374)
* REL: get ready for a release by [@jordandeklerk](https://github.com/jordandeklerk) in [#375](https://github.com/jordandeklerk/moderndid/pull/375)
* DOC: fix ordering of links by [@jordandeklerk](https://github.com/jordandeklerk) in [#376](https://github.com/jordandeklerk/moderndid/pull/376)
* MAINT: fix skill by [@jordandeklerk](https://github.com/jordandeklerk) in [#377](https://github.com/jordandeklerk/moderndid/pull/377)


**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.2.0...v0.3.0

[Changes][v0.3.0]


<a id="v0.2.0"></a>
# [v0.2.0](https://github.com/jordandeklerk/moderndid/releases/tag/v0.2.0) - 2026-03-16

## What's Changed
* Update changelog for v0.1.1 by [@github-actions](https://github.com/github-actions)[bot] in [#214](https://github.com/jordandeklerk/moderndid/pull/214)
* DOC: add better descriptions for dataset functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#215](https://github.com/jordandeklerk/moderndid/pull/215)
* FEAT: add GPU acceleration for `cont_did()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#216](https://github.com/jordandeklerk/moderndid/pull/216)
* REF: update installation guide and add `dev` directory by [@jordandeklerk](https://github.com/jordandeklerk) in [#217](https://github.com/jordandeklerk/moderndid/pull/217)
* REF: unify synthetic data function names by [@jordandeklerk](https://github.com/jordandeklerk) in [#218](https://github.com/jordandeklerk/moderndid/pull/218)
* Update actions/download-artifact action to v8 by [@renovate](https://github.com/renovate)[bot] in [#220](https://github.com/jordandeklerk/moderndid/pull/220)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.4 by [@renovate](https://github.com/renovate)[bot] in [#219](https://github.com/jordandeklerk/moderndid/pull/219)
* MAINT: reorganize backend CuPY functionality by [@jordandeklerk](https://github.com/jordandeklerk) in [#221](https://github.com/jordandeklerk/moderndid/pull/221)
* FEAT: add more treatment effects and Bell-McCaffrey small-sample inference for `did_multiplegt()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#222](https://github.com/jordandeklerk/moderndid/pull/222)
* DEV: enhance development documentation and CI by [@jordandeklerk](https://github.com/jordandeklerk) in [#223](https://github.com/jordandeklerk/moderndid/pull/223)
* BUG: fix clustered standard errors not propagating through `att_gt()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#224](https://github.com/jordandeklerk/moderndid/pull/224)
* FEAT: add maketables plug-in interface for publication-quality tables by [@jordandeklerk](https://github.com/jordandeklerk) in [#225](https://github.com/jordandeklerk/moderndid/pull/225)
* DOC: update README to include publication tables feature by [@jordandeklerk](https://github.com/jordandeklerk) in [#226](https://github.com/jordandeklerk/moderndid/pull/226)
* DOC: reorganize plotting guide by [@jordandeklerk](https://github.com/jordandeklerk) in [#227](https://github.com/jordandeklerk/moderndid/pull/227)
* FEAT: expose result extraction for estimators by [@jordandeklerk](https://github.com/jordandeklerk) in [#228](https://github.com/jordandeklerk/moderndid/pull/228)
* REF: update result objects for core estimators by [@jordandeklerk](https://github.com/jordandeklerk) in [#229](https://github.com/jordandeklerk/moderndid/pull/229)
* BUG: fix docs failing by [@jordandeklerk](https://github.com/jordandeklerk) in [#230](https://github.com/jordandeklerk/moderndid/pull/230)
* DOC: update user guide and README by [@jordandeklerk](https://github.com/jordandeklerk) in [#231](https://github.com/jordandeklerk/moderndid/pull/231)
* DOC: update README example to match Callaway Sant'Anna example by [@jordandeklerk](https://github.com/jordandeklerk) in [#232](https://github.com/jordandeklerk/moderndid/pull/232)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.5 by [@renovate](https://github.com/renovate)[bot] in [#234](https://github.com/jordandeklerk/moderndid/pull/234)
* Update github/codeql-action action to v4.32.6 by [@renovate](https://github.com/renovate)[bot] in [#233](https://github.com/jordandeklerk/moderndid/pull/233)
* DOC: update installation guide link by [@jordandeklerk](https://github.com/jordandeklerk) in [#235](https://github.com/jordandeklerk/moderndid/pull/235)
* DOC: enhance new estimator guide and architecture docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#236](https://github.com/jordandeklerk/moderndid/pull/236)
* MAINT: update installation to not include spark by default by [@jordandeklerk](https://github.com/jordandeklerk) in [#237](https://github.com/jordandeklerk/moderndid/pull/237)
* DOC: update GPU guide by [@jordandeklerk](https://github.com/jordandeklerk) in [#238](https://github.com/jordandeklerk/moderndid/pull/238)
* CI: update test workflows to support multiple OS environments by [@jordandeklerk](https://github.com/jordandeklerk) in [#239](https://github.com/jordandeklerk/moderndid/pull/239)
* ENH: Add error handling for NA values by [@jordandeklerk](https://github.com/jordandeklerk) in [#240](https://github.com/jordandeklerk/moderndid/pull/240)
* ENH: align Wald pre-test, confidence bands, and bootstrap quantiles by [@jordandeklerk](https://github.com/jordandeklerk) in [#241](https://github.com/jordandeklerk/moderndid/pull/241)
* CI: conditionally ignore validation tests on Windows by [@jordandeklerk](https://github.com/jordandeklerk) in [#242](https://github.com/jordandeklerk/moderndid/pull/242)
* REL: prepare release v0.2.0 by [@jordandeklerk](https://github.com/jordandeklerk) in [#243](https://github.com/jordandeklerk/moderndid/pull/243)


**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.1.1...v0.2.0

[Changes][v0.2.0]


<a id="v0.1.1"></a>
# [v0.1.1](https://github.com/jordandeklerk/moderndid/releases/tag/v0.1.1) - 2026-02-26

## What's Changed
* Update changelog for v0.1.0 by [@github-actions](https://github.com/github-actions)[bot] in [#203](https://github.com/jordandeklerk/moderndid/pull/203)
* REF: remove unused parameter by [@jordandeklerk](https://github.com/jordandeklerk) in [#204](https://github.com/jordandeklerk/moderndid/pull/204)
* DOC: update installation and overview docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#205](https://github.com/jordandeklerk/moderndid/pull/205)
* BUG: remove imports and exports from Dask and Spark backends by [@jordandeklerk](https://github.com/jordandeklerk) in [#207](https://github.com/jordandeklerk/moderndid/pull/207)
* DEV: introduce pixi dev tools by [@jordandeklerk](https://github.com/jordandeklerk) in [#208](https://github.com/jordandeklerk/moderndid/pull/208)
* DOC: simplify plotting examples by [@jordandeklerk](https://github.com/jordandeklerk) in [#209](https://github.com/jordandeklerk/moderndid/pull/209)
* DOC: fix prose and links in README by [@jordandeklerk](https://github.com/jordandeklerk) in [#210](https://github.com/jordandeklerk/moderndid/pull/210)
* DOC: enhance documentation by [@jordandeklerk](https://github.com/jordandeklerk) in [#211](https://github.com/jordandeklerk/moderndid/pull/211)
* DOC: update usage docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#212](https://github.com/jordandeklerk/moderndid/pull/212)
* REL: bump version to 0.1.1 and update release notes by [@jordandeklerk](https://github.com/jordandeklerk) in [#213](https://github.com/jordandeklerk/moderndid/pull/213)


**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.1.0...v0.1.1

[Changes][v0.1.1]


<a id="v0.1.0"></a>
# [v0.1.0](https://github.com/jordandeklerk/moderndid/releases/tag/v0.1.0) - 2026-02-25

## What's Changed
* Overhaul documentation style and add release notes by [@jordandeklerk](https://github.com/jordandeklerk) in [#140](https://github.com/jordandeklerk/moderndid/pull/140)
* ruff: fix linting imports by [@jordandeklerk](https://github.com/jordandeklerk) in [#141](https://github.com/jordandeklerk/moderndid/pull/141)
* Remove redundant section in getting started section  by [@jordandeklerk](https://github.com/jordandeklerk) in [#142](https://github.com/jordandeklerk/moderndid/pull/142)
* DOC: add version import to init by [@jordandeklerk](https://github.com/jordandeklerk) in [#143](https://github.com/jordandeklerk/moderndid/pull/143)
* Add dataframe-agnostic functionality via narwhals by [@jordandeklerk](https://github.com/jordandeklerk) in [#144](https://github.com/jordandeklerk/moderndid/pull/144)
* DOC: remove trailing commas from output by [@jordandeklerk](https://github.com/jordandeklerk) in [#145](https://github.com/jordandeklerk/moderndid/pull/145)
* FEAT: Add `didinter` module for intertemporal DiD estimation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#146](https://github.com/jordandeklerk/moderndid/pull/146)
* FEAT: add plotting and bootstrapping for intertemporal DiD by [@jordandeklerk](https://github.com/jordandeklerk) in [#147](https://github.com/jordandeklerk/moderndid/pull/147)
* DOC: update main README by [@jordandeklerk](https://github.com/jordandeklerk) in [#148](https://github.com/jordandeklerk/moderndid/pull/148)
* build(deps): bump github/codeql-action from 4.31.9 to 4.32.0 by [@dependabot](https://github.com/dependabot)[bot] in [#149](https://github.com/jordandeklerk/moderndid/pull/149)
* Configure Renovate by [@renovate](https://github.com/renovate)[bot] in [#151](https://github.com/jordandeklerk/moderndid/pull/151)
* REF: standardize bootstrap args and consolidate linting on ruff by [@jordandeklerk](https://github.com/jordandeklerk) in [#150](https://github.com/jordandeklerk/moderndid/pull/150)
* Update williambdean/closing-labels action to v0.0.6 by [@renovate](https://github.com/renovate)[bot] in [#153](https://github.com/jordandeklerk/moderndid/pull/153)
* Update actions/checkout action to v6.0.2 by [@renovate](https://github.com/renovate)[bot] in [#152](https://github.com/jordandeklerk/moderndid/pull/152)
* Update dependency python to 3.14 by [@renovate](https://github.com/renovate)[bot] in [#155](https://github.com/jordandeklerk/moderndid/pull/155)
* Update dependency scipy to >=1.17,<1.18 by [@renovate](https://github.com/renovate)[bot] in [#156](https://github.com/jordandeklerk/moderndid/pull/156)
* REF: refactor numba integration and update influence function aggregation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#158](https://github.com/jordandeklerk/moderndid/pull/158)
* FEAT: allow optional dependencies for installation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#159](https://github.com/jordandeklerk/moderndid/pull/159)
* BEN: add benchmarks for core estimators against R packages by [@jordandeklerk](https://github.com/jordandeklerk) in [#160](https://github.com/jordandeklerk/moderndid/pull/160)
* FEAT: use minimal theme for plotting by [@jordandeklerk](https://github.com/jordandeklerk) in [#161](https://github.com/jordandeklerk/moderndid/pull/161)
* DOC: add continuous DiD background section by [@jordandeklerk](https://github.com/jordandeklerk) in [#162](https://github.com/jordandeklerk/moderndid/pull/162)
* Update GitHub Actions by [@renovate](https://github.com/renovate)[bot] in [#163](https://github.com/jordandeklerk/moderndid/pull/163)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.0 by [@renovate](https://github.com/renovate)[bot] in [#164](https://github.com/jordandeklerk/moderndid/pull/164)
* FEAT: add new benchmark commands by [@jordandeklerk](https://github.com/jordandeklerk) in [#165](https://github.com/jordandeklerk/moderndid/pull/165)
* CHORE: disable dependency dashboard by [@jordandeklerk](https://github.com/jordandeklerk) in [#166](https://github.com/jordandeklerk/moderndid/pull/166)
* DOC: add new exmaples for user guide and background sections by [@jordandeklerk](https://github.com/jordandeklerk) in [#167](https://github.com/jordandeklerk/moderndid/pull/167)
* DOC: refactor documentation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#168](https://github.com/jordandeklerk/moderndid/pull/168)
* FEAT: implement DDD preprocessing pipeline by [@jordandeklerk](https://github.com/jordandeklerk) in [#169](https://github.com/jordandeklerk/moderndid/pull/169)
* REF: add prettytable output by [@jordandeklerk](https://github.com/jordandeklerk) in [#170](https://github.com/jordandeklerk/moderndid/pull/170)
* DOC: update architecture docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#171](https://github.com/jordandeklerk/moderndid/pull/171)
* ENH: add thread-based parallelism for group-time estimations by [@jordandeklerk](https://github.com/jordandeklerk) in [#172](https://github.com/jordandeklerk/moderndid/pull/172)
* ENH: add validation checks for various parameters by [@jordandeklerk](https://github.com/jordandeklerk) in [#173](https://github.com/jordandeklerk/moderndid/pull/173)
* FEAT: add CuPY GPU acceleration for regression and propensity estimation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#175](https://github.com/jordandeklerk/moderndid/pull/175)
* Update github/codeql-action action to v4.32.2 by [@renovate](https://github.com/renovate)[bot] in [#174](https://github.com/jordandeklerk/moderndid/pull/174)
* DOC: update README for GPU instructions by [@jordandeklerk](https://github.com/jordandeklerk) in [#176](https://github.com/jordandeklerk/moderndid/pull/176)
* REF: update validation logic and error handling by [@jordandeklerk](https://github.com/jordandeklerk) in [#177](https://github.com/jordandeklerk/moderndid/pull/177)
* ENH: improve several computations in `didhonest` module by [@jordandeklerk](https://github.com/jordandeklerk) in [#178](https://github.com/jordandeklerk/moderndid/pull/178)
* ENH: refine `npiv` docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#179](https://github.com/jordandeklerk/moderndid/pull/179)
* ENH: add validation tox testing env by [@jordandeklerk](https://github.com/jordandeklerk) in [#180](https://github.com/jordandeklerk/moderndid/pull/180)
* DOC: update README with hardware specifications by [@jordandeklerk](https://github.com/jordandeklerk) in [#181](https://github.com/jordandeklerk/moderndid/pull/181)
* FEAT: add panel data utilities by [@jordandeklerk](https://github.com/jordandeklerk) in [#182](https://github.com/jordandeklerk/moderndid/pull/182)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.1 by [@renovate](https://github.com/renovate)[bot] in [#184](https://github.com/jordandeklerk/moderndid/pull/184)
* Update github/codeql-action action to v4.32.3 by [@renovate](https://github.com/renovate)[bot] in [#183](https://github.com/jordandeklerk/moderndid/pull/183)
* FEAT: add distributed computing via Dask and multi-gpu support by [@jordandeklerk](https://github.com/jordandeklerk) in [#185](https://github.com/jordandeklerk/moderndid/pull/185)
* DOC: fix readme  by [@jordandeklerk](https://github.com/jordandeklerk) in [#186](https://github.com/jordandeklerk/moderndid/pull/186)
* TEST: fix Dask test suite and update CI by [@jordandeklerk](https://github.com/jordandeklerk) in [#187](https://github.com/jordandeklerk/moderndid/pull/187)
* REF: refresh benchmarks against R packages by [@jordandeklerk](https://github.com/jordandeklerk) in [#188](https://github.com/jordandeklerk/moderndid/pull/188)
* FEAT: add spark distributed backend by [@jordandeklerk](https://github.com/jordandeklerk) in [#191](https://github.com/jordandeklerk/moderndid/pull/191)
* DOC: update images by [@jordandeklerk](https://github.com/jordandeklerk) in [#193](https://github.com/jordandeklerk/moderndid/pull/193)
* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.2 by [@renovate](https://github.com/renovate)[bot] in [#190](https://github.com/jordandeklerk/moderndid/pull/190)
* Update actions/setup-java action to v5 by [@renovate](https://github.com/renovate)[bot] in [#192](https://github.com/jordandeklerk/moderndid/pull/192)
* Update github/codeql-action action to v4.32.4 by [@renovate](https://github.com/renovate)[bot] in [#189](https://github.com/jordandeklerk/moderndid/pull/189)
* REF: enhance R validation tests and fix bug in `compute_conditional_cs_rmb()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#194](https://github.com/jordandeklerk/moderndid/pull/194)
* FEAT: add first pass at distributed `did_multiplegt()` and `cont_did()` by [@jordandeklerk](https://github.com/jordandeklerk) in [#195](https://github.com/jordandeklerk/moderndid/pull/195)
* DOC: add plotting guide by [@jordandeklerk](https://github.com/jordandeklerk) in [#196](https://github.com/jordandeklerk/moderndid/pull/196)
* TEST: improve test coverage by [@jordandeklerk](https://github.com/jordandeklerk) in [#197](https://github.com/jordandeklerk/moderndid/pull/197)
* TEST: add more tests for coverage by [@jordandeklerk](https://github.com/jordandeklerk) in [#199](https://github.com/jordandeklerk/moderndid/pull/199)
* CI: stop ignoring distributed tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#200](https://github.com/jordandeklerk/moderndid/pull/200)
* DOC: refactor README's and fix math rendering by [@jordandeklerk](https://github.com/jordandeklerk) in [#201](https://github.com/jordandeklerk/moderndid/pull/201)
* REL: prepare 0.1.0 release by [@jordandeklerk](https://github.com/jordandeklerk) in [#202](https://github.com/jordandeklerk/moderndid/pull/202)

## New Contributors
* [@renovate](https://github.com/renovate)[bot] made their first contribution in [#151](https://github.com/jordandeklerk/moderndid/pull/151)

**Full Changelog**: https://github.com/jordandeklerk/moderndid/commits/v0.1.0

[Changes][v0.1.0]


<a id="v0.0.3"></a>
# [v0.0.3](https://github.com/jordandeklerk/moderndid/releases/tag/v0.0.3) - 2026-01-21

## What's Changed
* Bump version to 0.0.2 by [@jordandeklerk](https://github.com/jordandeklerk) in [#116](https://github.com/jordandeklerk/moderndid/pull/116)
* Add permissions for post-release workflow by [@jordandeklerk](https://github.com/jordandeklerk) in [#117](https://github.com/jordandeklerk/moderndid/pull/117)
* Update changelog for v0.0.2 by [@github-actions](https://github.com/github-actions)[bot] in [#118](https://github.com/jordandeklerk/moderndid/pull/118)

## New Contributors
* [@github-actions](https://github.com/github-actions)[bot] made their first contribution in [#118](https://github.com/jordandeklerk/moderndid/pull/118)

**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.0.2...v0.0.3

[Changes][v0.0.3]


<a id="v0.0.2"></a>
# [v0.0.2](https://github.com/jordandeklerk/moderndid/releases/tag/v0.0.2) - 2026-01-21

## What's Changed
* Add new logos by [@jordandeklerk](https://github.com/jordandeklerk) in [#25](https://github.com/jordandeklerk/moderndid/pull/25)
* Fix readthedocs error by [@jordandeklerk](https://github.com/jordandeklerk) in [#26](https://github.com/jordandeklerk/moderndid/pull/26)
* Fix layout for headers by [@jordandeklerk](https://github.com/jordandeklerk) in [#27](https://github.com/jordandeklerk/moderndid/pull/27)
* Update index and API documentation by [@jordandeklerk](https://github.com/jordandeklerk) in [#28](https://github.com/jordandeklerk/moderndid/pull/28)
* API doc fix for build by [@jordandeklerk](https://github.com/jordandeklerk) in [#29](https://github.com/jordandeklerk/moderndid/pull/29)
* Nonparametric IV Functions for Sieve Dimension and UCB Estimation  by [@jordandeklerk](https://github.com/jordandeklerk) in [#32](https://github.com/jordandeklerk/moderndid/pull/32)
* Fix references in NPIV functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#33](https://github.com/jordandeklerk/moderndid/pull/33)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#31](https://github.com/jordandeklerk/moderndid/pull/31)
* chore(deps): bump github/codeql-action from 3.29.4 to 3.29.8 by [@dependabot](https://github.com/dependabot)[bot] in [#35](https://github.com/jordandeklerk/moderndid/pull/35)
* chore(deps): bump actions/download-artifact from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#34](https://github.com/jordandeklerk/moderndid/pull/34)
* Add Splines and Panel Processing Functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#36](https://github.com/jordandeklerk/moderndid/pull/36)
* Add more processing functions for continuous DiD by [@jordandeklerk](https://github.com/jordandeklerk) in [#37](https://github.com/jordandeklerk/moderndid/pull/37)
* Add main continuous DiD function and preprocessing functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#41](https://github.com/jordandeklerk/moderndid/pull/41)
* chore(deps): bump actions/checkout from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#40](https://github.com/jordandeklerk/moderndid/pull/40)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#39](https://github.com/jordandeklerk/moderndid/pull/39)
* chore(deps): bump github/codeql-action from 3.29.8 to 3.29.10 by [@dependabot](https://github.com/dependabot)[bot] in [#38](https://github.com/jordandeklerk/moderndid/pull/38)
* Add new docs by [@jordandeklerk](https://github.com/jordandeklerk) in [#42](https://github.com/jordandeklerk/moderndid/pull/42)
* chore(deps): bump github/codeql-action from 3.29.10 to 3.29.11 by [@dependabot](https://github.com/dependabot)[bot] in [#44](https://github.com/jordandeklerk/moderndid/pull/44)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#43](https://github.com/jordandeklerk/moderndid/pull/43)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#45](https://github.com/jordandeklerk/moderndid/pull/45)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#48](https://github.com/jordandeklerk/moderndid/pull/48)
* chore(deps): bump github/codeql-action from 3.29.11 to 3.30.1 by [@dependabot](https://github.com/dependabot)[bot] in [#47](https://github.com/jordandeklerk/moderndid/pull/47)
* chore(deps): bump actions/setup-python from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#46](https://github.com/jordandeklerk/moderndid/pull/46)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#50](https://github.com/jordandeklerk/moderndid/pull/50)
* chore(deps): bump github/codeql-action from 3.30.1 to 3.30.3 by [@dependabot](https://github.com/dependabot)[bot] in [#49](https://github.com/jordandeklerk/moderndid/pull/49)
* Add `cont_did()` plotting functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#51](https://github.com/jordandeklerk/moderndid/pull/51)
* Add docs for `didcont` by [@jordandeklerk](https://github.com/jordandeklerk) in [#52](https://github.com/jordandeklerk/moderndid/pull/52)
* Update API links by [@jordandeklerk](https://github.com/jordandeklerk) in [#53](https://github.com/jordandeklerk/moderndid/pull/53)
* Optimize full test suite for speed by [@jordandeklerk](https://github.com/jordandeklerk) in [#56](https://github.com/jordandeklerk/moderndid/pull/56)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#58](https://github.com/jordandeklerk/moderndid/pull/58)
* Update analytical standard errors for estimators by [@jordandeklerk](https://github.com/jordandeklerk) in [#62](https://github.com/jordandeklerk/moderndid/pull/62)
* chore(deps): bump github/codeql-action from 3.30.3 to 4.31.0 by [@dependabot](https://github.com/dependabot)[bot] in [#69](https://github.com/jordandeklerk/moderndid/pull/69)
* chore(deps): bump actions/download-artifact from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#67](https://github.com/jordandeklerk/moderndid/pull/67)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#70](https://github.com/jordandeklerk/moderndid/pull/70)
* Fix propensity trimming in estimators by [@jordandeklerk](https://github.com/jordandeklerk) in [#73](https://github.com/jordandeklerk/moderndid/pull/73)
* Refactor tox testing environments by [@jordandeklerk](https://github.com/jordandeklerk) in [#74](https://github.com/jordandeklerk/moderndid/pull/74)
* chore(deps): bump github/codeql-action from 4.31.0 to 4.31.2 by [@dependabot](https://github.com/dependabot)[bot] in [#75](https://github.com/jordandeklerk/moderndid/pull/75)
* Unify preprocessing logic for all estimators by [@jordandeklerk](https://github.com/jordandeklerk) in [#78](https://github.com/jordandeklerk/moderndid/pull/78)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#76](https://github.com/jordandeklerk/moderndid/pull/76)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#79](https://github.com/jordandeklerk/moderndid/pull/79)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#80](https://github.com/jordandeklerk/moderndid/pull/80)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#82](https://github.com/jordandeklerk/moderndid/pull/82)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#85](https://github.com/jordandeklerk/moderndid/pull/85)
* chore(deps): bump github/codeql-action from 4.31.2 to 4.31.6 by [@dependabot](https://github.com/dependabot)[bot] in [#84](https://github.com/jordandeklerk/moderndid/pull/84)
* chore(deps): bump actions/checkout from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#83](https://github.com/jordandeklerk/moderndid/pull/83)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#86](https://github.com/jordandeklerk/moderndid/pull/86)
* Add baterries-included plotting via `PlotCollection` object  by [@jordandeklerk](https://github.com/jordandeklerk) in [#87](https://github.com/jordandeklerk/moderndid/pull/87)
* Fix package dependencies by [@jordandeklerk](https://github.com/jordandeklerk) in [#88](https://github.com/jordandeklerk/moderndid/pull/88)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#89](https://github.com/jordandeklerk/moderndid/pull/89)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#90](https://github.com/jordandeklerk/moderndid/pull/90)
* chore(deps): bump github/codeql-action from 4.31.6 to 4.31.9 by [@dependabot](https://github.com/dependabot)[bot] in [#92](https://github.com/jordandeklerk/moderndid/pull/92)
* chore(deps): bump actions/download-artifact from 6 to 7 by [@dependabot](https://github.com/dependabot)[bot] in [#93](https://github.com/jordandeklerk/moderndid/pull/93)
* Add continuous DiD preprocessing functions by [@jordandeklerk](https://github.com/jordandeklerk) in [#94](https://github.com/jordandeklerk/moderndid/pull/94)
* Add `random_state` for bootstrap reproducibility by [@jordandeklerk](https://github.com/jordandeklerk) in [#96](https://github.com/jordandeklerk/moderndid/pull/96)
* Add 2-period triple DiD preprocessing logic by [@jordandeklerk](https://github.com/jordandeklerk) in [#97](https://github.com/jordandeklerk/moderndid/pull/97)
* Add nuisance function estimation for DDD by [@jordandeklerk](https://github.com/jordandeklerk) in [#98](https://github.com/jordandeklerk/moderndid/pull/98)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#99](https://github.com/jordandeklerk/moderndid/pull/99)
* Add 2 period panel DDD estimator by [@jordandeklerk](https://github.com/jordandeklerk) in [#100](https://github.com/jordandeklerk/moderndid/pull/100)
* Add multi-period DDD estimator by [@jordandeklerk](https://github.com/jordandeklerk) in [#101](https://github.com/jordandeklerk/moderndid/pull/101)
* Add aggregated treatment effects for triple DiD by [@jordandeklerk](https://github.com/jordandeklerk) in [#102](https://github.com/jordandeklerk/moderndid/pull/102)
* Add unified DDD estimator and numba optimizations by [@jordandeklerk](https://github.com/jordandeklerk) in [#103](https://github.com/jordandeklerk/moderndid/pull/103)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#104](https://github.com/jordandeklerk/moderndid/pull/104)
* Add multi-period repeated cross section DDD functionality  by [@jordandeklerk](https://github.com/jordandeklerk) in [#105](https://github.com/jordandeklerk/moderndid/pull/105)
* Update test instructions in contributing guide by [@jordandeklerk](https://github.com/jordandeklerk) in [#106](https://github.com/jordandeklerk/moderndid/pull/106)
* Add the `didtriple` readme by [@jordandeklerk](https://github.com/jordandeklerk) in [#108](https://github.com/jordandeklerk/moderndid/pull/108)
* Add tighter tolerances for point estimates in R tests by [@jordandeklerk](https://github.com/jordandeklerk) in [#109](https://github.com/jordandeklerk/moderndid/pull/109)
* Update testing instructions for Tox usage by [@jordandeklerk](https://github.com/jordandeklerk) in [#110](https://github.com/jordandeklerk/moderndid/pull/110)
* Migrate codebase from pandas to polars  by [@jordandeklerk](https://github.com/jordandeklerk) in [#111](https://github.com/jordandeklerk/moderndid/pull/111)
* Integrate plotnine for all plotting by [@jordandeklerk](https://github.com/jordandeklerk) in [#113](https://github.com/jordandeklerk/moderndid/pull/113)
* Unify main function arguments by [@jordandeklerk](https://github.com/jordandeklerk) in [#114](https://github.com/jordandeklerk/moderndid/pull/114)


**Full Changelog**: https://github.com/jordandeklerk/moderndid/compare/v0.0.1...v0.0.2

[Changes][v0.0.2]


<a id="v0.0.1"></a>
# [v0.0.1](https://github.com/jordandeklerk/moderndid/releases/tag/v0.0.1) - 2025-08-02

## What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#3](https://github.com/jordandeklerk/moderndid/pull/3)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#4](https://github.com/jordandeklerk/moderndid/pull/4)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#5](https://github.com/jordandeklerk/moderndid/pull/5)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#6](https://github.com/jordandeklerk/moderndid/pull/6)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#8](https://github.com/jordandeklerk/moderndid/pull/8)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#9](https://github.com/jordandeklerk/moderndid/pull/9)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#11](https://github.com/jordandeklerk/moderndid/pull/11)
* chore(deps): bump github/codeql-action from 3.29.0 to 3.29.1 by [@dependabot](https://github.com/dependabot)[bot] in [#10](https://github.com/jordandeklerk/moderndid/pull/10)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#13](https://github.com/jordandeklerk/moderndid/pull/13)
* chore(deps): bump github/codeql-action from 3.29.1 to 3.29.2 by [@dependabot](https://github.com/dependabot)[bot] in [#12](https://github.com/jordandeklerk/moderndid/pull/12)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#14](https://github.com/jordandeklerk/moderndid/pull/14)
* chore(deps): bump github/codeql-action from 3.29.2 to 3.29.3 by [@dependabot](https://github.com/dependabot)[bot] in [#15](https://github.com/jordandeklerk/moderndid/pull/15)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#16](https://github.com/jordandeklerk/moderndid/pull/16)
* Debug Grid Bounds by [@jordandeklerk](https://github.com/jordandeklerk) in [#17](https://github.com/jordandeklerk/moderndid/pull/17)
* Fix test issues by [@jordandeklerk](https://github.com/jordandeklerk) in [#18](https://github.com/jordandeklerk/moderndid/pull/18)
* Add documentation for implemented modules by [@jordandeklerk](https://github.com/jordandeklerk) in [#21](https://github.com/jordandeklerk/moderndid/pull/21)
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] in [#20](https://github.com/jordandeklerk/moderndid/pull/20)
* chore(deps): bump github/codeql-action from 3.29.3 to 3.29.4 by [@dependabot](https://github.com/dependabot)[bot] in [#19](https://github.com/jordandeklerk/moderndid/pull/19)
* Add no commit to main  by [@jordandeklerk](https://github.com/jordandeklerk) in [#22](https://github.com/jordandeklerk/moderndid/pull/22)
* Fix default grid bounds  by [@jordandeklerk](https://github.com/jordandeklerk) in [#23](https://github.com/jordandeklerk/moderndid/pull/23)
* Change package name by [@jordandeklerk](https://github.com/jordandeklerk) in [#24](https://github.com/jordandeklerk/moderndid/pull/24)

## New Contributors
* [@pre-commit-ci](https://github.com/pre-commit-ci)[bot] made their first contribution in [#3](https://github.com/jordandeklerk/moderndid/pull/3)
* [@dependabot](https://github.com/dependabot)[bot] made their first contribution in [#10](https://github.com/jordandeklerk/moderndid/pull/10)
* [@jordandeklerk](https://github.com/jordandeklerk) made their first contribution in [#17](https://github.com/jordandeklerk/moderndid/pull/17)

**Full Changelog**: https://github.com/jordandeklerk/moderndid/commits/v0.0.1

[Changes][v0.0.1]


[v0.4.0]: https://github.com/jordandeklerk/moderndid/compare/v0.3.1...v0.4.0
[v0.3.1]: https://github.com/jordandeklerk/moderndid/compare/v0.3.0...v0.3.1
[v0.3.0]: https://github.com/jordandeklerk/moderndid/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/jordandeklerk/moderndid/compare/v0.1.1...v0.2.0
[v0.1.1]: https://github.com/jordandeklerk/moderndid/compare/v0.1.0...v0.1.1
[v0.1.0]: https://github.com/jordandeklerk/moderndid/compare/v0.0.3...v0.1.0
[v0.0.3]: https://github.com/jordandeklerk/moderndid/compare/v0.0.2...v0.0.3
[v0.0.2]: https://github.com/jordandeklerk/moderndid/compare/v0.0.1...v0.0.2
[v0.0.1]: https://github.com/jordandeklerk/moderndid/tree/v0.0.1

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
