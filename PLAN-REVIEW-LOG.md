# Plan Review Log — GMM-Based Fusion Router

Append-only transcript of the claudex-loop plan review.

## Setup (2026-10-29)

- **Entry point:** `/claudex-loop:codex-review` → `claudex-loop mode=review`
- **Roles:** host/planner = Claude Code (Opus 5.5); plan reviewer = Codex CLI 0.163.0; builder = Claude (not authorized yet); inspector = Codex (fresh session).
- **Model request:** none explicit; Codex uses its CLI configuration (`~/.codex/config.toml`: `gpt-6-astra`, effort `high`). Observed model is reported from result records when available.
- **PLAN_FILE:** `docs/spec-gmm-router.md` (no PLAN.md exists; the spec written earlier in this conversation is the plan).
- **Supporting context for reviewer:** `docs/konteks-notebook.md`, `docs/research-charter.md`, `docs/eksperimen/EXP-00{1,2}-*.md`, `AGENTS.md`, `adaptive_fusion new.ipynb`, `appliances_fusion_persample (2).csv`, `src/`, `tests/`.
- **Scope / authorization:** review only. Implementation NOT authorized by this invocation.
- **MAX_ROUNDS:** 5.
- **Artifacts:** outside checkout, in the session scratchpad (`.../scratchpad/claudex/`).
- **Safety:** Codex had gmail, google-drive, and github connector plugins enabled (write-capable). With user approval, `~/.codex/config.toml` was backed up and the features `apps`, `plugins`, `remote_plugin`, and `tool_suggest` were disabled for the duration of the review. A probe confirmed no connector tools were exposed. The original config will be restored from the backup (SHA256 `30075e0d…96f9`) afterwards.

## Rounds

### Round 1 — Codex — verdict **REVISE**

- Result: `/tmp/claude-1000/-home-ds-adaptive-fusion/62ea5f50-c598-4d89-8fed-0924aa304782/scratchpad/claudex/round1/claudex-tnwdxiu8/result.json`
- Plan SHA256 reviewed: `260979708cf2b8b2b39831011d20f6bed9c6ba2f274b08c0a6fb811c639fb1d5`
- Session: `01a1853d-ae8b-7663-849c-73776c32fd38` · CLI codex-cli 0.163.0 · observed model: unknown (not returned) · 130.79s
- Summary: The plan hash matches. Five defects need resolution before implementation: premature test evaluation, unspecified OOF checkpoint isolation, colliding artifact paths, mismatched index spaces, and an invalid shuffled-label acceptance criterion.

#### Findings (full)

- **GMM-001** (high) `docs/spec-gmm-router.md:128`
  - Evidence: T3 evaluates test performance and uses agreement with notebook test accuracy as a sanity criterion; T4 then creates test-derived router labels before T6 selects router hyperparameters. This contradicts §7's requirement to evaluate test only after everything is frozen. Fixing implementations until they meet the T3 test tolerance would make test performance part of development selection. EXP-001 repeats this acceptance criterion.
  - Fix: Restrict T3 and T4 to train/validation outputs. Freeze checkpoints and selected router configurations before T7, then compute all test metrics and test router labels together. Establish implementation parity using synthetic fixtures or train/validation predictions rather than a test-accuracy acceptance threshold.
- **GMM-002** (high) `docs/spec-gmm-router.md:70`
  - Evidence: Stage A specifies training on four folds and selecting checkpoints by validation loss, but never defines that validation set. Using the fifth fold to select the checkpoint exposes the very labels used to produce supposedly unseen OOF predictions; using the global validation set contradicts D1's train-only requirement. The T4 index-coverage test would detect neither violation.
  - Fix: Define checkpoint selection inside each outer training partition, using an inner validation split or a training schedule selected entirely within that partition. Explicitly exclude the outer held-out fold and global validation/test sets from checkpoint selection, and require provenance checks covering both optimizer and checkpoint-selection indices.
- **GMM-003** (medium) `docs/spec-gmm-router.md:64`
  - Evidence: The plan supports developing on the zip dataset and later switching to the large dataset, but both write data/processed/split.json. T4 confidences and T3/T7 results are distinguished only by seed/method. A seed-42 large-data run can therefore overwrite the zip split while leaving zip confidences or other seed results behind; downstream selection or summary generation has no specified dataset/split identity check. The proposed result schema contains no such identity.
  - Fix: Namespace splits, confidences, checkpoints, selected configurations, and results by dataset/run identity. Persist source fingerprints, ordered sample IDs, split identity, configuration identity, and checkpoint references; require consumers and summary writers to reject incompatible artifacts.
- **GMM-004** (medium) `docs/spec-gmm-router.md:126`
  - Evidence: T1 requires test indices to match the notebook CSV's subset_idx, but §4 creates indices into the filtered 484-row dataset. Notebook cell 57 writes subset_idx as np.where(valid)[0][idx_test], which refers to original CSV rows. The archived CSV begins with subset_idx=885 and contains 993, both outside the filtered index range 0–483. Direct comparison fails; treating those IDs as filtered indices also fails.
  - Fix: Specify both index spaces explicitly. Preserve the ordered valid-to-source-row mapping, use filtered indices for tensor access, and compare mapped source IDs against subset_idx in test_pos order. Persist this mapping with the split.
- **GMM-005** (medium) `docs/spec-gmm-router.md:131`
  - Evidence: T6 requires shuffled router labels to yield approximately majority-class router accuracy, while §9 requires Dynamic accuracy to approximate the majority router. Neither follows from shuffling. An independent predictor with method frequencies q has expected router accuracy sum(p_j*q_j), not max(p_j); class weighting explicitly allowed in §5 can further change q. Its Dynamic accuracy can also differ from the constant majority method because the three fusion models have different predictions. These checks can reject a correct implementation.
  - Fix: Define a permutation-control protocol with repeated shuffles, fixed validation evaluation, and uncertainty bounds. Compare against the appropriate independent-routing expectation or empirical permutation distribution. Keep majority routing as a separate baseline, and reserve exact constant-method behavior for the K=1 check.

#### Coverage
- Verified the on-disk plan SHA256 against the supplied hash.
- Read the research charter, notebook context, phase procedures, experiment specifications, and related workflow documents as evidence.
- Inspected all existing src Python files, fusion registry callers, contract tests, seed helper, dependencies, and Makefile.
- Traced notebook filtering, split construction, training/checkpoint selection, checkpoint writing/loading, inference ordering, and per-sample CSV writing.
- Inspected archive inventory and CSV index/schema evidence; traced proposed artifact producers and consumers across T1–T7 and experiment summaries.

#### Limitations
- Static review only: no files edited, tests run, notebook cells executed, or work delegated.
- The router, loader, training pipeline, configurations, and result artifacts are not yet implemented; their runtime correctness could not be assessed.
- Large-run embeddings, source CSV, and checkpoints are absent. Large-run alignment and sampling could not be verified.
- Archived embedding contents and checkpoint binaries were not loaded; dataset statistics and numerical reproducibility were not independently validated.

#### Host dispositions (round 1)
All five findings accepted and applied. Please re-review the revised plan; do not relitigate resolved points without new evidence.

- GMM-001 (high) ACCEPTED. T3 now trains Full models and records validation metrics only, with no test evaluation. Notebook parity uses synthetic fixtures (T2) and a non-gating comparison with the notebook's validation-loss log (T3). All test metrics, test posteriors, and test router labels are computed together in T7/Tahap E, after A–D are frozen. §7 adds a "no test metric before T7" item. EXP-001 now says notebook test numbers are a post-hoc reference only.
- GMM-002 (high) ACCEPTED. Tahap A now uses an inner stratified 80/20 split of each outer fold's training part (inner-train/inner-val). Checkpoints are selected on inner-val. The held-out fold, global val, and test are excluded. Per-fold provenance (optimizer/selection/predict indices) is persisted and tested for disjointness in T4. §7 updated.
- GMM-003 (medium) ACCEPTED. New §4a: every artifact is namespaced by dataset_id (data/processed, results, checkpoints) and records the source fingerprint, split_sha256, config_sha256, and seed. Consumers and the summary script reject mismatches. The result schema includes the identity fields.
- GMM-004 (medium) ACCEPTED. §4 now defines the filtered vs source index spaces. split.json persists the valid→source_row mapping. The T1 test compares source_row[idx_test] (test_pos order) to the CSV subset_idx as an identity check only.
- GMM-005 (medium) ACCEPTED. The shuffled-label criterion was replaced by a permutation-control protocol (R repeated shuffles, fixed val, empirical distribution, expectation Σ p_j q_j, 95th-percentile diagnostic, not an automated pass/fail). Majority router stays a separate baseline. K=1 → exact constant-method check retained. §9 updated.

### Round 2 — Codex — verdict **REVISE**

- Result: `/tmp/claude-1000/-home-ds-adaptive-fusion/62ea5f50-c598-4d89-8fed-0924aa304782/scratchpad/claudex/round2/claudex-ab6aqjmb/result.json`
- Plan SHA256 reviewed: `c188842fa988c6a233598f410d6484c7015b78fb9aaf355c9a6e5d2ac253291e`
- Session: `01a1853d-ae8b-7663-849c-73776c32fd38` (resumed) · observed model: unknown (not returned) · 94.14s
- Summary: The plan hash matches. The test-timing, OOF isolation, and index-space findings are resolved. Cross-dataset isolation is fixed, but artifact compatibility remains incomplete. The replacement permutation diagnostic introduces selection bias.

#### Findings (full)

- **GMM-003** (medium) `docs/spec-gmm-router.md:80`
  - Evidence: The new compatibility rule requires consumers to reject dataset/source/split mismatches, but does not require configuration or checkpoint compatibility. T3 writes checkpoints to paths distinguished only by dataset, method, and seed; T4 caches their validation confidences separately. Rerunning T3 with a changed training configuration can overwrite a checkpoint while leaving old T4 confidences and T6 selection results. All explicitly required compatibility checks still pass. T7 can then evaluate different Full models from those used for validation selection. The result schema records mutable checkpoint paths, not checkpoint hashes.
  - Fix: Bind confidence caches and selection artifacts to the exact upstream checkpoint hashes and relevant training configuration identities. Require T4–T7 to validate those dependencies and reject stale artifacts. Use immutable run paths or invalidate dependent artifacts when checkpoints change. Summary compatibility should permit per-seed router selections while enforcing a common experimental protocol.
- **GMM-006** (medium) `docs/spec-gmm-router.md:167`
  - Evidence: T6 selects K/PCA/covariance/router hyperparameters on validation performance, then the new permutation procedure fixes that selected configuration and evaluates shuffled-label routers on the same validation set. The original score benefits from searching many configurations; the permutation distribution does not repeat that search. A configuration selected through chance validation alignment can therefore exceed this distribution's 95th percentile without learning a real signal. Calling the comparison diagnostic does not justify the explicit conclusion at line 171 that exceeding the threshold demonstrates learning.
  - Fix: Either repeat the same selection procedure for every permutation, evaluate the diagnostic on data excluded from configuration/checkpoint selection, or retain the fixed-configuration comparison as descriptive only and remove the 95th-percentile claim of learned signal.

#### Coverage
- Verified SHA256 c188842fa988c6a233598f410d6484c7015b78fb9aaf355c9a6e5d2ac253291e and checked all five prior findings against the revised plan.
- Confirmed GMM-001, GMM-002, and GMM-004 are resolved; the erroneous majority-accuracy expectation from GMM-005 is removed.
- Read the research charter, notebook context, revised EXP-001/EXP-002 documents, phase procedures, and related workflow references.
- Traced T3 checkpoint writers, T4 confidence consumers/writers, T6 selection, T7 evaluation, and summary compatibility requirements.
- Reinspected existing fusion implementations, registry callers, contract tests, seed handling, and notebook filtering, split, checkpoint, inference, and CSV-writing code.

#### Limitations
- Static review only; no files edited, tests run, notebook cells executed, or work delegated.
- The planned loader, router, training pipeline, and artifact consumers are not implemented, so runtime enforcement could not be verified.
- Large-run source data and checkpoints remain unavailable; their alignment and sampling were not verified.
- Embedding and checkpoint binaries were not loaded; numerical reproducibility was not assessed.

#### Host dispositions (round 2)
Both findings accepted and applied. Please re-review the revised plan.

- GMM-003 (medium) ACCEPTED. §4a now requires immutable per-stage run directories with run_id = sha256(dataset fingerprint, split_sha256, stage config_sha256, upstream run_id, seed). Changing a config or upstream yields a new run_id, and writing into an existing run_id with different content is refused. Every artifact carries a manifest with the SHA256 of each upstream checkpoint/artifact plus the training config identity. Consumers (T4–T7, summary) re-hash upstream files and reject mismatches. T7 must use Full checkpoints whose hashes equal those used by T4 (val confidences) and T6 (selection). The summary rejects mixed dataset_id or protocol (training config, selection grid, R) but allows per-seed router selections. Paths in T3/T4/T6/T7, the result schema (run_id, upstream run_ids, checkpoint path+sha256), the §7 checklist, and EXP-001/EXP-002 are updated.
- GMM-006 (medium) ACCEPTED (option 1: repeat selection). Each permutation now reruns the full selection procedure (entire K/PCA/covariance/MLP grid, same Q4 criterion on val). The best post-selection val score forms the null distribution, so selection bias is included. Label-free GMM posteriors are cached per grid config; only the MLP is retrained per permutation. The 95th-percentile comparison is now against this post-selection null. An empirical p-value (1+#{null≥obs})/(R+1) is reported, with an explicit note that R=20 gives p_min≈0.048 (weak evidence). §7 adds a checklist item.

### Round 3 — Codex — verdict **APPROVED**

- Result: `/tmp/claude-1000/-home-ds-adaptive-fusion/62ea5f50-c598-4d89-8fed-0924aa304782/scratchpad/claudex/round3/claudex-2ru4bqll/result.json`
- Plan SHA256 reviewed: `64af459e2e20656013d5b67378e72584c9ad25bb30e6afdc788016307020e2e7`
- Session: `01a1853d-ae8b-7663-849c-73776c32fd38` (resumed) · observed model: unknown (not returned) · 38.81s
- Summary: The plan hash matches. All prior findings, GMM-001 through GMM-006, are resolved at the specification level. No additional material defects were identified.
- Findings: none

#### Coverage
- Verified the revised plan against the supplied SHA256 and prior findings.
- Traced checkpoint, confidence, selection, evaluation, and summary dependencies; immutable runs and upstream hash verification address stale-artifact mixing.
- Confirmed each permutation repeats the full selection procedure using the same validation criterion.
- Checked research charter, phase procedures, EXP-001/EXP-002, notebook split/checkpoint/CSV references, existing fusion registry and implementations, contract tests, dependencies, and seed handling.

#### Limitations
- Static plan review only; no files edited, tests run, notebook cells executed, or work delegated.
- Approval does not verify implementation: the planned pipeline and artifact consumers are not yet present.
- The explicitly open research decisions remain subject to their documented task gates.
- Large-run source artifacts remain unavailable; embedding contents, checkpoint binaries, and numerical reproducibility were not validated.

## Outcome

- **APPROVED** in round 3 of 5, bound to `docs/spec-gmm-router.md` SHA256 `64af459e2e20656013d5b67378e72584c9ad25bb30e6afdc788016307020e2e7`.
- `runner.py check` result: "Approval matches the current plan." Any further edit to the spec invalidates this approval and needs re-review.
- Findings raised: 6 (2 high, 4 medium), all accepted and resolved. None rejected.
- Reviewer limitations carried forward: static review only; implementation not verified; open questions Q0–Q6 remain gated per task; large-run artifacts unavailable.
- Codex config restored from backup (SHA256 `30075e0d…96f9` matches the original). Features `apps`, `plugins`, `remote_plugin`, and `tool_suggest` are back to `true`.
- Implementation not started (not authorized by this invocation). When it is built, the final code needs a fresh Codex inspection session (Phase 3).

## Build phase

- **Authorization:** user said "build it" (2026-10-29). Builder = Claude (host). Base commit `6012ae1`. Pre-existing uncommitted docs from the planning phase are part of the inspected diff.
- **Decisions:** Q0–Q6 defaults accepted (`docs/keputusan-gmm-router.md`). Scope: code + tests + T1–T6 dev runs; T7 (test) only after inspection passes.
- **Internal review (project `reviewer` subagent):** PERLU PERBAIKAN (1 PENTING, 11 MINOR). Accepted and fixed #1–6 and #8–12. Rejected #7 (re-seeding inside the router loop): the stage already calls `seed_everything` and the result is equivalent.
- **Proof commands (host-run):** `make lint` clean; `make test` 63 passed.

### Inspection 1 — Codex (fresh session) — verdict **REVISE**

- Result: `/tmp/claude-1000/-home-ds-adaptive-fusion/62ea5f50-c598-4d89-8fed-0924aa304782/scratchpad/claudex/inspect1/claudex-b_j9e96g/result.json`
- Snapshot sha256: `f584806921fa9dffe0114875542d6fa660a27fec86e3bf8532d171155d84b4bb` · session `01a18560-2cf8-7971-ae8d-a89e9b20098f` · 319.97s
- Summary: The core train/validation isolation follows the plan, but concrete defects remain in aggregation, fusion edge cases, artifact concurrency, and report reproducibility.

- **REV-001** (medium) `src/adaptive_fusion/experiments/gmm_router.py:720`
  - Evidence: The summary aggregates each fixed fusion separately but omits the global method selected by validation for each seed. This misses the charter's principal comparator. The stored validation results illustrate the problem: Dynamic averages 0.9786 versus 0.9701 for the strongest fixed-method average, although Dynamic ties the validation-selected global model on every seed. McNemar entries preserve method names but do not supply this comparator's aggregate metrics.
  - Fix: Add a best_global_by_val summary row using each seed's already-selected method, with mean, standard deviation, and per-seed metrics. Also report paired Dynamic-minus-selected-global accuracy differences.
- **REV-002** (medium) `src/adaptive_fusion/fusion/cross_attention.py:31`
  - Evidence: The constructor accepts two sources. With image as query and only text as key/value, attention has one key, so its softmax weight is always 1. The output is independent of image and its gradient is zero, violating FusionModule's requirement that gradients reach every source. Existing contract fixtures use three sources and miss this accepted configuration.
  - Fix: Reject configurations with fewer than three sources for this notebook-compatible implementation, and add a two-source rejection test. Supporting two sources would require a separately approved architectural change.
- **REV-003** (medium) `src/adaptive_fusion/experiments/gmm_router.py:205`
  - Evidence: _run_stage checks for existing files without atomically claiming the run. Two simultaneous invocations can both pass the checks before produce() creates output files. Both then write the same checkpoints/results and overwrite the manifest through write_manifest(). Concurrent T7 invocations also both evaluate test, defeating the stated once-only behavior.
  - Fix: Acquire an exclusive per-run lock or atomic reservation before checking or producing artifacts. Publish completed outputs and the manifest atomically; competing invocations should wait for and verify the completed run or fail clearly.
- **REV-004** (medium) `src/adaptive_fusion/experiments/gmm_router.py:230`
  - Evidence: Report and summary readers discover existing runs by recomputing identities from the reader's current device, PyTorch version, and entire package source hash. The supplied full-run manifests match the CUDA identity; with the default device:auto on a CPU-only machine, the reader computes a different identity and reports that the run was never executed, even when all original artifacts are present. Editing only summary code likewise makes existing experiment runs undiscoverable.
  - Fix: Allow reporting consumers to select explicit immutable run manifests and verify their recorded producer identities, hashes, and protocol compatibility independently of the reporting machine. Preserve strict environment checks for stages that actually perform computation.
- **REV-005** (low) `pyproject.toml:11`
  - Evidence: scripts/laporan_sementara.py imports matplotlib and reportlab unconditionally, but neither dependency is declared in the project or dev extra. A clean installation following make setup cannot execute the documented report-generation command.
  - Fix: Declare a reporting extra containing matplotlib and reportlab and document its installation, or include them in the supported setup dependencies.

#### Coverage
- Verified the on-disk plan SHA256 matches the supplied value.
- Read the changed source, configuration, workflow, documentation, and test files; inspected notebook source and relevant saved outputs.
- Traced data preparation, split handling, training, OOF provenance, GMM fitting, router selection, permutation controls, T7 evaluation, and shared artifact writers/readers.
- Inspected the existing fusion registry, base contract, contract tests, CLI/Makefile callers, and report consumer beyond the implementation file list.
- Read all supplied result JSON files and manifests; verified referenced artifact hashes. Inspected all three report figures and extracted PDF text, comparing displayed validation numbers with stored results.

#### Limitations
- Static review only: no files edited, tests run, training/inference executed, or work delegated. Existing test-success claims were not independently verified.
- Checkpoint, featurizer, and confidence files were hash-checked but not deserialized or numerically replayed; raw-data statistics and alignment were not independently recomputed.
- No T7 result artifacts or large-run dataset artifacts were available for outcome verification.

#### Host dispositions (inspection 1)
All five accepted: REV-001 add best-global-by-val row + paired difference; REV-002 cross_attn requires >= 3 sources; REV-003 exclusive per-run lock + atomic manifest; REV-004 readers accept explicit run ids and verify recorded manifests; REV-005 `report` optional dependency.

### Between inspections

- REV-001..005 fixed; reviewer #1 code-fingerprint change means every code edit yields new run ids.
- User asked for the full dataset: `scripts/build_full_features.py` rebuilt all 84,949 image reviews (84,437 valid; 512 HTTP 404). Dev stages T3-T6 run on `appliances_full` (3 seeds, 2h42m) and re-run on `appliances_zip1000` (identical numbers).
- Interim PDF reports generated per dataset under `reports/<dataset_id>/` (validation only).

### Inspection 2 — Codex (fresh session) — verdict **REVISE**

- Result: `/tmp/claude-1000/-home-ds-adaptive-fusion/62ea5f50-c598-4d89-8fed-0924aa304782/scratchpad/claudex/inspect2/claudex-t3giyl73/result.json`
- Snapshot sha256: `26aa7b65744ab84b010b19fd2172c817cc251f2e9e06208fed50e35ec544cd96` · session `01a1860c-6a56-7260-97af-03a1f7d607ab` · 219.26s
- Summary: The plan hash matches, and the core train/validation isolation follows the specification. Four concrete defects remain in reporting, recovery, fusion edge cases, and locking.

- **REV-006** (medium) `scripts/laporan_sementara.py:48`
  - Evidence: Explicit --select-run inputs are accepted without checking protocol compatibility or duplicate seeds. load_run checks dataset/split and file hashes, but load_results never compares training, OOF, or selection configurations across runs. Moreover, lines 272–273 describe the reader's current grid and permutation count. Loading the existing 20-permutation results with a config specifying 100 permutations would produce a report claiming 100 permutations and the wrong minimum p-value.
  - Fix: Reject duplicate seeds and incompatible producer protocols, including upstream training and OOF configurations. Derive report descriptions from verified producer metadata and stored results, not the reader's current config.
- **REV-007** (medium) `scripts/build_full_features.py:183`
  - Evidence: download_log.json is written before encoding. If encoding subsequently fails or is interrupted, rerunning make data-full downloads missing images and then always raises FileExistsError at this existing log, preventing recovery. Even removing the log does not resume completed embedding stages: each embedding is recomputed before write_once rejects its existing destination. A completed build also prevents make data-full from reaching the loader command.
  - Fix: Make recovery stage-aware: validate and reuse completed artifacts, resume only missing stages, and publish outputs atomically. Treat a completed build as reusable without modifying its frozen inputs.
- **REV-008** (medium) `src/adaptive_fusion/fusion/layers.py:12`
  - Evidence: projection always applies LayerNorm(out_dim). MeanFusion and ConcatFusion accept out_dim=1, which the base contract permits. LayerNorm over one coordinate removes all input information, so these configurations produce input-independent outputs and zero input gradients. Setting d_branch=1 likewise destroys every modality before fusion, even with a larger output dimension. Existing contract tests use out_dim=4 and miss these cases.
  - Fix: Explicitly reject branch/output widths that make the notebook architecture degenerate, or support scalar projections without scalar LayerNorm. Add focused contract coverage for out_dim=1 and d_branch=1.
- **REV-009** (medium) `src/adaptive_fusion/experiments/gmm_router.py:250`
  - Evidence: Unlinking the lock file allows concurrent callers to lock different inodes for the same run. Caller B can open the old file and pause before flock; caller A then unlinks and releases it; caller C creates and locks a new file; B subsequently locks the old inode. After a stage failure before output creation, B and C can both pass _run_stage's empty-directory checks and produce the same run concurrently. The current nested lock test does not cover this interleaving.
  - Fix: Keep the lock file permanently and release only the flock/file descriptor. Add a multiprocess regression covering an opener paused between open and flock while another caller releases the lock.

#### Coverage
- Verified the on-disk plan SHA256 against the supplied hash.
- Read changed source, configurations, workflow documents, research decisions, experiment documents, and tests; inspected notebook source and applicable phase procedures.
- Traced data preparation, training, OOF provenance, GMM fitting, router selection, permutation controls, T7 evaluation, summaries, and report generation.
- Inspected related registry/base-contract code and shared artifact readers and writers beyond the change list.
- Read all 42 result JSON files, including 18 manifests; all 60 uniquely referenced artifact files matched their recorded hashes.
- Inspected all six report figures and extracted text from both PDFs, comparing displayed validation results with stored JSON.

#### Limitations
- Static review only: no files edited, tests run, training/inference executed, or work delegated. Test-success claims were not independently verified.
- Checkpoint, confidence, and featurizer binaries were hash-checked but not deserialized or numerically replayed.
- Raw-data statistics and sample alignment were not independently recomputed; no T7 result artifacts were available.
- PDF text and standalone figures were inspected; complete PDF page layout was not rendered.

#### Host dispositions (inspection 2)
All four accepted and fixed by the host (Claude):
- REV-006: report rejects duplicate seeds and runs whose select/full `config_sha256` differ; candidate and permutation counts are read from stored results.
- REV-007: feature builder is stage-aware (reuses finished CSV/URLs/download log/embeddings; atomic writes). Verified: `make data-full` on the finished build reuses everything (4.7s), fingerprint unchanged.
- REV-008: `projection` rejects widths < 2; tests for out_dim=1 and d_branch=1 on all three methods.
- REV-009: lock file is never unlinked; regression test for the paused-opener interleaving.
- Proof (host-run): `make lint` clean; `make test` 72 passed.

**Inspection budget exhausted (MAX_INSPECTION_ROUNDS = 2).** The REV-006..009 fixes have NOT been independently inspected. Decision on an extra inspection round is left to the user.
- **Extra inspection round 3:** approved by the user (2026-10-29) beyond MAX_INSPECTION_ROUNDS.
- **Inspection 3 attempt 1: FAILED (not approval).** Codex exited after 53.5s with "You've hit your usage limit ... try again at 1:18 PM". Diagnostics: `.../scratchpad/claudex/inspect3/claudex-b7zpo6qv/`. Will retry after the limit resets; dev stages (no test access) re-run meanwhile because the code fingerprint changed.
- **Dev re-run with fixed code (2026-10-29):** T3–T6 re-run on both datasets with new run ids. Outputs reproduced bit-identically versus the pre-fix runs (zip 12/12, full 27/27: checkpoints, OOF confidences, featurizers, routers by hash; JSON content equal apart from run ids). Full runs: full 47c90b5abb7c/0d95e379dbc1/bdfcb517910c, oof 0b3355efc132/97fc4398a29b/45007a2c3290, select 11b48e13ac02/68a734194d12/4704ebc9a41f.
- **Inspection 3 attempt 2: FAILED (not approval).** Codex rejected the prompt: 1,354,211 chars > 1,048,576 limit (notebook 453K, two PDFs 210K each, zip selection JSONs 57K each).
- **Inspection 3 attempt 3 (base workaround):** `--base eef823f7f7c30646839199821c2df0b53349a1f8`, a dangling commit object (`git commit-tree`, no branch/HEAD/index change) = `6012ae10` + six non-code artifacts copied byte-for-byte from the working tree (`adaptive_fusion new.ipynb`, both PDFs, `results/appliances_zip1000/*/router_selection/seed4{2,3,4}.json`). All code/tests/configs/docs remain in the diff; a host note explaining the base was passed via `--feedback`. Codex connectors disabled during the run; `~/.codex/config.toml` restored (SHA `30075e0d…` verified).

### Inspection 3 — Codex (fresh session) — verdict: REVISE
- Summary: The plan hash matches. Scalar-width validation and persistent locking address REV-008/009. Reporting compatibility remains incomplete, and resumed feature builds can change encoder inputs.

- **REV-006** (medium, reopened) `scripts/laporan_sementara.py:62`
  - Evidence: The report compares only Full and selection config hashes; `stage_config()` excludes OOF settings from both, so seeds produced with different `n_folds`/`inner_val_size` pass. The report also unconditionally describes five-fold training.
  - Fix: Verify each selection run's upstream OOF manifest, compare its config hash across seeds, derive the fold description from producer metadata; add coverage for mixed OOF protocols.
- **REV-010** (medium) `scripts/build_full_features.py:179`
  - Evidence: Resumed builds reload the CSV with default pandas NA parsing then `fillna('')`; literal 'NA'/'N/A' text (e.g. rows 3861, 6361) becomes '' on resume, giving different embeddings from an uninterrupted build.
  - Fix: Identical, lossless text loading for fresh and resumed builds; interrupted-build regression.
- **REV-011** (low) `scripts/laporan_sementara.py:216` — permutation legend hardcodes "20x". Fix: use stored count.
- **REV-012** (low) `reports/*/laporan_sementara_gmm_router.pdf` — PDFs cite selection runs that no longer exist. Fix: regenerate from retained runs.

#### Coverage / limitations (Codex)
- Verified plan SHA256; inspected all changed files; traced data prep, fusion contracts, training, OOF isolation, GMM, permutation selection, T7, summary/report; inspected artifact writers, verification, locking, feature-build recovery; all 42 result JSONs read and 60 referenced artifacts hash-matched; figures and PDF text compared with stored artifacts.
- Static review only; binaries hash-checked not replayed; no T7 artifacts present; PDF layout not rendered.

#### Host dispositions (inspection 3)
All four accepted and fixed by the host (Claude); all changes are in `scripts/` (outside the package code fingerprint, so pipeline run ids and results are unchanged):
- REV-006: `load_results` loads and hash-verifies the upstream `oof` run of every select run, requires one `config_sha256` per stage (select, oof, full) across seeds, checks oof→full lineage, and reads `n_folds` from the producer's `confidences.npz` (`fold_of`) for the PDF text. Test `test_report_rejects_mixed_oof_protocols`.
- REV-010: `read_reviews_csv` (`keep_default_na=False`, `title`/`text` as str) and shared `text_inputs`. Test `test_feature_builder_resume_keeps_text_inputs_identical` (interrupt before text encoding, resume, compare encoder inputs). Real-data impact check: the actual full build encoded text in one uninterrupted run, so it was unaffected — the 8 NA-like rows (6 titles, 2 texts) have stored embeddings with cosine 1.0000 to the literal strings (≈0.60 to ''); `read_reviews_csv` returns all 84,949 title/text strings identical to a fresh zip read.
- REV-011: legend uses stored `n_permutations`, with a length check on `null_scores`.
- REV-012: both PDFs regenerated; they cite zip 30238143d404/ceb6172553a0/ac471a326ab1 and full 11b48e13ac02/68a734194d12/4704ebc9a41f.
- Proof (host-run): the 3 changed tests fail on HEAD scripts and pass with the fixes; `make lint` clean; `make test` 74 passed.
- Internal `reviewer` subagent on the round-3 fixes: OK, two minor points, both applied: (1) report also checks lineage by content hash (select's `confidences` input = oof output; select checkpoints = full checkpoints via `_same_checkpoints`); (2) `ruff format` on the new code. Not applied (needs team approval, changes the package fingerprint and every run_id): `load_run` asserting `identity.run_id == requested run_id`. Re-proof: `make test` 74 passed, `make lint` clean, both PDFs regenerated.
- **Status:** inspection 3 verdict was REVISE; the fixes are host-verified but NOT independently inspected by Codex. T7 not run.
