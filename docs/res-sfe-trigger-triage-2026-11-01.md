# RES surface-efficiency GitHub Actions trigger triage

*Observed November 1, 2026. Non-study operational note; do not treat as a frozen protocol amendment or new inference.*

## Confirmed state
- Research branch: `res-surface-efficiency-assay`
- Branch commit at check: `5007859e0199d668df74edf9a9448026824e7dd2` (attempt 4)
- Only changed file in that commit: `experiments/res_surface_efficiency_assay/control.json` (`"attempt": 3` to `4`; `"operation": "run"` unchanged).
- Workflow: `.github/workflows/res-surface-efficiency-assay.yml`, triggered by a branch-filtered `push` with a path filter covering exactly that control file. It has no manual-dispatch trigger.
- Branch Actions listing: three runs, all from September 2026. Latest historical run: [36499730628](https://github.com/NeonInsight/evals/actions/runs/36499730628). No November run was shown when checked.
- Last *verified* frozen checkpoint: 73 of 128 fixtures, 511 of 896 observations; 55 fixtures and 385 observations outstanding. The 18 original Action archives were previously backed up. An absence of runs is not experimental completion.

## Hypotheses to resolve, without guessing
1. GitHub may suppress new workflow runs caused by a `GITHUB_TOKEN` push (GitHub's documented behavior), **but the credential used for attempt 4 is not established**.
2. Repository Actions may be disabled/restricted, or the workflow may be disabled.
3. GitHub may have registered neither the filtered push event nor a qualifying run. Check settings and event source instead of assuming a particular cause.

## Safe recovery
1. Recheck queued and recent runs immediately before any trigger. Abort any plan that would start duplicate full runs.
2. Inspect repository Actions settings and workflow status in an authenticated GitHub UI. Current connector read access could not prove these settings.
3. Determine how the attempt-4 commit was authenticated. Confirm whether it used a GitHub Actions `GITHUB_TOKEN`, an installation token, or a user credential.
4. If Actions are enabled, use **one** known working GitHub event path. A user-originated control-only commit is the minimal existing `push` path, but only after verifying credentials and run state. A manual dispatch route would require adding `workflow_dispatch` and registering its workflow file on the default branch first. Do not blindly increment attempts or use `Re-run jobs` as a substitute for a new frozen continuation.
5. Before running, review usage/cost implications of 16 shard jobs, prior slow shards, and whether the chosen trigger is an authorized use of the runner.
6. When a workflow exists, record its run ID, SHA, job results and artifact checksums. Keep the frozen model, dtype, scoring, fixture order, inference code, and preregistration unchanged. Report completion **only** from the validated manifest.

## References
- [Attempt-4 commit](https://github.com/NeonInsight/evals/commit/5007859e0199d668df74edf9a9448026824e7dd2)
- [Frozen workflow](https://github.com/NeonInsight/evals/blob/res-surface-efficiency-assay/.github/workflows/res-surface-efficiency-assay.yml)
- [Historical Actions run](https://github.com/NeonInsight/evals/actions/runs/36499730628)
- [GitHub: triggering a workflow](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
- [GitHub: manual workflow registration](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)

This file intentionally does not change the frozen experiment or request compute resources.
