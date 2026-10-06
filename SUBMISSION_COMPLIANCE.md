# AISTATS 2027 Submission Compliance Manifest

This file documents the mechanical compliance gates enforced by the final-submission workflow. It does not replace the official conference instructions.

## Manuscript
- Anonymous author/institution placeholders are present.
- AISTATS 2027 style file is used without the `accepted` option.
- Main scientific narrative must end by page 8.
- AI Use Statement must appear immediately before references and outside the 8-page main-text budget.
- Checklist and appendices follow references.
- Title/abstract framing is treated as frozen except for minor consistency edits.
- Observational sensor analyses are explicitly labeled stress tests rather than causal ground truth.

## Supplement
- Built by `src/build_supplementary_archive.py`.
- Includes manuscript source/style, canonical implementation, clean processed data, canonical reports, verification scripts, and promoted post-audit evidence.
- Identifying repository/author links are forbidden by automated archive scan.
- Exploratory files are included only when they delimit or reproduce promoted claims.

## Automated gates
`.github/workflows/submission-final.yml` runs:
1. `verify_science.py`;
2. `verify_submission_synthesis.py`;
3. LaTeX compilation;
4. supplementary archive construction;
5. `verify_artifacts.py`;
6. PDF page-placement checks;
7. manuscript and archive anonymity scans;
8. commit of generated PDF/archive only after all gates pass.

A generated PDF or ZIP should be treated as submission-candidate material only when the workflow for its source commit completes successfully.
