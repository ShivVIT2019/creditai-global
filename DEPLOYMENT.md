# Deployment Journey — CreditAI Global

Honest chronological log of what it actually took to go from local-only Gradio app to a live, free, publicly-shareable demo on Hugging Face Spaces. Written as a debugging journal, not a success story — because the mistakes are the most useful part to re-read later.

---

## Context

The project was originally built on Google Cloud Run with a $300 free trial. After the trial expired, the live demo URL started 500'ing and I needed a new home for it. This document captures the entire path, including dead-ends, so future-me understands *why* the final solution looks the way it does.

---

## Phase 1 — Picking a Platform

### What I considered

| Platform | Why I looked | Why I ruled it out (or picked it) |
|---|---|---|
| GCP Cloud Run (keep paying) | Already deployed there | Didn't want $1-2/mo recurring; wanted $0 |
| Oracle Cloud "Always Free" | Full VM, no cold starts, free forever | Ampere A1 ARM capacity constantly out — hit "Out of capacity for shape VM.Standard.A1.Flex in availability domain AD-1" and AD-2 the same morning |
| Azure for Students ($100 credit) | Native cloud, generous credit | Student accounts have hidden region restrictions. Container Registry create was rejected with "Resource ... was disallowed by Azure" in US East and Central US |
| Hugging Face Spaces | Free forever, Gradio-native | **Picked.** No infra config, auto-build from a git push |
| Streamlit Cloud | Free, GitHub integration | Only hosts Streamlit — CreditAI is Gradio |

**Lesson:** "Free tier" is marketing. The actual constraints (region restrictions, ARM capacity, SDK compatibility) only reveal themselves when you try to use them.

---

## Phase 2 — First Attempt: Oracle Cloud VM (Dead End)

I spent several hours trying to launch an Always Free Ampere ARM VM on Oracle. Each attempt failed with:
Out of capacity for shape VM.Standard.A1.Flex in availability domain AD-1.

Tried AD-1, AD-2, and planned to try AD-3. ARM capacity on Oracle's free tier is in global demand and simply unavailable most of the time. There's no way to reserve or queue — you just retry until you get lucky.

**What I DID get out of this:**
- Oracle Cloud account created (`sakella58` tenancy, US East Ashburn region)
- $1 budget alert configured as a safety net
- SSH keys generated and stored at `~/.ssh/oracle-finsight.key` (saved for later use if I ever retry)

**Lesson:** Don't commit to infra paths where availability is unpredictable, especially under time pressure.

---

## Phase 3 — Pivot to Hugging Face Spaces

Decided to switch to Hugging Face Spaces because:
- CreditAI is already a Gradio app → zero UI rework
- HF Spaces is designed exactly for ML demos
- Free forever, no credit card for the account itself
- Public URL that works instantly

Created the Space here: https://huggingface.co/spaces/sivasaiatchyut/creditai-global

Now the real debugging began.

---

## Phase 4 — Push Attempt #1: Rejected (Authentication)

First push:

```bash
git remote add hf https://huggingface.co/spaces/sivasaiatchyut/creditai-global
git push hf main
```

Got:
remote: Invalid username or password.
fatal: Authentication failed for ‘.../creditai-global/’

**Problem:** macOS Keychain had stale credentials cached. The push was silently using those instead of prompting for an HF Access Token.

**Fix:** embed the token in the URL once:

```bash
git push https://sivasaiatchyut:<HF_TOKEN>@huggingface.co/spaces/sivasaiatchyut/creditai-global main
```

The token (`hf_...`) was created at `huggingface.co/settings/tokens` with Write permissions. Named it `creditai-deploy`.

**Lesson:** `git push` failing silently with saved credentials is a real pitfall on macOS. Always control the auth path explicitly on first push.

---

## Phase 5 — Push Attempt #2: Rejected (Non-Fast-Forward)

Second push with embedded token:
! [rejected] main -> main (fetch first)
error: failed to push some refs

**Problem:** When I created the HF Space in the browser, HF auto-generated an initial commit with a placeholder README. My local repo had no knowledge of that commit. Git refuses to overwrite history without permission.

**Fix:** force push, because our local README with HF YAML frontmatter is what we actually want:

```bash
git push --force ...
```

**Lesson:** Creating a Space via the UI *does* create an initial commit on the HF side. First push from local always needs `--force` to overwrite that placeholder.

---

## Phase 6 — Push Attempt #3: Rejected (Binary Files)
remote: Your push was rejected because it contains binary files.
remote: Offending files:
remote: - ml/artifacts/credit_model.joblib (ref: refs/heads/main)

**Problem:** HF Spaces blocks binary files over a certain size through plain git. The 9.8 MB `credit_model.joblib` was raw-committed months ago. Needed Git LFS.

**Attempted fix:**

```bash
brew install git-lfs
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git commit -m "Track .joblib with LFS"
git push --force ...
```

Still rejected with the same error.

**Why it still failed:** LFS tracking only applies to **new commits**. The February 2026 commit (`553bca4`) that originally added the model file still contained the raw binary in git history. HF's pre-receive hook scans every commit being pushed, found the old one, and rejected the whole push.

**Actual fix:** rewrite history to put the binary into LFS retroactively:

```bash
git lfs migrate import --include="*.joblib" --everything
```

This went through every commit (13 of them) that touched a `.joblib` file and replaced the raw binary with an LFS pointer. Commit hashes changed as a result — expected because history was literally rewritten.

Verified:

```bash
git lfs ls-files
```

Showed all 5 joblib files with LFS hashes. Pushed again — succeeded this time:
Uploading LFS objects: 100% (5/5), 10 MB

1fca4ef...e0fa9ad main -> main (forced update)

**Lesson:** Adding `git lfs track` is not retroactive. Any binaries committed *before* LFS tracking existed will still fail. `git lfs migrate import --everything` is the fix, but it rewrites history and requires `--force` on all remotes.

---

## Phase 7 — First Build Failure: Python Version Mismatch

The push worked, but HF immediately showed:
Build error
Job failed with exit code: 1

Scrolling the build log showed HF was using Python 3.13 (its default) and `scikit-learn==1.6.1` wasn't installing cleanly on it. My repo had `.python-version` set to 3.11 and `runtime.txt` as `python-3.11`.

**The real insight:** HF Spaces **ignores both** `.python-version` and `runtime.txt`. The actual contract is a `python_version` key in the YAML frontmatter of README.md.

**Fix:** patched the README's YAML to add `python_version: "3.11"`:

```yaml
---
title: CreditAI Global
sdk: gradio
sdk_version: 4.31.0
python_version: "3.11"
app_file: app.py
---
```

Committed, pushed, HF auto-rebuilt — this time successfully. App came up at the live URL.

**Lesson:** Each platform has its own version-pinning contract. "The right way to do X" depends on the platform, not on general Python conventions. Always find the platform's actual contract (HF docs call it the Space Configuration Reference).

---

## Phase 8 — Second Round of Binary File Rejection (Screenshots)

Later, when adding README screenshots:
- Copied 5 PNG files (~1 MB each) into `docs/images/`
- Committed and pushed

HF rejected again:
remote: Offending files:
remote: - docs/images/1.png
remote: - docs/images/2.png
remote: - docs/images/3.png
remote: - docs/images/4.png
remote: - docs/images/5.png

**Problem:** HF treats PNGs the same as `.joblib` — any binary beyond a small size needs LFS. I'd assumed PNGs were safe because GitHub handles them without LFS. That assumption was wrong.

**Fix:** same pattern as the joblib fix:

```bash
git lfs track "*.png"
git add .gitattributes
git commit -m "Track PNG files with Git LFS"
git lfs migrate import --include="*.png" --everything
git push --force ...
```

Pushed again — succeeded. Final `git lfs ls-files` showed 5 PNGs + 5 joblib files all LFS-tracked.

**Lesson:** HF's binary file policy is stricter than GitHub's. Any binary > ~500KB-ish needs LFS. Easy rule going forward: on HF Spaces, LFS-track every binary file type from day one.

---

## Phase 9 — Final State

- **Live URL:** https://huggingface.co/spaces/sivasaiatchyut/creditai-global
- **GitHub mirror:** https://github.com/ShivVIT2019/creditai-global
- **Hardware:** 2 vCPU, 16 GB RAM (HF free tier)
- **Monthly cost:** $0
- **Cold start:** ~30-60 seconds after inactivity, then instant for ~15 min window
- **LFS-tracked:** 5 `.joblib` artifacts + 5 PNG screenshots

---

## Timeline Summary

| Attempt | Result | Root cause |
|---|---|---|
| Oracle VM (AD-1) | Capacity error | ARM capacity exhausted |
| Oracle VM (AD-2) | Capacity error | Same |
| HF push #1 (default auth) | Auth failed | Stale Keychain credentials |
| HF push #2 (force) | Rejected non-fast-forward | HF created initial commit, local repo didn't have it |
| HF push #3 (force) | Rejected binary | credit_model.joblib was raw-committed, not LFS |
| HF push #4 (after lfs track) | Still rejected | LFS tracking isn't retroactive; needed `lfs migrate` |
| HF push #5 (after lfs migrate) | Push succeeded | — |
| HF build #1 | Build error | Python 3.13 incompatible with scikit-learn 1.6.1 |
| HF build #2 (python_version pinned) | Build succeeded | — |
| HF push #6 (PNGs added) | Rejected binary again | Same mistake as joblib — PNGs weren't LFS-tracked |
| HF push #7 (after lfs migrate pngs) | Push succeeded | — |

Total: **10 attempts, 7 failures, 3 successes.**

---

## Hard-Won Lessons (For Future Me)

1. **"Free tier" doesn't mean "no constraints."** Oracle has ARM capacity scarcity. Azure Student has region restrictions. HF has binary size limits. GCP had the API key leak risk from May. Every platform has its own tax.

2. **Git LFS tracking is NOT retroactive.** `git lfs track` only affects future commits. Old commits with raw binaries still exist in history. `git lfs migrate import --everything` is the full fix.

3. **Platform version pinning conventions differ.** `.python-version` works for pyenv. `runtime.txt` works for Heroku. HF Spaces needs `python_version` in README YAML frontmatter. Don't assume — check the platform's config reference.

4. **macOS Keychain can silently sabotage git pushes.** When auth "just fails," check whether cached credentials are being used before assuming the token is wrong.

5. **UI-created HF Spaces have an initial commit.** First push from local always needs `--force` to overwrite the placeholder.

6. **Separate training data from runtime artifacts aggressively.** 4.7 GB of CSVs vs 10 MB of trained model — only one needs to ship. The rest goes in `.gitignore`.

7. **Document failures, not just successes.** The "clean steps" version of this doc would've been shorter but useless. The honest version is what helps when something breaks again.

---

## Things I'd Do Differently Next Time

- Set up Git LFS **before** the first commit, not after
- Pin Python version in the deployment config from day one
- Test the full build pipeline locally with the same Python version the platform will use
- Keep screenshots and other binaries LFS-tracked from the start if there's any chance of deploying to HF
- When migrating platforms, do one small push first (just the README) to catch auth/config issues before pushing the full project
