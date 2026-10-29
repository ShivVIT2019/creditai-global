# Deployment Notes — CreditAI Global

How this project went from local-only to live on Hugging Face Spaces.

## Background

Originally deployed on Google Cloud Run. After the GCP $300 trial expired, I explored free-tier alternatives for keeping the demo live. This document captures what worked, what didn't, and why.

## Options I Considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| GCP (keep paying) | Fast cold starts, familiar | ~$1-2/mo recurring | Rejected: wanted $0 |
| Oracle Cloud Always Free | Full VM, no cold starts, $0 forever | ARM capacity constantly out, hands-on infra setup | Hit "out of capacity" in AD-1 and AD-2 |
| Azure for Students ($100) | Native cloud, generous credit | Region restrictions for Container Registry on student accounts | Partial progress, stopped |
| **Hugging Face Spaces** | Free forever, Gradio-native, zero infra | Cold start (~30-60s), 2 vCPU / 16GB limit | Picked |
| Streamlit Cloud | Free forever | Streamlit only (CreditAI is Gradio) | N/A |

## Steps That Worked

### 1. Prepared the repo for HF Spaces

Added YAML frontmatter to README.md so HF Spaces knows how to build and run the app:

```yaml
---
title: CreditAI Global
sdk: gradio
sdk_version: 4.31.0
python_version: "3.11"
app_file: app.py
license: mit
---
```

Key insight: HF **ignores** `.python-version` and `runtime.txt`. The `python_version` key in README YAML is the actual contract.

### 2. Trimmed what gets pushed

Training data (`ml/data/` — 4.7 GB of raw Lending Club CSVs) is not needed at runtime; only the trained model (`ml/artifacts/` — 10 MB) is. Added to `.gitignore`:

### 3. Set up Git LFS for binary model files

HF Spaces rejects binary files over a certain size through plain git. The 9.8 MB `credit_model.joblib` triggered this.

```bash
brew install git-lfs
git lfs install
git lfs track "*.joblib"
git add .gitattributes
```

### 4. Rewrote git history to put model files into LFS retroactively

The model file was committed as a raw binary months earlier. Adding LFS tracking only affects future commits — the old commits still contain the raw binary, and HF's pre-receive hook rejects the whole push because of them.

Fix:

```bash
git lfs migrate import --include="*.joblib" --everything
```

This rewrites every commit touching `.joblib` files, replacing raw binaries with LFS pointers across all history.

### 5. Added HF as a second git remote and force-pushed

```bash
git remote add hf https://huggingface.co/spaces/sivasaiatchyut/creditai-global
git push --force hf main
```

### 6. Fixed Python version mismatch

First build failed because HF defaulted to Python 3.13 and `scikit-learn==1.6.1` doesn't install cleanly on it. Added `python_version: "3.11"` to README frontmatter, committed, re-pushed — build succeeded.

## Final State

- **Live URL:** https://huggingface.co/spaces/sivasaiatchyut/creditai-global
- **Hardware:** 2 vCPU, 16 GB RAM (HF free tier)
- **Cost:** $0/month
- **Cold start:** ~30-60s after idle, then instant for ~15 min window
- **Model files:** 5 `.joblib` artifacts totaling ~10 MB, stored in Git LFS

## Lessons Learned

1. **Managed platforms beat raw infra for portfolio projects.** HF Spaces took ~2 hours including LFS rewrite; the Oracle VM path would've been 4-6 hours plus ongoing maintenance.
2. **Separate training data from runtime artifacts.** 4.7 GB of CSVs vs. 10 MB of model — only one needs to deploy.
3. **Git LFS is non-negotiable for ML repos with model files.** Setting it up before the first binary commit saves a history rewrite later.
4. **Free tiers have real tradeoffs** — cold starts, hardware limits, SDK constraints. Pick based on what the project actually needs, not what sounds nicest.

## What I'd Do Differently

- Set up Git LFS from day one when creating ML repos
- Pin Python version in both the deployment config and `.python-version` so local dev matches CI/prod
