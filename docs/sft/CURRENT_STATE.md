# Current State and Future Entry Point

**Status:** Canonical public resume point  
**Curriculum 1–6:** complete on the ~750M LLaMA-2-style model (32L / 1280d / 20H, 4096 context).  
**Phase 6.3:** GOLD exists. Do not stitch answers in the RAG controller.  
**1.4B:** planned, not started. See [SCALE_1_4B.md](SCALE_1_4B.md).  
**Math:** `regression_basic` is a known structural gap — do not remediate it on 750M.

**Order of work** (do not skip stages):

1. Unified from-scratch pretrain — [../training/01_UNIFIED_FROM_SCRATCH.md](../training/01_UNIFIED_FROM_SCRATCH.md)
2. Domain corpus + adaptation — [../training/02_DOMAIN_CORPUS.md](../training/02_DOMAIN_CORPUS.md) · [../training/02_DOMAIN_ADAPTATION.md](../training/02_DOMAIN_ADAPTATION.md)
3. Context extension 1024 → 4096 — [../training/03_CONTEXT_EXTENSION.md](../training/03_CONTEXT_EXTENSION.md)
4. SFT phases 1–6 — [README.md](README.md) · [SFT_PIPELINE_GUIDE.md](SFT_PIPELINE_GUIDE.md)

Maps: [../training/README.md](../training/README.md) (pretrain) · [README.md](README.md) (SFT). Autopilot: [AUTOPILOT.md](AUTOPILOT.md).

Tuned mix weights, measured eval results, checkpoint scores, and run history are not published.
