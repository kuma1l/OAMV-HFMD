# Claude Code prompt — prep the OAMV-HFMD combined run (decisive A/B vs baseline)

Paste the fenced block into Claude Code from the repo root `D:\Research-WS\PIVOT\OAMV-HFMD`.
It is self-contained. It only PREPS + verifies — it does not launch training.

---

```text
You are in the OAMV-HFMD repo (D:\Research-WS\PIVOT\OAMV-HFMD). Obey CLAUDE.md: surgical
changes, think before coding, simplicity first. Do NOT modify docs/PLAN.md,
docs/CURRENT_STATUS.md, or oamv_hfmd/trainer.py.

## Goal
Make the OAMV-HFMD *combined* method launchable as ONE command, so we can run a single
OAMV N=4 training and compare it head-to-head against the already-trained MV-HFMD baseline
N=4 (results/E1_mvhfmd_upstream/N4_seed42/eval_results.json, field test_top1). This A/B is
the decisive "does the mechanism help" experiment, so correctness matters more than speed.

## What already exists (read these before writing anything)
- oamv_hfmd/model.py : OverlapAwareHybrid (subclass of MultiImageHybrid). At n>1 its
  forward() computes S = cos-sim matrix via a FROZEN DINOv2-S oracle and RETURNS it as
  output["similarity"] (shape B,N,N). MLP(S[i,:]) replaces the static img_embed_matrix when
  enable_overlap_embed=True. Constructor args: (arch, num_classes, n, pretrained_weights,
  oracle_kind="dinov2_s", mlp_hidden=64, enable_overlap_embed=True).
- oamv_hfmd/losses.py : overlap_md_loss(z_mv, z_single, S, tau_overlap, tau_kl,
  lambda_hyperparam) — already Hinton-scaled, drop-in vs baseline_md_loss.
- oamv_hfmd/trainer.py LINES 187-206 : the trainer ALREADY routes. If "similarity" is in
  the model output it calls md_loss_fn with the overlap signature (z_mv, z_single, S,
  tau_overlap=, tau_kl=, lambda_hyperparam=); otherwise the baseline signature. So the
  trainer needs NO changes for the combined variant.
- scripts/01_train_mvhfmd_baseline.py : the baseline driver. Use it as the structural
  template for the new script (config loading via _load_config_with_inherits, arg parsing,
  dataset/loader build, TrainConfig wiring, exhaustive Evaluator, eval_results.json dump).
- configs/E1_mvhfmd_upstream.yaml : the upstream baseline config (data_dir: hotel_8k_images,
  num_classes 7774, split_source). configs/E2_oamvhfmd_combined.yaml : the OLD combined
  config (still points at the Stream-3 data via base.yaml). configs/base.yaml has the
  oracle: block (kind dinov2_s, mlp_hidden 64) and loss: block (tau_overlap 4.0).

## Deliverable 1 — scripts/02_train_oamvhfmd.py
Mirror scripts/01_train_mvhfmd_baseline.py, with these differences only:
- Add --variant {combined,embed_only,loss_only}, default "combined".
- IMPLEMENT ONLY "combined" for now. For embed_only/loss_only, raise
  NotImplementedError with this note: "embed_only/loss_only need the trainer's
  similarity-based routing (trainer.py L192) decoupled from the loss choice, since the
  OverlapAwareHybrid always returns 'similarity' at n>1 — handle as a follow-up (E3/E4)."
  Do NOT silently produce broken ablation variants.
- Build the model with OverlapAwareHybrid instead of MultiImageHybrid:
    model = OverlapAwareHybrid(arch=arch, num_classes=ds_train.num_classes, n=n_views,
        pretrained_weights=cfg["model"].get("pretrained", True),
        oracle_kind=cfg["oracle"]["kind"], mlp_hidden=cfg["oracle"]["mlp_hidden"],
        enable_overlap_embed=cfg["model"].get("enable_overlap_embed", True))
- Pass md_loss_fn=overlap_md_loss (from oamv_hfmd.losses) into the Trainer. The trainer
  auto-routes via output["similarity"] — do not pass S yourself.
- Reuse the exhaustive Evaluator for the N=4 test eval exactly as script 01 does; the
  headline comparison number is mv_collection top1 -> eval_results.json "test_top1".
- The per-image n=1 second pass in script 01 is awkward for OAMV (the n=1 OverlapAwareHybrid
  has oracle+MLP params and no similarity path). SKIP that second pass for now; instead write
  the same eval_results.json fields the Evaluator provides (test_top1/test_top5/per_view_metrics
  etc.), and set single_view_per_image_top1_full_test to null with a comment. Keep it simple.
- config.json snapshot: include variant, oracle_kind, mlp_hidden, tau_overlap, and
  split_source via cfg.get("split_source", "unknown").

## Deliverable 2 — configs/E2_oamvhfmd_upstream.yaml
Like configs/E1_mvhfmd_upstream.yaml but for the combined method. Set:
  inherits: base.yaml
  method: "oamvhfmd_combined_upstream"
  split_source: "upstream_author_split_2026-05-29"
  model: { variant: "overlap_aware", num_classes: 7774, enable_overlap_embed: true }
  oracle: { kind: "dinov2_s", mlp_hidden: 64 }
  paths: { data_dir: "hotel_8k_images", out_dir: "./results/E2_oamvhfmd_upstream" }
  sweep: { n_views: [4], seeds: [42] }

## Deliverable 3 — local instantiation test (NO training, NO GPU needed)
Write a short throwaway check (run it, paste output, then delete it — do not commit):
- Build OverlapAwareHybrid(arch="vit_small_r26_s32_224", num_classes=50, n=4,
  pretrained_weights=False) on CPU. (First build downloads DINOv2-S via torch.hub —
  needs internet; if offline, report that and stop.)
- Forward a synthetic batch x = torch.randn(2,4,3,224,224). Assert output has keys
  "single","mv_collection","similarity"; assert output["similarity"].shape == (2,4,4).
- Compute overlap_md_loss(output["mv_collection"]["logits"],
  einops.rearrange(output["single"]["logits"], "(b n) k -> b n k", b=2, n=4),
  output["similarity"]); assert it is finite.
- loss.backward(); assert at least one model.overlap_mlp parameter has a non-None grad,
  and assert every model.oracle parameter has grad is None (oracle stays frozen).
Report PASS/FAIL on each assertion.

## Verification gate (report, then stop — do not train)
1. python -c "import ast; ast.parse(open('scripts/02_train_oamvhfmd.py').read())" (syntax).
2. The instantiation test above passes all assertions.
3. Print the exact launch command (below) but DO NOT run it.

## Launch command (for the human to run on the VM AFTER baseline N4 finishes)
Run a single OAMV combined N=4, seed 42, at the SAME batch size the baseline used (40) so
the A/B is fair:
  python scripts/02_train_oamvhfmd.py --config configs/E2_oamvhfmd_upstream.yaml \
    --seed 42 --n-views 4 --batch-size 40 --num-workers 16
Output -> results/E2_oamvhfmd_upstream/N4_seed42/eval_results.json (test_top1).
Compare test_top1 against results/E1_mvhfmd_upstream/N4_seed42/eval_results.json.

## Memory + fairness caveats to print in your final summary
- OAMV adds a frozen DINOv2-S forward on every view, so it uses MORE VRAM than the baseline.
  Baseline N=4 nearly maxed 24 GB at batch 40. If the OAMV run OOMs at 40, drop to 32 — BUT
  then the baseline N=4 must ALSO be re-run at 32 for a fair comparison (batch size changes
  the OneCycle schedule). Flag this; do not silently mismatch batch sizes.
- The first OAMV run downloads DINOv2-S weights via torch.hub (~90 MB) — needs internet on
  the VM (Vast has it).

## Rules recap
Surgical only. New files: scripts/02_train_oamvhfmd.py, configs/E2_oamvhfmd_upstream.yaml.
Do NOT edit trainer.py / model.py / losses.py (they already support this), and do NOT touch
PLAN.md or CURRENT_STATUS.md. Match script 01's style. Report what you changed and the
instantiation-test output; do not launch any training run.
```

---

## Note for Kumail (not part of the prompt)

- Good news from the code read: the trainer already auto-routes the overlap loss, so this is
  just a new driver + config, not a trainer rewrite.
- The prompt deliberately scopes to the **combined** variant only — that's the decisive A/B.
  The isolation ablations (embed_only/loss_only, your E3/E4) need a small trainer-routing
  tweak and are left as a flagged follow-up so nothing breaks silently.
- It does not launch anything. After the instantiation test passes, you run the one launch
  command on the VM once the baseline N=4 is done.
