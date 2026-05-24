# results/

Outputs of every experiment, structured per PLAN.md §11.

## Layout

```
results/
├── .schema/                                # Reference templates for run artifacts
│   ├── config.json
│   ├── eval_results.json
│   └── convergence_report.json
├── E1_mvhfmd_baseline/
│   ├── N2_seed42/    {config.json, train.log, checkpoint_best.pt, convergence_report.json, eval_results.json}
│   ├── N2_seed1337/
│   ├── N2_seed2024/
│   ├── N4_seed{42,1337,2024}/
│   └── single_seed{42,1337,2024}/
├── E2_oamvhfmd_combined/N4_seed{42,1337,2024}/
├── E3_embed_only/N4_seed{42,1337,2024}/
├── E4_loss_only/N4_seed{42,1337,2024}/
├── E5_n_scaling/{mvhfmd,oamvhfmd}_N{2,4,6,8}_seed{42,1337,2024}/
├── E6_hmdmv_baseline/N4_seed{42,1337,2024}/
├── E7_oracle_ablation/{dinov2_s,clip_b16,siglip_b16,resnet50_in1k,random_cnn}/
├── E8_weight_schedule/
├── E9_mlp_ablation/
├── E10_hotels50k/
├── E11_chexpert/
└── tables/                                 # Aggregated CSVs produced by scripts/aggregate_results.py
    ├── mvhfmd_baseline.csv
    ├── oamvhfmd_combined.csv
    ├── head_to_head.csv
    ├── n_scaling.csv
    ├── oracle_ablation.csv
    ├── weight_schedule_ablation.csv
    ├── mlp_ablation.csv
    └── hotels50k.csv
```

## What's committed vs gitignored

- **Committed**: `config.json`, `eval_results.json`, `convergence_report.json`, and the per-experiment `tables/*.csv` aggregates.
- **Ignored**: `checkpoint_best.pt`, training logs, anything > 100 MB. See `.gitignore`.

## How to aggregate

After running all seeds for one experiment:
```
python scripts/aggregate_results.py --exp results/E1_mvhfmd_baseline --out results/tables/mvhfmd_baseline.csv
```
