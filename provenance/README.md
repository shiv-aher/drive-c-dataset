# Environment provenance

- `environment.yaml`, `train.txt`, `val.txt`, `train_log.csv`, and the two YAML
  policy files are preserved provenance from the recovered Version 2 release.
- `corrected_training_pip_freeze.txt` is the complete package snapshot of the
  environment used for the corrected 22-epoch Version 3 training run.
- The corrected run specification, selected-epoch metrics, launch command, and
  checkpoint hash are stored at
  `../../outputs/corrected_training/RUN_SPEC.md` in the revision workspace.

The selected corrected checkpoint is `../checkpoints/best_health_mae.pth`
(SHA-256
`fcfb0ea54d6329bae2756f5fd32c39be860a90b04967dfde97f7b213c316836c`).
It was selected at epoch 7 by minimum held-out KITTI health MAE. DRIVE-C
evaluation results were not used for model selection.
