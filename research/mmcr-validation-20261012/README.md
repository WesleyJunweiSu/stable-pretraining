# MMCR validation for PR #464

Evidence for the two review requests on
[galilai-group/stable-pretraining#464](https://github.com/galilai-group/stable-pretraining/pull/464).
This branch is the PR branch plus validation tooling. It is **not intended to be merged**.

Workflow: [`.github/workflows/mmcr-validation.yml`](../../.github/workflows/mmcr-validation.yml).
Results from each run are committed to branch `mmcr-validation-results`.

## 1. Real two-process DDP on Linux

Job `ddp` runs the unchanged `stable_pretraining/tests/distributed/test_mmcr.py`
on `ubuntu-latest` with two Gloo CPU processes. `linux-ci/check_junit.py` fails
the job unless the test is reported as passed (a skip counts as failure).

`linux-ci/ddp_mutation_check.py` reruns the test's exact assertions against
three injected bugs and requires each to be caught:

| Injected bug | Detected by |
|---|---|
| No cross-rank gather | loss mismatch |
| Gather without autograd (own slice re-inserted) | gradient mismatch (loss matches) |
| Extra world-size factor on the loss | loss mismatch |

## 2. Longer training

Same protocol on every run: CIFAR-10 train split, seed-7 permutation, first
5,000 images held out and the next 10,000 used for training (official test set
unused), CIFAR ResNet-18 from scratch, 2 views, batch 64, λ = 0, AdamW lr 1e-3,
20 epochs (3,120 steps). The final epoch is reported, not the best.

- **Online probe**: linear head trained alongside, scored on the held-out 5,000 each epoch.
- **Frozen ridge probe** (`frozen_probe.py`): StandardScaler + RidgeClassifier
  (α = 1, no search) refitted on frozen features of the *initial* and *final*
  encoders. The gain therefore cannot come from probe training.
- **Effective rank** (`linux-ci/feature_spectrum.py`): spectral check for
  low-rank collapse, which per-dimension variance cannot rule out.
- **SimCLR reference**: the same example script with `--method simclr` and the
  same budget. Neither method was tuned and this is one seed, so it is a sanity
  reference, not a ranking.

`mac-run/` holds the original run on an Apple M4 (MPS, PyTorch 2.9.1, with
`PYTORCH_ENABLE_MPS_FALLBACK=1` so SVD runs on the CPU). That run predates the
snapshot-overwrite guard in `examples/mmcr_cifar10.py`; the CI runs use the
guarded code from the PR.

## Scope

One seed, a CIFAR subset, CPU/Gloo DDP only. This does not validate NCCL or
multi-GPU training and does not reproduce the paper's benchmark numbers.
