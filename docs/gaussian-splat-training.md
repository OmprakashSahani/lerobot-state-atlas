# Real Gaussian Splat Training — First Experiment

## Goal

Answer one question: can the already-prepared real COLMAP reconstruction produce a visually
useful Gaussian Splat? This does not establish renderer readiness, robot/world alignment,
metric calibration, production or mobile suitability, or performance suitability.

## Known prepared input

Dataset: `.cache/gaussian-splat/colmap-4fps/gs-input-model0`

Verified: 167 undistorted images, 68,630 sparse points, 282,087 observations, mean track
length 4.110258, COLMAP-reported mean reprojection error 0.852467 px, a coherent sparse
workcell reconstruction, a plausible recovered camera trajectory, and approximately 111 MB.

Do not rerun feature extraction, matching, mapping, or undistortion for this experiment.
Train from this prepared reconstruction. It is not yet a Gaussian Splat.

## Why training is remote

The local NVIDIA GeForce 920MX has 2 GB VRAM and is unsuitable. Use a temporary, private
Linux NVIDIA GPU environment. A 24 GB-class RTX 3090, RTX 4090, A10, or equivalent is the
preferred safe target, not a measured memory requirement for this dataset.

## Prepare only the training input

Use explicit shell commands from the repository root; keep paths quoted and inspect the archive.

```bash
DATASET=.cache/gaussian-splat/colmap-4fps/gs-input-model0
PARENT=$(dirname "$DATASET")
test -d "$DATASET/images"
test -d "$DATASET/sparse"
find "$DATASET/images" -maxdepth 1 -type f | wc -l
du -sh "$DATASET"
tar -C "$PARENT" -czf gaussian-training-input.tar.gz "$(basename "$DATASET")"
tar -tzf gaussian-training-input.tar.gz
sha256sum gaussian-training-input.tar.gz
```

The archive must contain only `gs-input-model0/`: its 167 images and selected sparse model.
Do not archive `.cache/gaussian-splat`. Exclude `IMG_5299.MOV`, other `frames-4fps` content,
`database.db`, matching or mapper logs, discarded models, contact sheets, visualization PNGs,
unrelated cache content, and Git metadata. Confirm the listing contains only the prepared
dataset tree. For experiment one, archive SHA-256 is sufficient transfer-integrity evidence.

## Private transfer and environment

Upload only the archive to a private temporary GPU instance. Verify its SHA-256 after transfer
and extract privately. Do not upload the MOV or unrelated cache content. After safely returning
results, remove remote storage per provider workflow. Upload is manual and not automated here.

Before installation or training, record a specific Nerfstudio release or Git commit; do not use
floating `latest`. Save `versions.txt` with the date, OS, GPU, NVIDIA driver, applicable CUDA,
Python, PyTorch, Nerfstudio version or commit, and gsplat version. No version is selected here.

## Confirm version-sensitive commands

Use `ns-train --help` and relevant subcommand help on the pinned remote installation.
Confirm, rather than fabricate, flags for the seed, step count, dataparser, evaluation,
checkpoint resume, and PLY export. The intended structure is schematic and version-sensitive:

```text
ns-train splatfacto ... colmap --data <dataset>
```

It is not an exact validated command until confirmed against the installed version.

## Ingestion gate before full training

First make Nerfstudio ingest the prepared dataset. Confirm expected cameras and images; inspect
camera frustums and initial SfM points; and verify orientation, structure, and paths. If geometry
looks wrong, **stop**. Do not start training or automatically rerun COLMAP; investigate first.

## Baseline training and evaluation

Use standard Splatfacto, not `splatfacto-big`, and avoid premature tuning. Use a fixed seed and
step policy if supported. Save the exact command, resolved config, logs, and checkpoints. The
objective is a visually useful reconstruction, not maximizing metrics.
Training quality remains independent of later renderer SPZ restrictions; do not force SH
degree 0 during training without a demonstrated training reason.

Visually inspect novel or fixed-path renders for workcell coherence, both regions, floaters,
duplicates, missing surfaces, stretched or spiky Gaussians, unstable backgrounds, transparent
or reflective artifacts, and discontinuities around known COLMAP gaps. Record configured
held-out metrics, but do not invent or require PSNR, SSIM, or LPIPS. Visual review is mandatory.

## Preserve artifacts, then stop

Return the exact command, `versions.txt`, resolved config, logs, checkpoint, evaluation renders,
full exported Gaussian PLY, and its SHA-256. Store private results below ignored
`.cache/gaussian-splat/`; never commit them. The full PLY is the reference training result;
do not overwrite or destructively modify it.

After visually evaluating the PLY, **STOP AND REVIEW**. Do not reduce splats, remove spherical
harmonics, convert to SPZ, package for Spark, add a real production manifest, align robot
coordinates, calibrate scale, profile browsers, or enable production. Those are separate
experiments. Success proves only that this capture can produce a Gaussian reconstruction; it
does not satisfy renderer, calibration, alignment, performance, or production gates.
