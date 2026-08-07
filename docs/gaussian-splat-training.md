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

## Experiment 1 result — 2026-08-07

### Remote environment

The first real training experiment ran on Google Colab Free with this pinned environment:

- GPU: NVIDIA Tesla T4, 15,360 MiB VRAM
- Python: 3.12.13
- PyTorch: 2.11.0+cu128
- PyTorch CUDA: 12.8
- CUDA compilation tools: 12.8
- Nerfstudio: 1.1.5
- gsplat: 1.4.0
- NumPy: 1.26.4

The prepared private training archive had SHA-256:

```text
f65d888248a0b38f3e8097c722c1bc768ed7ac4946b9481dd8793fd42413782c
```

The hash matched after browser upload. Extraction produced the expected 167 undistorted
images and selected COLMAP sparse model.

### Runtime constraints discovered

Initial Splatfacto attempts were killed by host OOM during gsplat's first-time CUDA compilation.
Parallel gsplat compiler child processes remained after the killed parent process.

Training was stable with:

```bash
export MAX_JOBS=1
```

and:

```text
--pipeline.datamanager.cache-images cpu
--pipeline.datamanager.max-thread-workers 1
```

A 2-iteration smoke test then completed end-to-end and wrote a valid checkpoint. A subsequent
100-iteration timing run completed with `EXIT_CODE=0`; after compilation and warm-up, ordinary
iterations were approximately 22–27 ms.

### Ingestion gate

A Nerfstudio viewer inspection showed training camera frustums distributed through a coherent
3D trajectory, with no obvious camera collapse to one point and no catastrophic pose explosion.
The scene was undertrained at that point, as expected. Therefore, only the ingestion and
camera-geometry gate passed at that stage.

With PyTorch 2.11, Nerfstudio 1.1.5 viewer checkpoint loading required:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

because PyTorch's `torch.load` behavior defaults to `weights_only=True` in this environment,
while this Nerfstudio version expects the older loading behavior. This override was used only
for checkpoints generated in this trusted experiment.

### 3,000-step baseline

A fresh 3,000-step Splatfacto run completed successfully. Its viewer result showed a coherent,
recognizable workcell but substantial blur and ghosted detail.

The checkpoint was `step-000002999.ckpt`. Its Gaussian tensors each contained 419,097 splats:

- `features_dc`: `(419097, 3)`
- `features_rest`: `(419097, 15, 3)`
- `means`: `(419097, 3)`
- `opacities`: `(419097, 1)`
- `quats`: `(419097, 4)`
- `scales`: `(419097, 3)`

### Resume investigation

Attempting to resume from the step-2,999 checkpoint failed after loading. The render assertion
saw `opacities.shape == torch.Size([68630])`. That count equals the original 68,630 COLMAP
points, while the saved checkpoint contains 419,097 Gaussian parameters.

Source inspection established that `SplatfactoModel.load_state_dict` resizes Gaussian parameters
to the checkpoint size and `VanillaPipeline.load_pipeline` calls that model loader. Splatfacto's
gsplat `strategy_state` is initialized separately with `strategy.initialize_state(...)`, and no
path containing strategy state was found in the saved checkpoint. In this experiment with pinned
Nerfstudio 1.1.5 and gsplat 1.4.0, resume did not restore sufficient strategy state for the
densified model. Manual reconstruction was intentionally avoided, so subsequent runs were
started fresh.

### 10,000-step baseline

A fresh 10,000-step Splatfacto run completed successfully. Visual quality improved over 3,000
steps and the workcell remained geometrically coherent, but large blurred and incomplete regions
remained.

### 30,000-step baseline

A fresh 30,000-step baseline used this validated command:

```bash
export MAX_JOBS=1

ns-train splatfacto \
  --output-dir /content/gaussian-experiment/real-run-30000-fresh \
  --max-num-iterations 30000 \
  --steps-per-save 5000 \
  --vis tensorboard \
  --pipeline.datamanager.cache-images cpu \
  --pipeline.datamanager.max-thread-workers 1 \
  colmap \
  --data /content/gaussian-experiment/gs-input-model0 \
  --colmap-path sparse
```

It completed with `EXIT_CODE=0`. Late iterations were typically around 73–98 ms, and training
remained stable through step 29,999.

### Visual-quality decision

**Result: the 30,000-step baseline does not pass the visually useful reconstruction gate.**

The reconstruction stayed geometrically coherent. The robot, benches, and major workcell
structures were recognizable. However, large soft or blurred regions, ghosting, incomplete
surfaces, and stretched or streaked Gaussian artifacts remained visible.

These artifacts remained after increasing training from 10,000 to 30,000 iterations. The
evidence from this experiment indicates that additional training duration did not resolve the
dominant artifacts. Capture coverage and/or reconstruction quality are therefore the next
variables to investigate.

This result does not establish renderer readiness, calibration, robot/world alignment, mobile
suitability, browser performance, or production readiness.

### Preserved reference artifact

The complete 30,000-step model was exported with spherical-harmonic coefficients as the
non-destructive reference PLY:

```text
.cache/gaussian-splat/trained/splat-30000.ply
```

Size:

```text
106006465 bytes
```

SHA-256:

```text
28fbf3a13262751dc5d45260bd09955dfbc48e4933439bd1d46a247fee70423f
```

The downloaded Windows copy and the copy under `.cache/gaussian-splat/trained/` were
independently hashed and matched the Colab export. The `.cache/` tree is ignored by Git, so the
PLY must not be committed.

### Experiment boundary

Do not:

- convert the PLY to SPZ
- reduce splats
- remove spherical harmonics
- integrate into Spark
- add a real production environment manifest
- align robot coordinates
- calibrate scale
- profile browsers
- enable production

The next Gaussian experiment should address capture/reconstruction quality first.
