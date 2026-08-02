# Base π0.5 versus Fine-tuned π0.5 checkpoint comparison

Roadmap item 6 compares the pinned Base π0.5 checkpoint with the final Dream Machines fine-tuned π0.5 checkpoint. It does not claim that an early fine-tuning checkpoint is published.

| Input | Repository | Revision |
| --- | --- | --- |
| Dataset | `DreamMachines/actuator_unboxing_4h_diverse` | `e973df866c80f52884cc68355579043cab828e78` |
| Base π0.5 | `lerobot/pi05_base` | `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` |
| Fine-tuned π0.5 | `DreamMachines/actuator_unboxing_4h_diverse_fullft_bs256` | `6c50dbbccd576e4e384ed51a8244272aab5f3c62` |

## Execution model

The workflow is local-only and network-free. Stage every declared file before running it; the command never downloads a model, dataset, tokenizer, processor, or image. A runner manifest pins paths, byte counts, hashes, device settings, explicit arm transforms, acknowledgement flags, and the output destination. Paths are relative to that manifest and may not traverse or pass through symlinks.

The exact three-camera observation is validated first. The fine-tuned robot-specific PI05 configuration and learned QUANTILES normalization state are then verified and shared by both policies. The declared tokenizer is `google/paligemma-3b-pt-224`, but it must exist as a complete verified local directory. Base π0.5 runs and is fully released before Fine-tuned π0.5 is constructed. Both receive one preprocessing result and clones of the same explicit `[1, 50, 32]` float32 diffusion-noise tensor.

The authoritative output is 50 postprocessed absolute position targets with 14 components at 50 FPS. The outputs are not deltas and are neither integrated nor added to the initial state. FK positions and XYZW orientations are derived visualization data. Raw generated gripper targets are device-specific unproven scalars—not physical jaw widths, calibrated openings, or hardware-safety claims.

## Commands

Run the non-mutating resource and static-input check first:

```console
lerobot-state-atlas run-checkpoint-comparison runner.json --preflight-only
```

Machine-readable output is available with `--json`. After every required local asset is staged and preflight succeeds, execute the complete transaction:

```console
lerobot-state-atlas run-checkpoint-comparison runner.json
```

The installed layout is:

```text
<run-directory>/
  comparison/
    manifest.json
    plans.json
  run-receipt.json
```

The comparison bundle uses schema 1.1. The receipt contains deterministic software, hardware, checkpoint, configuration, processor, tokenizer, noise, observation, projection, and artifact identities; it deliberately excludes timestamps, durations, absolute input paths, and temporary names. Installation stages and validates the entire run before an atomic rename. If replacement cleanup begins and later fails, the complete new destination remains installed and any partial backup is retained with a recovery diagnostic; never restore such a backup blindly.

## Required local inputs and acknowledgements

The runner manifest schema is `lerobot-state-atlas.checkpoint-comparison-runner` version 1.0. It declares the observation manifest, two single-file SafeTensors checkpoints, fine-tuned config, pre/postprocessor JSON and learned state, tokenizer directory, TRLC-DK1-Follower URDF, CUDA device/dtypes, resource thresholds, projection policy, and output.

Real execution is CUDA-only. Memory thresholds are caller-supplied because checkpoint size alone cannot predict architecture, activation, decoded-tensor, mmap, and allocator overhead. `bfloat16` also requires a compatible CUDA device. Uncalibrated arm transforms require explicit acknowledgement. `allow-with-recorded-violations` requires a separate acknowledgement and never clips targets. An unavailable projection must be explicitly selected with an external reason; runtime failures are not converted into an unavailable result.

## Current blockers

No real comparison has been produced in this repository. The base and fine-tuned checkpoint files, both learned processor-state SafeTensors files, a locally staged PaliGemma tokenizer, and a valid synchronized three-camera observation sample are absent. The current workstation also lacks compatible CUDA and sufficient RAM. Synthetic fixtures exist only for tests and local UI inspection and must never be presented as real inference.

Operational failures are phase-tagged. Resolve the reported local input, hardware, compatibility, integrity, inference, projection, validation, installation, or cleanup issue and rerun from the beginning; the runner does not retry silently or return a base-only comparison.
