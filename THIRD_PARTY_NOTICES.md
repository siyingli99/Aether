# Third-party notices

This repository contains stripped or integration-oriented copies of third-party projects. The AETHER research-only license does not relicense them.

| Component | Upstream source | Retained license | Included here | External material | Local changes |
|---|---|---|---|---|---|
| DriveStudio / OmniRe | https://github.com/ziyc/drivestudio | MIT (`third_party/drivestudio/LICENSE`) | Stripped source, configs, dataset/runtime and training code | Datasets, pretrained checkpoints, and some optional dependencies are not supplied | Reduced to the AETHER-required surface; path/config integration and compatibility adjustments may differ from upstream |
| SplatAD gsplat fork | https://github.com/NVlabs/splatad | Apache-2.0 (`third_party/splatad_gsplat/LICENSE`) | Stripped Python/CUDA rasterization package | Trained Phase-B sidecars and bundles are not supplied | Packaged under `splatad_gsplat`; retained only for AETHER LiDAR integration and may differ from upstream layout |
| UniAD | https://github.com/OpenDriveLab/UniAD | Apache-2.0 (`third_party/uniad_sim/LICENSE`) | Stripped simulation/adapter source | UniAD checkpoint is not supplied | AETHER/HUGSIM-compatible launcher, IPC, path, and closed-loop integration changes |
| Difix3D / Difix | https://github.com/NVlabs/Difix3D | NVIDIA license with non-commercial restrictions (`third_party/difix3d/LICENSE.txt`) | Stripped pipeline and helper scripts | Hugging Face model weights are not supplied | Directory/daemon wrappers and AETHER refinement integration |
| LimSim | https://github.com/PJLab-ADG/LimSim | GPL-3.0 | Not included; runs as a separate process | Clone the AETHER integration branch described in the README | Headless lockstep bridge is maintained in the separate LimSim fork |

## Dataset and model notices

- PandaSet is an external dataset. AETHER does not grant or assert a right to redistribute PandaSet files, including derived/processed copies. Consult the dataset provider's current terms before use or publication.
- `SMPL_NEUTRAL.pkl` is not needed by the bundled scene 095 runtime and must not be redistributed through this project.
- UniAD checkpoints and Difix model weights must be acquired by the user from an authorized source under the upstream terms.
- The two downloadable vehicle assets are converted from 3DRealCar sequences and retain their source sequence IDs. The 3DRealCar project declares the dataset Apache-2.0; preserve its license and citation when redistributing or publishing results.
- `assets/local/pandaset095_processed.zip` is an internal local copy, ignored by version control, and must not be published without a separate rights review.

## Downstream obligations

Before redistributing any checkout or artifact:

1. Preserve every nested third-party license and notice.
2. Review the exact upstream revision and local diff; this stripped copy does not claim byte-for-byte identity with upstream.
3. Remove external datasets, checkpoints, weights, generated outputs, caches, and private archives.
4. Verify that the planned use satisfies both the AETHER research-only terms and every applicable third-party/data/model license.

URLs above identify upstream projects only; they are not download promises for compatible assets.
