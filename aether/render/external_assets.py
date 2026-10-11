"""Runtime 3DRealCar Gaussian assets for DriveStudio ``RigidNodes``."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


_PARAMETER_NAMES = (
    "_means", "_features_dc", "_features_rest", "_opacities", "_scales", "_quats"
)

_ASSET_FORMAT = "aether.gaussian_vehicle.v1"


def _quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product for scalar-first quaternions, with broadcasting."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


@dataclass(frozen=True)
class GaussianVehicleAsset:
    means: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    size_lwh: tuple[float, float, float]
    source: str

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    def as_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "_means": self.means,
            "_features_dc": self.features_dc,
            "_features_rest": self.features_rest,
            "_opacities": self.opacities,
            "_scales": self.scales,
            "_quats": self.quats,
        }

    def save(self, model_dir: str | Path) -> Path:
        """Write the portable AETHER vehicle-asset format.

        The bundle contains only tensors used by DriveStudio ``RigidNodes``.
        It is therefore much smaller than the original training checkpoint and
        can be moved between scenes without any training-code dependency.
        """
        root = Path(model_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "asset.pt"
        torch.save(
            {
                "format": _ASSET_FORMAT,
                "size_lwh": list(self.size_lwh),
                "source": self.source,
                "tensors": {
                    name.removeprefix("_"): tensor.detach().cpu().contiguous()
                    for name, tensor in self.as_tensors().items()
                },
            },
            path,
        )
        return path

    @staticmethod
    def _prune(tensors: dict[str, torch.Tensor], max_gaussians: int) -> dict[str, torch.Tensor]:
        count = int(tensors["means"].shape[0])
        keep_count = min(max(1, int(max_gaussians)), count)
        if keep_count == count:
            return tensors
        # Approximate projected contribution: opacity times the two largest
        # physical scale axes. This is deterministic and preserves broad body
        # surfaces as well as high-opacity detail.
        alpha = torch.sigmoid(tensors["opacities"].squeeze(-1))
        physical_scales = torch.exp(tensors["scales"]).clamp(max=0.25)
        area = physical_scales.topk(k=2, dim=-1).values.prod(dim=-1)
        keep = torch.topk(alpha * area, k=keep_count, sorted=False).indices
        return {name: value[keep].contiguous() for name, value in tensors.items()}

    @classmethod
    def _load_aether(cls, root: Path, max_gaussians: int) -> "GaussianVehicleAsset":
        payload = torch.load(root / "asset.pt", map_location="cpu", weights_only=False)
        if payload.get("format") != _ASSET_FORMAT:
            raise ValueError(
                f"unsupported vehicle asset format in {root / 'asset.pt'}: "
                f"{payload.get('format')!r}"
            )
        required = {name.removeprefix("_") for name in _PARAMETER_NAMES}
        raw = payload.get("tensors", {})
        missing = sorted(required.difference(raw))
        if missing:
            raise ValueError(f"vehicle asset is missing tensors: {missing}")
        tensors = {
            name: raw[name].float().contiguous()
            for name in sorted(required)
        }
        count = int(tensors["means"].shape[0])
        if count < 1 or tensors["means"].shape != (count, 3):
            raise ValueError(f"invalid vehicle means shape: {tuple(tensors['means'].shape)}")
        for name, value in tensors.items():
            if int(value.shape[0]) != count:
                raise ValueError(
                    f"vehicle tensor {name!r} has {value.shape[0]} rows; expected {count}"
                )
        tensors = cls._prune(tensors, max_gaussians)
        size = tuple(float(value) for value in payload["size_lwh"])
        if len(size) != 3 or any(value <= 0 for value in size):
            raise ValueError(f"invalid size_lwh in {root / 'asset.pt'}: {size}")
        return cls(
            means=tensors["means"],
            features_dc=tensors["features_dc"],
            features_rest=tensors["features_rest"],
            opacities=tensors["opacities"],
            scales=tensors["scales"],
            quats=tensors["quats"],
            size_lwh=size,
            source=str(payload.get("source") or root.resolve()),
        )

    @classmethod
    def load(cls, model_dir: str | Path, max_gaussians: int = 50_000) -> "GaussianVehicleAsset":
        """Load an AETHER bundle or convert a native 3DRealCar asset.

        3DRealCar asset axes are ``x=length, y=down, z=width`` with the roof
        near y=0. DriveStudio rigid canonical axes are
        ``x=length, y=width, z=up`` centred on the 3D box.  The fixed -90°
        rotation about x below performs that conversion.
        """
        root = Path(model_dir)
        if (root / "asset.pt").is_file():
            return cls._load_aether(root, max_gaussians)
        gs_path, size_path = root / "gs.pth", root / "wlh.json"
        if not gs_path.is_file() or not size_path.is_file():
            raise FileNotFoundError(f"3DRealCar asset requires gs.pth and wlh.json: {root}")
        with open(size_path, "r", encoding="utf-8") as handle:
            width, length, height = (float(value) for value in json.load(handle))
        state = torch.load(gs_path, map_location="cpu", weights_only=False)[0]
        tensors = {
            "means": state[1].float().contiguous(),
            "features_dc": state[2].float().contiguous(),
            "features_rest": state[3].float().contiguous(),
            "scales": state[5].float().contiguous(),
            "quats": state[6].float().contiguous(),
            "opacities": state[7].float().contiguous(),
        }
        if tensors["features_dc"].ndim == 3 and tensors["features_dc"].shape[1] == 1:
            tensors["features_dc"] = tensors["features_dc"].squeeze(1)

        tensors = cls._prune(tensors, max_gaussians)

        # Rx(-pi/2): (x, y, z)_asset -> (x, z, -y)_DriveStudio.
        means = tensors["means"][:, [0, 2, 1]].clone()
        means[:, 2].mul_(-1.0).sub_(height * 0.5)
        root_half = 2.0 ** -0.5
        q_axis = torch.tensor([root_half, -root_half, 0.0, 0.0]).expand_as(tensors["quats"])
        quats = _quat_multiply(q_axis, tensors["quats"])
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        return cls(
            means=means,
            features_dc=tensors["features_dc"],
            features_rest=tensors["features_rest"],
            opacities=tensors["opacities"],
            scales=tensors["scales"],
            quats=quats,
            size_lwh=(length, width, height),
            source=str(root.resolve()),
        )


class ExternalVehiclePool:
    """Append reusable 3DRealCar instances to a loaded DriveStudio model."""

    def __init__(self, renderer: Any, asset: GaussianVehicleAsset | list[GaussianVehicleAsset],
                 capacity: int = 6, hide_original_rigid: bool = True,
                 hide_original_dynamic: bool | None = None) -> None:
        assets = list(asset) if isinstance(asset, list) else [asset] * int(capacity)
        if not assets or (not isinstance(asset, list) and int(capacity) <= 0):
            raise ValueError("vehicle pool capacity must be positive")
        if isinstance(asset, list) and int(capacity) not in (len(assets), 6):
            raise ValueError("capacity must match the heterogeneous asset list")
        model = renderer.trainer.models.get("RigidNodes")
        if model is None:
            raise RuntimeError("checkpoint does not contain RigidNodes")
        self.model = model
        self._models = renderer.trainer.models
        self.assets = assets
        self.asset = assets[0]
        self.capacity = len(assets)
        self.original_instance_count = int(model.instances_fv.shape[1])
        self._original_visibility = model.instances_fv[:, :self.original_instance_count].detach().clone()
        self.instance_ids = tuple(
            range(self.original_instance_count, self.original_instance_count + self.capacity)
        )
        self._append_instances()
        # ``hide_original_rigid`` is retained for old scenario files.  The
        # complete scene-editing path must also remove pedestrians/cyclists
        # kept by DriveStudio in DeformableNodes/SMPLNodes.
        hide_all = bool(hide_original_rigid) if hide_original_dynamic is None \
            else bool(hide_original_dynamic)
        self.removed_originals: dict[str, int] = dict(
            getattr(renderer, "excluded_dynamic_counts", {})
        )
        self._hidden_dynamic_ranges: dict[str, int] = {}
        if hide_all:
            self._hide_all_original_dynamic()

    def _hide_all_original_dynamic(self) -> None:
        """Hide every reconstructed dynamic instance, preserving new slots."""
        for name in ("RigidNodes", "DeformableNodes", "SMPLNodes"):
            model = self._models.get(name)
            visibility = getattr(model, "instances_fv", None) if model is not None else None
            if visibility is None or visibility.ndim != 2:
                continue
            original_count = (
                self.original_instance_count if name == "RigidNodes"
                else int(visibility.shape[1])
            )
            original_count = min(original_count, int(visibility.shape[1]))
            before = visibility[:, :original_count].detach()
            self.removed_originals[name] = int(before.any(dim=0).sum().item())
            self._hidden_dynamic_ranges[name] = original_count
            with torch.no_grad():
                visibility[:, :original_count] = False
        self.assert_original_dynamic_hidden()

    def assert_original_dynamic_hidden(self) -> dict[str, int]:
        """Raise if any reconstructed dynamic instance can still be rendered."""
        for name, count in self._hidden_dynamic_ranges.items():
            visibility = self._models[name].instances_fv[:, :count]
            if bool(visibility.any().item()):
                raise RuntimeError(
                    f"original dynamic foreground remains visible in {name}"
                )
        return dict(self.removed_originals)

    def original_gaussian_indices(self, instance_id: int) -> torch.Tensor:
        """Return the exact splat indices assigned to a reconstructed car."""
        instance = int(instance_id)
        if not 0 <= instance < self.original_instance_count:
            raise IndexError(f"original RigidNodes instance does not exist: {instance}")
        return torch.nonzero(
            self.model.point_ids.reshape(-1) == instance, as_tuple=False
        ).reshape(-1)

    def original_gaussian_counts(self) -> dict[int, int]:
        """List reconstructed instance IDs that still own Gaussian splats."""
        ids, counts = torch.unique(self.model.point_ids.reshape(-1), return_counts=True)
        return {
            int(instance): int(count)
            for instance, count in zip(ids.tolist(), counts.tolist())
            if 0 <= int(instance) < self.original_instance_count
        }

    def extract_original(self, instance_id: int) -> GaussianVehicleAsset:
        """Copy one reconstructed car's Gaussians into an editable asset."""
        indices = self.original_gaussian_indices(instance_id)
        if indices.numel() == 0:
            raise ValueError(f"RigidNodes instance {instance_id} has no Gaussians")
        tensors = {
            name: getattr(self.model, name).detach().index_select(0, indices).cpu().clone()
            for name in _PARAMETER_NAMES
        }
        size = self.model.instances_size[int(instance_id)].detach().cpu().tolist()
        return GaussianVehicleAsset(
            means=tensors["_means"],
            features_dc=tensors["_features_dc"],
            features_rest=tensors["_features_rest"],
            opacities=tensors["_opacities"],
            scales=tensors["_scales"],
            quats=tensors["_quats"],
            size_lwh=tuple(float(value) for value in size),
            source=f"checkpoint:RigidNodes:{int(instance_id)}",
        )

    def set_original_visible(self, instance_id: int, visible: bool) -> None:
        """Delete or restore a single original car across the scene frames."""
        self.original_gaussian_indices(instance_id)  # validate the ID
        with torch.no_grad():
            self.model.instances_fv[:, int(instance_id)] = (
                self._original_visibility[:, int(instance_id)] if visible else False
            )

    @staticmethod
    def _replace_parameter(model: Any, name: str, value: torch.Tensor) -> None:
        old = getattr(model, name)
        requires_grad = bool(getattr(old, "requires_grad", False))
        setattr(model, name, nn.Parameter(value, requires_grad=requires_grad))

    def _append_instances(self) -> None:
        model = self.model
        device = model._means.device
        with torch.no_grad():
            for name in _PARAMETER_NAMES:
                appended = torch.cat(
                    [asset.as_tensors()[name].to(device) for asset in self.assets], dim=0
                )
                old = getattr(model, name).detach()
                self._replace_parameter(model, name, torch.cat((old, appended), dim=0))

            point_ids = torch.cat([
                torch.full(
                    (asset.num_gaussians, 1), self.original_instance_count + index,
                    dtype=torch.long, device=device,
                )
                for index, asset in enumerate(self.assets)
            ], dim=0)
            model.point_ids = torch.cat((model.point_ids, point_ids), dim=0)

            frames = int(model.instances_trans.shape[0])
            translations = torch.zeros((frames, self.capacity, 3), dtype=torch.float32, device=device)
            quaternions = torch.zeros((frames, self.capacity, 4), dtype=torch.float32, device=device)
            quaternions[..., 0] = 1.0
            validity = torch.zeros((frames, self.capacity), dtype=torch.bool, device=device)
            sizes = torch.tensor(
                [asset.size_lwh for asset in self.assets], dtype=torch.float32, device=device
            )
            self._replace_parameter(
                model, "instances_trans",
                torch.cat((model.instances_trans.detach(), translations), dim=1),
            )
            self._replace_parameter(
                model, "instances_quats",
                torch.cat((model.instances_quats.detach(), quaternions), dim=1),
            )
            model.instances_fv = torch.cat((model.instances_fv, validity), dim=1)
            model.instances_size = torch.cat((model.instances_size, sizes), dim=0)
