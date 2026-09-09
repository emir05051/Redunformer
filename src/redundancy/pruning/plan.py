from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from redundancy.hooks import HeadPruningHook
from redundancy.utils import get_model_metadata, get_output_projection


@dataclass
class PruningPlan:
    method: str
    model_name: str
    requested_ratio: float
    actual_ratio: float
    seed: int
    eligible_layers: list[int]
    selected_heads: dict[int, list[int]]
    num_heads: int
    head_dim: int
    model_revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported pruning plan schema: {self.schema_version}")
        if not 0 <= self.requested_ratio <= 1:
            raise ValueError("requested_ratio must be between 0 and 1")
        eligible = set(self.eligible_layers)
        if set(self.selected_heads) - eligible:
            raise ValueError("selected_heads contains a layer that is not eligible")
        for layer, heads in self.selected_heads.items():
            if len(heads) != len(set(heads)):
                raise ValueError(f"Layer {layer} contains duplicate head indices")
            if any(head < 0 or head >= self.num_heads for head in heads):
                raise ValueError(f"Layer {layer} contains an invalid head index")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["selected_heads"] = {
            str(layer): heads for layer, heads in sorted(self.selected_heads.items())
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PruningPlan":
        normalized = dict(data)
        normalized["selected_heads"] = {
            int(layer): list(heads) for layer, heads in data["selected_heads"].items()
        }
        plan = cls(**normalized)
        plan.validate()
        return plan

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PruningPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def apply_pruning_plan(model, plan: PruningPlan):
    plan.validate()
    model_metadata = get_model_metadata(model)
    if model_metadata.num_heads != plan.num_heads or model_metadata.head_dim != plan.head_dim:
        raise ValueError("Pruning plan dimensions do not match the loaded model")
    if not set(plan.eligible_layers).issubset(set(model_metadata.eligible_layers)):
        raise ValueError("Pruning plan targets layers that are not eligible in the loaded model")

    handles = []
    for layer, heads in sorted(plan.selected_heads.items()):
        if not heads:
            continue
        projection = get_output_projection(model, layer, model_metadata.model_type)
        handles.append(
            projection.register_forward_pre_hook(HeadPruningHook(heads, plan.head_dim))
        )
    return handles
