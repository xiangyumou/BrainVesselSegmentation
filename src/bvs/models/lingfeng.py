from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

N_BASE_FILTERS = 16
FEATURE_DIM = 16
STAGES = ("s1", "s2", "s3", "s4")


class InstanceNorm3d(nn.Module):
    def __init__(self, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        variance = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + self.epsilon)


class GeneralConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_size: int = 3,
        stride: int = 1,
        norm_type: str | None = "Ins",
        act_type: str | None = "lrelu",
    ) -> None:
        super().__init__()
        pad = (k_size - 1) // 2
        self.pad_layer: nn.Module | None = nn.ReplicationPad3d(pad) if pad else None
        self.conv = nn.Conv3d(
            in_channels, out_channels, k_size, stride, padding=0, bias=norm_type is None
        )
        self.dropout = nn.Identity()
        self.norm = InstanceNorm3d() if norm_type == "Ins" else nn.Identity()
        self.act_type = act_type

    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        del is_training
        if self.pad_layer is not None:
            x = self.pad_layer(x)
        x = self.norm(self.conv(x))
        if self.act_type == "relu":
            return F.relu(x)
        if self.act_type == "lrelu":
            return F.leaky_relu(x, 0.2)
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_units: int, out_units: int, act_type: str | None) -> None:
        super().__init__()
        self.linear = nn.Linear(in_units, out_units, bias=False)
        self.act_type = act_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.act_type == "relu":
            return F.relu(x)
        if self.act_type == "lrelu":
            return F.leaky_relu(x, 0.2)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = N_BASE_FILTERS) -> None:
        super().__init__()
        c = base_channels
        self.e1_c1 = GeneralConv3d(in_channels, c)
        self.e1_c2 = GeneralConv3d(c, c)
        self.e1_c3 = GeneralConv3d(c, c)
        self.e2_c1 = GeneralConv3d(c, c * 2, stride=2)
        self.e2_c2 = GeneralConv3d(c * 2, c * 2)
        self.e2_c3 = GeneralConv3d(c * 2, c * 2)
        self.e3_c1 = GeneralConv3d(c * 2, c * 4, stride=2)
        self.e3_c2 = GeneralConv3d(c * 4, c * 4)
        self.e3_c3 = GeneralConv3d(c * 4, c * 4)
        self.e4_c1 = GeneralConv3d(c * 4, c * 8, stride=2)
        self.e4_c2 = GeneralConv3d(c * 8, c * 8)
        self.e4_c3 = GeneralConv3d(c * 8, c * 8)

    def forward(
        self, x: torch.Tensor, is_training: bool = True, drop_rate: float = 0.3
    ) -> dict[str, torch.Tensor]:
        del drop_rate
        e1_c1 = self.e1_c1(x, is_training)
        e1 = e1_c1 + self.e1_c3(self.e1_c2(e1_c1, is_training), is_training)
        e2_c1 = self.e2_c1(e1, is_training)
        e2 = e2_c1 + self.e2_c3(self.e2_c2(e2_c1, is_training), is_training)
        e3_c1 = self.e3_c1(e2, is_training)
        e3 = e3_c1 + self.e3_c3(self.e3_c2(e3_c1, is_training), is_training)
        e4_c1 = self.e4_c1(e3, is_training)
        e4 = e4_c1 + self.e4_c3(self.e4_c2(e4_c1, is_training), is_training)
        return {"s1": e1, "s2": e2, "s3": e3, "s4": e4}


class MaskDecoder(nn.Module):
    def __init__(self, num_classes: int = 2, base_channels: int = N_BASE_FILTERS) -> None:
        super().__init__()
        c = base_channels
        self.d3_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d3_c1 = GeneralConv3d(c * 8, c * 4)
        self.d3_c2 = GeneralConv3d(c * 8, c * 4)
        self.d3_out = GeneralConv3d(c * 4, c * 4, k_size=1)
        self.d2_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d2_c1 = GeneralConv3d(c * 4, c * 2)
        self.d2_c2 = GeneralConv3d(c * 4, c * 2)
        self.d2_out = GeneralConv3d(c * 2, c * 2, k_size=1)
        self.d1_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.d1_c1 = GeneralConv3d(c * 2, c)
        self.d1_c2 = GeneralConv3d(c * 2, c)
        self.d1_out = GeneralConv3d(c, c, k_size=1)
        self.seg_logit = GeneralConv3d(c, num_classes, k_size=1, norm_type=None, act_type=None)

    def forward(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d3 = self.d3_c1(self.d3_up(inputs["s4"]))
        d3 = self.d3_out(self.d3_c2(torch.cat((d3, inputs["s3"]), dim=1)))
        d2 = self.d2_c1(self.d2_up(d3))
        d2 = self.d2_out(self.d2_c2(torch.cat((d2, inputs["s2"]), dim=1)))
        d1 = self.d1_c1(self.d1_up(d2))
        d1 = self.d1_out(self.d1_c2(torch.cat((d1, inputs["s1"]), dim=1)))
        logits = self.seg_logit(d1)
        return torch.softmax(logits, dim=1), logits, d1


class MetricLayer(nn.Module):
    def __init__(
        self, base_channels: int = N_BASE_FILTERS, feature_dim: int | None = None
    ) -> None:
        super().__init__()
        feature_dim = base_channels if feature_dim is None else feature_dim
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear_0 = LinearLayer(base_channels, feature_dim * 2, "relu")
        self.linear_1 = LinearLayer(feature_dim * 2, feature_dim, None)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = torch.flatten(self.avg_pool(x), 1)
        layer1 = self.linear_0(flat)
        layer2 = self.linear_1(layer1)
        normalized = layer2 / (torch.linalg.norm(layer2, dim=-1, keepdim=True) + 1e-7)
        return flat, layer1, layer2, normalized


def _branch_output(
    probabilities: torch.Tensor,
    logits: torch.Tensor,
    decoded: torch.Tensor,
    metric: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "logits": logits,
        "probabilities": probabilities,
        "decoder_feature": decoded,
        "metric_feature": metric[-1],
        # Backward-compatible alias used by the original bvs API.
        "features": metric[-1],
    }


class ConfigurableLingfengModel(nn.Module):
    """Dynamic Lingfeng teacher/student network.

    The four-modality ``mra,t1,t2,pd`` instance has the same tensor shapes and
    operation order as the archived ``MultiModalSeg`` implementation.
    """

    def __init__(
        self,
        modalities: Sequence[str],
        student_modality: str,
        in_channels: Mapping[str, int],
        num_classes: int,
        base_channels: int = N_BASE_FILTERS,
    ) -> None:
        super().__init__()
        ordered = tuple(str(item).lower() for item in modalities)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("modalities must be a non-empty list of unique names")
        if student_modality not in ordered:
            raise ValueError(f"student_modality '{student_modality}' is not in modalities")
        unknown_channels = set(in_channels) - set(ordered)
        if unknown_channels:
            raise ValueError(f"in_channels contains unknown modalities: {sorted(unknown_channels)}")
        missing_channels = set(ordered) - set(in_channels)
        if missing_channels:
            raise ValueError(f"in_channels is missing modalities: {sorted(missing_channels)}")
        if num_classes < 2 or base_channels < 1:
            raise ValueError("num_classes must be >= 2 and base_channels must be >= 1")

        self.modalities = ordered
        self.student_modality = student_modality
        self.in_channels = {name: int(in_channels[name]) for name in ordered}
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.encoders = nn.ModuleDict(
            {
                name: FeatureEncoder(self.in_channels[name], self.base_channels)
                for name in ordered
            }
        )
        count = len(ordered)
        self.attention = nn.ModuleDict()
        self.fusion = nn.ModuleDict()
        for index, stage in enumerate(STAGES):
            channels = self.base_channels * (2**index)
            self.attention[stage] = GeneralConv3d(
                channels * count, count, k_size=1, norm_type=None, act_type=None
            )
            self.fusion[stage] = GeneralConv3d(
                channels * count, channels, k_size=1
            )
        self.teacher_decoder = MaskDecoder(self.num_classes, self.base_channels)
        self.teacher_metric = MetricLayer(self.base_channels)
        self.student_decoder = MaskDecoder(self.num_classes, self.base_channels)
        self.student_metric = MetricLayer(self.base_channels)

    @property
    def model_spec(self) -> dict[str, object]:
        return {
            "name": "configurable_lingfeng",
            "modalities": list(self.modalities),
            "student_modality": self.student_modality,
            "in_channels": dict(self.in_channels),
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
        }

    def _student_tensor(
        self, inputs: Mapping[str, torch.Tensor] | torch.Tensor
    ) -> torch.Tensor:
        if torch.is_tensor(inputs):
            return inputs
        if self.student_modality not in inputs:
            raise KeyError(f"Missing student modality '{self.student_modality}'")
        return inputs[self.student_modality]

    def forward_student(
        self, inputs: Mapping[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoders[self.student_modality](
            self._student_tensor(inputs), self.training
        )
        probabilities, logits, decoded = self.student_decoder(encoded)
        return _branch_output(
            probabilities, logits, decoded, self.student_metric(decoded)
        )

    def forward_teacher(
        self, inputs: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        missing = [name for name in self.modalities if name not in inputs]
        if missing:
            raise KeyError(f"Missing teacher modalities: {missing}")
        encoded = {
            name: self.encoders[name](inputs[name], self.training)
            for name in self.modalities
        }
        shared: dict[str, torch.Tensor] = {}
        for stage in STAGES:
            stage_features = [encoded[name][stage] for name in self.modalities]
            weights = torch.sigmoid(self.attention[stage](torch.cat(stage_features, dim=1)))
            weighted = [
                feature * weights[:, index : index + 1]
                for index, feature in enumerate(stage_features)
            ]
            shared[stage] = self.fusion[stage](torch.cat(weighted, dim=1))
        probabilities, logits, decoded = self.teacher_decoder(shared)
        output = _branch_output(
            probabilities, logits, decoded, self.teacher_metric(decoded)
        )
        output["attention_features"] = shared
        return output

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor] | torch.Tensor,
        branch: str = "student",
    ) -> dict[str, torch.Tensor] | dict[str, dict[str, torch.Tensor]]:
        if branch == "student":
            return self.forward_student(inputs)
        if branch == "teacher":
            if not isinstance(inputs, Mapping):
                raise TypeError("Teacher branch requires a modality mapping")
            return self.forward_teacher(inputs)
        if branch == "both":
            if not isinstance(inputs, Mapping):
                raise TypeError("Both branches require a modality mapping")
            # Encoder sharing is intentionally not introduced here: legacy forward
            # computes the branches in this order and equivalence is the priority.
            return {
                "teacher": self.forward_teacher(inputs),
                "student": self.forward_student(inputs),
            }
        raise ValueError("branch must be one of: teacher, student, both")


class StudentInferenceView(nn.Module):
    """A parameter-sharing deployment view of a configurable model."""

    def __init__(self, model: ConfigurableLingfengModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model.forward_student(image)


class LingfengMRAStudent(ConfigurableLingfengModel):
    """Compatibility constructor for the original public single-MRA API."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__(["mra"], "mra", {"mra": 1}, num_classes, N_BASE_FILTERS)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_student(image)


class LingfengLegacyModel(nn.Module):
    """Exact active structure and key names of the archived MultiModalSeg."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        c = N_BASE_FILTERS
        self.input_mra_encoder = FeatureEncoder(1)
        self.ce_t1 = FeatureEncoder(1)
        self.ce_t2 = FeatureEncoder(1)
        self.ce_pd = FeatureEncoder(1)
        self.att_c1 = GeneralConv3d(c * 4, 4, k_size=1, norm_type=None, act_type=None)
        self.att_c2 = GeneralConv3d(c * 8, 4, k_size=1, norm_type=None, act_type=None)
        self.att_c3 = GeneralConv3d(c * 16, 4, k_size=1, norm_type=None, act_type=None)
        self.att_c4 = GeneralConv3d(c * 32, 4, k_size=1, norm_type=None, act_type=None)
        self.fusion_c1 = GeneralConv3d(c * 4, c, k_size=1)
        self.fusion_c2 = GeneralConv3d(c * 8, c * 2, k_size=1)
        self.fusion_c3 = GeneralConv3d(c * 16, c * 4, k_size=1)
        self.fusion_c4 = GeneralConv3d(c * 32, c * 8, k_size=1)
        self.mask_de_prs = MaskDecoder(num_classes)
        self.mask_de_abs = MaskDecoder(num_classes)
        self.metric_prs = MetricLayer()
        self.metric_abs = MetricLayer()

    def forward(
        self, inputs: Mapping[str, torch.Tensor], is_training: bool = True, drop_rate: float = 0.3
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        mra = self.input_mra_encoder(inputs["source"], is_training, drop_rate)
        modalities = [
            mra,
            self.ce_t1(inputs["input_t1"], is_training, drop_rate),
            self.ce_t2(inputs["input_t2"], is_training, drop_rate),
            self.ce_pd(inputs["input_pd"], is_training, drop_rate),
        ]
        shared: dict[str, torch.Tensor] = {}
        attention = (self.att_c1, self.att_c2, self.att_c3, self.att_c4)
        fusion = (self.fusion_c1, self.fusion_c2, self.fusion_c3, self.fusion_c4)
        for index, stage in enumerate(STAGES):
            stage_features = [item[stage] for item in modalities]
            weights = torch.sigmoid(attention[index](torch.cat(stage_features, dim=1)))
            weighted = [
                feature * weights[:, position : position + 1]
                for position, feature in enumerate(stage_features)
            ]
            shared[stage] = fusion[index](torch.cat(weighted, dim=1))

        pred_prs, logit_prs, decoded_prs = self.mask_de_prs(mra)
        pred_abs, logit_abs, decoded_abs = self.mask_de_abs(shared)
        *_, feat_prs = self.metric_prs(decoded_prs)
        *_, feat_abs = self.metric_abs(decoded_abs)
        return {
            "feature_mra": mra,
            "seg_pred_prs": pred_prs,
            "seg_logit_prs": logit_prs,
            "seg_pred_abs": pred_abs,
            "seg_logit_abs": logit_abs,
            "d1_out_prs": decoded_prs,
            "d1_out_abs": decoded_abs,
            "feat_prs": feat_prs,
            "feat_abs": feat_abs,
        }
