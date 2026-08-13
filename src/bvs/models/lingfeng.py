from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

N_BASE_FILTERS = 16
FEATURE_DIM = 16


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
            in_channels,
            out_channels,
            k_size,
            stride,
            padding=0,
            bias=norm_type is None,
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
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        c = N_BASE_FILTERS
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
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        c = N_BASE_FILTERS
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
        self.seg_logit = GeneralConv3d(
            c, num_classes, k_size=1, norm_type=None, act_type=None
        )

    def forward(
        self, inputs: dict[str, torch.Tensor]
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
    def __init__(self) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear_0 = LinearLayer(N_BASE_FILTERS, FEATURE_DIM * 2, "relu")
        self.linear_1 = LinearLayer(FEATURE_DIM * 2, FEATURE_DIM, None)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = torch.flatten(self.avg_pool(x), 1)
        layer1 = self.linear_0(flat)
        layer2 = self.linear_1(layer1)
        normalized = layer2 / (torch.linalg.norm(layer2, dim=-1, keepdim=True) + 1e-7)
        return flat, layer1, layer2, normalized


class LingfengMRAStudent(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.input_mra_encoder = FeatureEncoder(1)
        self.mask_de_prs = MaskDecoder(num_classes)
        self.metric_prs = MetricLayer()

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.input_mra_encoder(image, self.training)
        probabilities, logits, decoded = self.mask_de_prs(encoded)
        _, _, _, features = self.metric_prs(decoded)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "features": features,
        }


class LingfengLegacyModel(nn.Module):
    """Exact active structure of the archived MultiModalSeg checkpoint."""

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
        self, inputs: dict[str, torch.Tensor], is_training: bool = True, drop_rate: float = 0.3
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
        for index, stage in enumerate(("s1", "s2", "s3", "s4")):
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
