from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD

from bvs.training.losses import (
    CombinedSegmentationLoss,
    MetricContrastiveLoss,
    TemperatureKLLoss,
)
from bvs_nnunet.networks import KDNetwork, StudentNetwork, TeacherNetwork
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


def _unwrap_network(network: nn.Module) -> nn.Module:
    if isinstance(network, DistributedDataParallel):
        network = network.module
    if isinstance(network, OptimizedModule):
        network = network._orig_mod
    return network


class _BVSBaseTrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        del enabled

    def _build_loss(self) -> CombinedSegmentationLoss:
        if self.label_manager.has_regions or self.label_manager.has_ignore_label:
            raise ValueError(
                "BVS Lingfeng trainers require ordinary class labels without ignore regions"
            )
        return CombinedSegmentationLoss(
            ce_weight=1.0,
            dice_weight=1.0,
            dice_variant="foreground",
            num_classes=self.label_manager.num_segmentation_heads,
        )

    def configure_optimizers(self) -> tuple[torch.optim.Optimizer, PolyLRScheduler]:
        trainable = [parameter for parameter in self.network.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("The BVS network has no trainable parameters")
        optimizer = SGD(
            trainable,
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)


class nnUNetTrainerBVSTeacher(_BVSBaseTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        del plans_manager, configuration_manager, enable_deep_supervision
        return TeacherNetwork(num_input_channels, num_output_channels)


class nnUNetTrainerBVSStudent(_BVSBaseTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        del plans_manager, configuration_manager, enable_deep_supervision
        return StudentNetwork(num_input_channels, num_output_channels)


class nnUNetTrainerBVSKD(_BVSBaseTrainer):
    kd_weight = 0.5
    feature_weight = 0.5
    temperature = 10.0
    margin = 1.0
    projection_dim = 8

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.logit_distillation_loss = TemperatureKLLoss(self.temperature)
        self.feature_distillation_loss = MetricContrastiveLoss(self.margin)
        self._teacher_checkpoint_loaded = False

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        del plans_manager, configuration_manager, enable_deep_supervision
        return KDNetwork(num_input_channels, num_output_channels, projection_dim=8)

    def initialize(self) -> None:
        super().initialize()
        checkpoint_value = os.environ.get("BVS_TEACHER_CHECKPOINT")
        if not checkpoint_value:
            raise FileNotFoundError(
                "nnUNetTrainerBVSKD requires BVS_TEACHER_CHECKPOINT to point to "
                "checkpoint_best.pth from nnUNetTrainerBVSTeacher"
            )
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"BVS_TEACHER_CHECKPOINT does not exist: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Teacher checkpoint must be a mapping")
        network = _unwrap_network(self.network)
        if not isinstance(network, KDNetwork):
            raise TypeError(f"Expected KDNetwork, got {type(network).__name__}")
        network.load_teacher_checkpoint(checkpoint)
        self._teacher_checkpoint_loaded = True
        self.print_to_log_file(f"Loaded frozen KD teacher from {checkpoint_path}")

    def _kd_objective(
        self, output: dict[str, torch.Tensor], target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        segmentation = self.loss(output["student_logits"], target)
        logit_distillation = self.logit_distillation_loss(
            output["student_logits"], output["teacher_logits"]
        )
        feature_distillation = self.feature_distillation_loss(
            output["student_features"], output["teacher_features"]
        )
        total = (
            segmentation
            + self.kd_weight * logit_distillation
            + self.feature_weight * feature_distillation
        )
        return total, {
            "segmentation": segmentation,
            "logit_distillation": logit_distillation,
            "feature_distillation": feature_distillation,
        }

    def train_step(self, batch: dict[str, Any]) -> dict[str, Any]:
        if not self._teacher_checkpoint_loaded:
            raise RuntimeError("KD teacher checkpoint was not loaded")
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            raise TypeError("BVS trainers do not support deep-supervision target lists")
        target = target.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with amp_context:
            output = self.network(data, return_distillation=True)
            if not isinstance(output, dict):
                raise TypeError("KDNetwork did not return distillation tensors")
            loss, components = self._kd_objective(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.network.parameters()
                    if parameter.requires_grad
                ],
                12,
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.network.parameters()
                    if parameter.requires_grad
                ],
                12,
            )
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            **{
                name: value.detach().cpu().numpy()
                for name, value in components.items()
            },
        }
