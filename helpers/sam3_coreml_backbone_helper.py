"""Experimental Core ML adapter for SAM3 image backbone.

This module patches ``Sam3Processor.set_image`` without modifying the cloned
SAM3 package. It only runs when explicitly enabled by environment variable and
falls back to the original PyTorch path on any failure.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


_LOGGER = logging.getLogger(__name__)
_ENV_ENABLE = "PLATYPUS_SAM3_COREML_BACKBONE"
_ENV_MODEL_PATH = "PLATYPUS_SAM3_COREML_BACKBONE_MODEL"
_ENV_AUTO_CONVERT = "PLATYPUS_SAM3_COREML_AUTO_CONVERT"
_ENV_RETRY = "PLATYPUS_SAM3_COREML_RETRY"
_ENV_COMPUTE_UNITS = "PLATYPUS_SAM3_COREML_COMPUTE_UNITS"
_DEFAULT_PACKAGE_PATH = Path("checkpoints") / "sam3_backbone.mlpackage"
_DEFAULT_COMPILED_PATH = Path("checkpoints") / "sam3_backbone.mlmodelc"
_DEFAULT_MODEL_PATH = _DEFAULT_PACKAGE_PATH
_DEFAULT_CHECKPOINT_PATH = Path("checkpoints") / "sam3.1_multiplex.pt"


def enabled() -> bool:
    value = os.environ.get(_ENV_ENABLE)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _auto_convert_enabled() -> bool:
    value = os.environ.get(_ENV_AUTO_CONVERT)
    if value is None:
        return True
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _retry_after_failure_enabled() -> bool:
    value = os.environ.get(_ENV_RETRY)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _model_path() -> Path:
    explicit_path = os.environ.get(_ENV_MODEL_PATH)
    if explicit_path:
        return Path(explicit_path)
    if _DEFAULT_COMPILED_PATH.exists():
        return _DEFAULT_COMPILED_PATH
    return _DEFAULT_PACKAGE_PATH


def _metadata_path(model_path: Path) -> Path:
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    if metadata_path.exists() or model_path.suffix != ".mlmodelc":
        return metadata_path
    return model_path.with_suffix(".mlpackage.json")


def _compute_units(model_path: Path):
    import coremltools as ct

    value = os.environ.get(_ENV_COMPUTE_UNITS, "").strip().lower()
    if not value:
        return ct.ComputeUnit.CPU_AND_GPU
    options = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    if value not in options:
        raise ValueError(f"Unsupported {_ENV_COMPUTE_UNITS}: {value!r}")
    return options[value]


def _flatten_tensor_tree(value: Any, prefix: str = "out"):
    from sam3.model.data_misc import NestedTensor

    flat = []
    spec = None
    if isinstance(value, torch.Tensor):
        name = prefix.replace(".", "_").replace("[", "_").replace("]", "")
        flat.append((name, value))
        spec = {"kind": "tensor", "name": name}
    elif isinstance(value, NestedTensor):
        children = {}
        tensor_spec, tensor_flat = _flatten_tensor_tree(value.tensors, f"{prefix}_tensors")
        flat.extend(tensor_flat)
        children["tensors"] = tensor_spec
        if value.mask is None:
            children["mask"] = {"kind": "none"}
        else:
            mask_spec, mask_flat = _flatten_tensor_tree(value.mask, f"{prefix}_mask")
            flat.extend(mask_flat)
            children["mask"] = mask_spec
        spec = {"kind": "nested_tensor", "children": children}
    elif isinstance(value, list):
        items = []
        for index, item in enumerate(value):
            child_spec, child_flat = _flatten_tensor_tree(item, f"{prefix}_{index}")
            flat.extend(child_flat)
            items.append(child_spec)
        spec = {"kind": "list", "items": items}
    elif isinstance(value, tuple):
        items = []
        for index, item in enumerate(value):
            child_spec, child_flat = _flatten_tensor_tree(item, f"{prefix}_{index}")
            flat.extend(child_flat)
            items.append(child_spec)
        spec = {"kind": "tuple", "items": items}
    elif isinstance(value, dict):
        items = {}
        for key in sorted(value):
            child_spec, child_flat = _flatten_tensor_tree(value[key], f"{prefix}_{key}")
            flat.extend(child_flat)
            items[key] = child_spec
        spec = {"kind": "dict", "items": items}
    elif value is None:
        spec = {"kind": "none"}
    else:
        raise TypeError(f"Unsupported SAM3 backbone output type: {type(value)!r}")
    return spec, flat


def _restore_tensor_tree(spec: dict, outputs: dict, device: torch.device):
    from sam3.model.data_misc import NestedTensor

    kind = spec["kind"]
    if kind == "tensor":
        array = _output_array(outputs, spec["name"])
        return torch.from_numpy(np.asarray(array)).to(device)
    if kind == "nested_tensor":
        children = spec["children"]
        return NestedTensor(
            _restore_tensor_tree(children["tensors"], outputs, device),
            _restore_tensor_tree(children["mask"], outputs, device),
        )
    if kind == "list":
        return [_restore_tensor_tree(item, outputs, device) for item in spec["items"]]
    if kind == "tuple":
        return tuple(_restore_tensor_tree(item, outputs, device) for item in spec["items"])
    if kind == "dict":
        return {
            key: _restore_tensor_tree(child, outputs, device)
            for key, child in spec["items"].items()
        }
    if kind == "none":
        return None
    raise ValueError(f"Unknown SAM3 Core ML metadata kind: {kind!r}")


def _output_array(outputs: dict, name: str):
    if name in outputs:
        return outputs[name]
    aliases = {
        # The Core ML optimizer can drop this duplicate output from the
        # compiled model. In SAM3 image backbone it is the low-resolution
        # feature map represented by vision_features.
        "out_backbone_fpn_2": "out_vision_features",
    }
    alias = aliases.get(name)
    if alias is not None and alias in outputs:
        return outputs[alias]
    raise KeyError(name)


class _BackboneExportWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self._spec = None
        self._names = None

    def forward(self, image):
        output = self.backbone.forward_image(image)
        spec, flat = _flatten_tensor_tree(output)
        self._spec = spec
        self._names = [name for name, _ in flat]
        return tuple(tensor for _, tensor in flat)


def _find_vit_trunk(model):
    trunk = getattr(getattr(model.backbone, "vision_backbone", None), "trunk", None)
    if trunk is not None and getattr(trunk, "pos_embed", None) is not None:
        return trunk
    return None


@contextmanager
def _fixed_vit_trace_helpers(model, example_tensor):
    import sam3.model.vitdet as vitdet

    trunk = _find_vit_trunk(model)
    if trunk is None:
        yield
        return

    pos_embed = trunk.pos_embed
    token_count = int(pos_embed.shape[1])
    has_cls_token = bool(getattr(trunk, "pretrain_use_cls_token", False))
    source_tokens = token_count - 1 if has_cls_token else token_count
    source_size = int(round(source_tokens ** 0.5))
    patch = getattr(getattr(trunk, "patch_embed", None), "proj", None)
    if patch is None:
        yield
        return
    patch_h, patch_w = patch.kernel_size
    target_h = int(example_tensor.shape[-2]) // int(patch_h)
    target_w = int(example_tensor.shape[-1]) // int(patch_w)
    tile_abs_pos = bool(getattr(trunk, "tile_abs_pos", True))

    original_get_abs_pos = vitdet.get_abs_pos
    original_window_partition = vitdet.window_partition
    original_window_unpartition = vitdet.window_unpartition
    original_attention_forward = vitdet.Attention.forward
    original_mlp_forward = vitdet.Mlp.forward
    patched_attentions = []

    for module in trunk.modules():
        if isinstance(module, vitdet.Attention) and getattr(module, "freqs_cis", None) is not None:
            module._platypus_freqs_real = module.freqs_cis.real.to(torch.float32)
            module._platypus_freqs_imag = module.freqs_cis.imag.to(torch.float32)
            patched_attentions.append(module)

    def fixed_get_abs_pos(abs_pos, has_cls_token_arg, hw, retain_cls_token=False, tiling=True):
        if has_cls_token_arg:
            cls_pos = abs_pos[:, :1]
            pos = abs_pos[:, 1:]
        else:
            cls_pos = None
            pos = abs_pos

        if source_size != target_h or source_size != target_w:
            new_abs_pos = pos.reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2)
            if tile_abs_pos and tiling:
                repeat_h = target_h // source_size + 1
                repeat_w = target_w // source_size + 1
                new_abs_pos = new_abs_pos.tile([1, 1, repeat_h, repeat_w])[:, :, :target_h, :target_w]
            else:
                new_abs_pos = F.interpolate(
                    new_abs_pos,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                )
            if retain_cls_token:
                return torch.cat(
                    [cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, target_h * target_w, -1)],
                    dim=1,
                )
            return new_abs_pos.permute(0, 2, 3, 1)

        if retain_cls_token:
            return torch.cat([cls_pos, pos], dim=1)
        return pos.reshape(1, target_h, target_w, -1)

    def fixed_window_partition(x, window_size):
        b, _, _, c = x.shape
        pad_h = (window_size - target_h % window_size) % window_size
        pad_w = (window_size - target_w % window_size) % window_size
        padded_h = target_h + pad_h
        padded_w = target_w + pad_w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        x = x.view(b, padded_h // window_size, window_size, padded_w // window_size, window_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
        return windows, (padded_h, padded_w)

    def fixed_window_unpartition(windows, window_size, pad_hw, hw):
        padded_h, padded_w = pad_hw
        channels = int(windows.shape[-1])
        x = windows.reshape(
            -1,
            padded_h // window_size,
            padded_w // window_size,
            window_size,
            window_size,
            channels,
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, padded_h, padded_w, channels)
        return x[:, :target_h, :target_w, :]

    def fixed_attention_forward(self, x):
        def apply_rope_real(q, k):
            if not self.use_rope:
                return q, k
            freqs_real = getattr(self, "_platypus_freqs_real", None)
            freqs_imag = getattr(self, "_platypus_freqs_imag", None)
            if freqs_real is None or freqs_imag is None:
                return self._apply_rope(q, k)

            freqs_real = freqs_real.to(device=q.device, dtype=q.dtype).view(1, 1, freqs_real.shape[0], freqs_real.shape[1])
            freqs_imag = freqs_imag.to(device=q.device, dtype=q.dtype).view(1, 1, freqs_imag.shape[0], freqs_imag.shape[1])

            def rotate(x):
                pairs = x.reshape(*x.shape[:-1], -1, 2)
                real = pairs[..., 0]
                imag = pairs[..., 1]
                out_real = real * freqs_real - imag * freqs_imag
                out_imag = real * freqs_imag + imag * freqs_real
                return torch.stack((out_real, out_imag), dim=-1).flatten(3).type_as(x)

            return rotate(q), rotate(k)

        if x.ndim == 4:
            if getattr(self, "input_size", None) is not None:
                h, w = self.input_size
            else:
                h, w = target_h, target_w
            length = int(h) * int(w)
            qkv = self.qkv(x).reshape(-1, length, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

            q, k = apply_rope_real(q, k)
            if self.use_rel_pos:
                q, k = vitdet.concat_rel_pos(
                    q.flatten(0, 1),
                    k.flatten(0, 1),
                    (int(h), int(w)),
                    (int(h), int(w)),
                    self.rel_pos_h,
                    self.rel_pos_w,
                    rescale=True,
                    relative_coords=self.relative_coords,
                )
                q = q.reshape(-1, self.num_heads, length, q.shape[-1])
                k = k.reshape(-1, self.num_heads, length, k.shape[-1])

            if self.attn_type == vitdet.AttentionType.Vanilla:
                if self.use_fa3:
                    from sam3.perflib.fa3 import flash_attn_func

                    x = flash_attn_func(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                    ).transpose(1, 2)
                else:
                    x = F.scaled_dot_product_attention(q, k, v)
            else:
                raise NotImplementedError

            x = (
                x.reshape(-1, self.num_heads, int(h), int(w), self.head_dim)
                .permute(0, 2, 3, 1, 4)
                .reshape(-1, int(h), int(w), self.num_heads * self.head_dim)
            )
            return self.proj(x)

        return original_attention_forward(self, x)

    def fixed_mlp_forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    vitdet.get_abs_pos = fixed_get_abs_pos
    vitdet.window_partition = fixed_window_partition
    vitdet.window_unpartition = fixed_window_unpartition
    vitdet.Attention.forward = fixed_attention_forward
    vitdet.Mlp.forward = fixed_mlp_forward
    _LOGGER.info(
        "SAM3 Core ML trace uses fixed vit helpers source=%sx%s target=%sx%s tiling=%s",
        source_size,
        source_size,
        target_h,
        target_w,
        tile_abs_pos,
    )
    try:
        yield
    finally:
        vitdet.get_abs_pos = original_get_abs_pos
        vitdet.window_partition = original_window_partition
        vitdet.window_unpartition = original_window_unpartition
        vitdet.Attention.forward = original_attention_forward
        vitdet.Mlp.forward = original_mlp_forward
        for module in patched_attentions:
            if hasattr(module, "_platypus_freqs_real"):
                delattr(module, "_platypus_freqs_real")
            if hasattr(module, "_platypus_freqs_imag"):
                delattr(module, "_platypus_freqs_imag")


class _CoreMLBackboneEngine:
    def __init__(self, model, processor, original_set_image):
        self.model = model
        self.processor = processor
        self.original_set_image = original_set_image
        self.model_path = _model_path()
        self.metadata_path = _metadata_path(self.model_path)
        self._mlmodel = None
        self._metadata = None
        self._failed = False

    def set_image(self, image, state=None):
        if state is None:
            state = {}
        if self._failed and not _retry_after_failure_enabled():
            return self.original_set_image(image, state)
        try:
            return self._set_image_coreml(image, state)
        except Exception:
            self._failed = True
            _LOGGER.exception("SAM3 Core ML backbone failed; falling back to PyTorch set_image")
            return self.original_set_image(image, state)

    def _set_image_coreml(self, image, state):
        import PIL
        from torchvision.transforms import v2

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[:2]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        start = time.perf_counter()
        image_tensor = v2.functional.to_image(image).to(self.processor.device)
        image_tensor = self.processor.transform(image_tensor).unsqueeze(0)
        _LOGGER.info("SAM3 Core ML preprocess elapsed=%.1fms shape=%s", _elapsed_ms(start), tuple(image_tensor.shape))

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.predict(image_tensor)

        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    def predict(self, image_tensor):
        self._ensure_loaded_or_converted(image_tensor)
        start = time.perf_counter()
        outputs = self._mlmodel.predict({"image": image_tensor.detach().cpu().numpy()})
        _LOGGER.info("SAM3 Core ML backbone predict elapsed=%.1fms", _elapsed_ms(start))
        start = time.perf_counter()
        result = _restore_tensor_tree(self._metadata["spec"], outputs, self.processor.device)
        _LOGGER.info("SAM3 Core ML backbone restore elapsed=%.1fms", _elapsed_ms(start))
        return result

    def _ensure_loaded_or_converted(self, example_tensor):
        if self._mlmodel is not None:
            return
        if not self.model_path.exists():
            if not _auto_convert_enabled():
                raise FileNotFoundError(f"SAM3 Core ML model is missing: {self.model_path}")
            self.convert(example_tensor)
        self.load()

    def load(self):
        import coremltools as ct

        start = time.perf_counter()
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        compute_units = _compute_units(self.model_path)
        if self.model_path.suffix == ".mlmodelc":
            self._mlmodel = ct.models.CompiledMLModel(str(self.model_path), compute_units=compute_units)
        else:
            self._mlmodel = ct.models.MLModel(str(self.model_path), compute_units=compute_units)
        _LOGGER.info(
            "SAM3 Core ML backbone loaded elapsed=%.1fms path=%s compute_units=%s",
            _elapsed_ms(start),
            self.model_path,
            compute_units,
        )

    def convert(self, example_tensor):
        import coremltools as ct

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = _BackboneExportWrapper(self.model.backbone).eval()
        start = time.perf_counter()
        with _fixed_vit_trace_helpers(self.model, example_tensor):
            with torch.inference_mode():
                traced = torch.jit.trace(
                    wrapper,
                    example_tensor,
                    strict=False,
                    check_trace=False,
                )
        if not wrapper._names or wrapper._spec is None:
            with torch.inference_mode():
                wrapper(example_tensor)
        _LOGGER.info("SAM3 Core ML trace elapsed=%.1fms outputs=%s", _elapsed_ms(start), wrapper._names)

        start = time.perf_counter()
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=tuple(int(v) for v in example_tensor.shape))],
            outputs=[ct.TensorType(name=name) for name in wrapper._names],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.ALL,
            compute_precision=ct.precision.FLOAT16,
        )
        mlmodel.save(str(self.model_path))
        metadata = {
            "input_shape": [int(v) for v in example_tensor.shape],
            "output_names": wrapper._names,
            "spec": wrapper._spec,
        }
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        _LOGGER.info("SAM3 Core ML convert elapsed=%.1fms path=%s", _elapsed_ms(start), self.model_path)


def install(processor, model) -> bool:
    if not enabled():
        return False
    if getattr(processor, "_platypus_coreml_set_image_installed", False):
        return True
    original_set_image = processor.set_image
    engine = _CoreMLBackboneEngine(model, processor, original_set_image)
    processor.set_image = engine.set_image
    processor._platypus_coreml_backbone_engine = engine
    processor._platypus_coreml_set_image_installed = True
    _LOGGER.info("SAM3 Core ML backbone adapter installed")
    return True


def convert_from_checkpoint(
    checkpoint_path: str | Path = _DEFAULT_CHECKPOINT_PATH,
    model_path: str | Path | None = None,
    device: str = "cpu",
    force: bool = False,
) -> Path:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = Path(checkpoint_path)
    if model_path is not None:
        os.environ[_ENV_MODEL_PATH] = str(model_path)
    target_path = _model_path()
    if target_path.exists() and _metadata_path(target_path).exists() and not force:
        _LOGGER.info("SAM3 Core ML backbone already exists; skipping conversion: %s", target_path)
        return target_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint is missing: {checkpoint_path}")

    start = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    processor = Sam3Processor(model, device=torch.device(device))
    _LOGGER.info("SAM3 model loaded for Core ML conversion elapsed=%.1fms", _elapsed_ms(start))

    engine = _CoreMLBackboneEngine(model, processor, processor.set_image)
    example = torch.zeros((1, 3, processor.resolution, processor.resolution), dtype=torch.float32, device=torch.device(device))
    engine.convert(example)
    return target_path


def main(argv=None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Build experimental SAM3 Core ML backbone package.")
    parser.add_argument("--checkpoint", default=str(_DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = convert_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_path=args.output,
        device=args.device,
        force=args.force,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
