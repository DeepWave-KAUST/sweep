from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    """Structured description of a wavefield name exposed by an equation."""

    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    supports_source: bool = False
    supports_receiver: bool = False
    internal: bool = False
    boundary_related: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Structured description of a model parameter consumed by an equation."""

    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    unit: str = ""
    required: bool = True


def build_field_index(field_specs):
    index = {}
    for spec in field_specs:
        index[spec.name] = spec
        for alias in spec.aliases:
            index[alias] = spec
    return index


def build_model_index(model_specs):
    index = {}
    for spec in model_specs:
        index[spec.name] = spec
        for alias in spec.aliases:
            index[alias] = spec
    return index


def ensure_field_specs(wavefield_names, field_specs):
    if field_specs is None:
        field_specs = []

    normalized = []
    seen = set()
    for spec in field_specs:
        normalized.append(spec)
        seen.add(spec.name)

    for name in wavefield_names:
        if name not in seen:
            normalized.append(
                FieldSpec(
                    name=name,
                    description=f"Wavefield `{name}`.",
                    internal=True,
                )
            )

    return normalized


def ensure_model_specs(model_names, model_specs):
    if model_specs is None:
        model_specs = []

    normalized = []
    seen = set()
    for spec in model_specs:
        normalized.append(spec)
        seen.add(spec.name)

    for name in model_names:
        if name not in seen:
            normalized.append(
                ModelSpec(
                    name=name,
                    description=f"Model parameter `{name}`.",
                )
            )

    return normalized


def available_role_specs(field_specs, role):
    attr = "supports_source" if role == "source" else "supports_receiver"
    return [spec for spec in field_specs if getattr(spec, attr)]


def format_field_specs(field_specs):
    lines = []
    for spec in field_specs:
        alias_text = f" aliases: {', '.join(spec.aliases)}." if spec.aliases else ""
        desc = spec.description or f"Wavefield `{spec.name}`."
        lines.append(f"- {spec.name}:{alias_text} {desc}".strip())
    return "\n".join(lines)


def format_model_specs(model_specs):
    lines = []
    for spec in model_specs:
        alias_text = f" aliases: {', '.join(spec.aliases)}." if spec.aliases else ""
        unit_text = f" Units: {spec.unit}." if spec.unit else ""
        req_text = "" if spec.required else " Optional."
        desc = spec.description or f"Model parameter `{spec.name}`."
        lines.append(f"- {spec.name}:{alias_text} {desc}{unit_text}{req_text}".strip())
    return "\n".join(lines)
