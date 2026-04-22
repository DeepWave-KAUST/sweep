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


def build_field_index(field_specs):
    index = {}
    for spec in field_specs:
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
