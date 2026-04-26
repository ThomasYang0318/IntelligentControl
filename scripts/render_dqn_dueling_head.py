"""Render the Dueling DQN head diagram from the shared architecture spec."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from dqn_architecture_spec import (
    ADVANTAGE_STREAM,
    CNN_BACKBONE,
    COMBINE_FORMULA,
    INPUT_SHAPE,
    VALUE_STREAM,
)

COLORS = {
    "bg0": "#f8f3ea",
    "bg1": "#eaf3ef",
    "panel": "#fffaf1",
    "ink": "#172026",
    "muted": "#6c6a62",
    "line": "#d7c7aa",
    "dark": "#172026",
    "conv0": "#2f6f68",
    "conv1": "#77b7a5",
    "value0": "#c0842f",
    "value1": "#f1c76b",
    "adv0": "#b94f45",
    "adv1": "#e39b8f",
    "shared": "#40515b",
}


def layer_label(layer: dict[str, object]) -> str:
    if layer["type"] == "Conv2D":
        return f"Conv2D({layer['filters']}, {layer['kernel']}x{layer['kernel']}, stride {layer['stride']})"
    if layer["type"] == "Dense":
        suffix = f": {layer['name']}" if "name" in layer else ""
        return f"Dense({layer['units']}, {layer['activation']}){suffix}"
    return str(layer["type"])


def stream_labels(stream: list[dict[str, object]]) -> list[str]:
    labels: list[str] = []
    for layer in stream:
        if "preceded_by" in layer:
            labels.append(layer_label(layer["preceded_by"]))
        labels.append(layer_label(layer))
    return labels


def text(x: float, y: float, content: str, cls: str, anchor: str = "middle") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{escape(content)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 18) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" filter="url(#shadow)"/>'


def arrow(path: str) -> str:
    return f'<path class="arrow" d="{path}"/>'


def gradient(id_: str, start: str, end: str) -> str:
    return (
        f'<linearGradient id="{id_}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{start}"/>'
        f'<stop offset="1" stop-color="{end}"/>'
        '</linearGradient>'
    )


def render(output: Path) -> None:
    backbone_labels = [layer_label(layer) for layer in CNN_BACKBONE]
    value_labels = stream_labels(VALUE_STREAM)
    advantage_labels = stream_labels(ADVANTAGE_STREAM)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="520" viewBox="0 0 980 520">',
        '<defs>',
        gradient("bg", COLORS["bg0"], COLORS["bg1"]),
        gradient("conv", COLORS["conv0"], COLORS["conv1"]),
        gradient("value", COLORS["value0"], COLORS["value1"]),
        gradient("adv", COLORS["adv0"], COLORS["adv1"]),
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#172026" flood-opacity="0.14"/></filter>',
        '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M2,2 L10,6 L2,10 Z" fill="#475158"/></marker>',
        '<style>',
        ".title { font: 700 28px Georgia, 'Times New Roman', serif; fill: #172026; }",
        ".subtitle { font: 400 15px Georgia, 'Times New Roman', serif; fill: #6c6a62; }",
        ".box-title { font: 700 16px Georgia, 'Times New Roman', serif; fill: #ffffff; }",
        ".box-sub { font: 400 12px Georgia, 'Times New Roman', serif; fill: #fffaf1; }",
        ".dark-title { font: 700 16px Georgia, 'Times New Roman', serif; fill: #172026; }",
        ".dark-sub { font: 400 12px Georgia, 'Times New Roman', serif; fill: #475158; }",
        ".arrow { stroke: #475158; stroke-width: 2.5; fill: none; marker-end: url(#arrow); }",
        '</style>',
        '</defs>',
        '<rect width="980" height="520" fill="url(#bg)"/>',
        f'<rect x="24" y="24" width="932" height="472" rx="28" fill="{COLORS["panel"]}" stroke="{COLORS["line"]}"/>',
        text(490, 66, "Dueling DQN Head for Atari Breakout", "title"),
        text(490, 92, "Generated from scripts/dqn_architecture_spec.py, which is shared by code and figures", "subtitle"),
        rect(64, 170, 132, 88, COLORS["dark"]),
        text(130, 205, "Input", "box-title"),
        text(130, 229, " x ".join(map(str, INPUT_SHAPE)), "box-sub"),
        arrow("M196 214 H244"),
        rect(250, 136, 190, 156, "url(#conv)"),
        text(345, 169, "CNN Backbone", "box-title"),
    ]
    for idx, label in enumerate(backbone_labels):
        parts.append(text(345, 194 + idx * 20, label, "box-sub"))

    parts.extend([
        arrow("M440 214 H488"),
        rect(494, 170, 142, 88, COLORS["shared"]),
        text(565, 205, "Shared FC", "box-title"),
        text(565, 229, backbone_labels[-1], "box-sub"),
        arrow("M636 214 C675 214 675 150 714 150"),
        arrow("M636 214 C675 214 675 300 714 300"),
        rect(720, 108, 190, 92, "url(#value)"),
        text(815, 142, "Value Stream", "box-title"),
        text(815, 166, " -> ".join(value_labels), "box-sub"),
        rect(720, 258, 190, 92, "url(#adv)"),
        text(815, 292, "Advantage Stream", "box-title"),
        text(815, 316, " -> ".join(advantage_labels), "box-sub"),
        arrow("M815 200 V374"),
        arrow("M815 350 V374"),
        '<rect x="678" y="380" width="260" height="74" rx="18" fill="#f8f3ea" stroke="#d7c7aa" filter="url(#shadow)"/>',
        text(808, 410, "Combine", "dark-title"),
        text(808, 435, COMBINE_FORMULA, "dark-sub"),
        '<rect x="78" y="356" width="478" height="84" rx="18" fill="#ffffff" stroke="#d7c7aa"/>',
        text(104, 389, "Why this change?", "dark-title", "start"),
        text(104, 416, "It separates state quality V(s) from action-specific advantage A(s,a).", "dark-sub", "start"),
        text(104, 438, "Only the network head changes, keeping the comparison fair.", "dark-sub", "start"),
        '</svg>\n',
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    output = Path("docs/images/dqn_dueling_head.svg")
    render(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
