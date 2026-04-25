"""Render learned Gridworld Q-tables as dependency-free SVG figures."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from q_learning_gridworld import ACTIONS, GridworldConfig, train_once

SETTINGS = [
    ("baseline", {"epsilon": 0.05, "alpha": 0.10, "gamma": 1.00, "initial_q": 0.0}),
    ("more_exploration", {"epsilon": 0.20, "alpha": 0.10, "gamma": 1.00, "initial_q": 0.0}),
    ("optimistic_init", {"epsilon": 0.05, "alpha": 0.10, "gamma": 1.00, "initial_q": 5.0}),
]

ARROWS = {"UP": "^", "DOWN": "v", "LEFT": "<", "RIGHT": ">"}
COLORS = {
    "bg": "#f8f3ea",
    "panel": "#fffaf1",
    "ink": "#172026",
    "muted": "#6c6a62",
    "line": "#d7c7aa",
    "gold": "#e8b84a",
    "bomb": "#cc4b37",
    "best": "#226f68",
    "cell": "#fffdf7",
}


def color_for(value: float, low: float, high: float) -> str:
    if high == low:
        return COLORS["cell"]
    t = (value - low) / (high - low)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        k = t / 0.5
        r = round(204 + (255 - 204) * k)
        g = round(75 + (253 - 75) * k)
        b = round(55 + (247 - 55) * k)
    else:
        k = (t - 0.5) / 0.5
        r = round(255 + (34 - 255) * k)
        g = round(253 + (111 - 253) * k)
        b = round(247 + (104 - 247) * k)
    return f"#{r:02x}{g:02x}{b:02x}"


def text(x: float, y: float, content: str, size: int = 14, fill: str = COLORS["ink"], weight: str = "400", anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Georgia, Times New Roman, serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(content)}</text>'
    )


def render_setting(name: str, params: dict[str, float], output_dir: Path) -> None:
    config = GridworldConfig()
    agent, rewards, steps = train_once(seed=20260425, episodes=1000, slip_probability=0.0, **params)
    values = [value for q in agent.q.values() for value in q.values()]
    low, high = min(values), max(values)

    cell = 132
    margin_x = 42
    margin_y = 110
    width = margin_x * 2 + cell * config.width
    height = margin_y + cell * config.height + 94
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{COLORS["bg"]}"/>',
        f'<rect x="18" y="18" width="{width-36}" height="{height-36}" rx="26" fill="{COLORS["panel"]}" stroke="{COLORS["line"]}"/>',
        text(width / 2, 54, f"Learned Q-table: {name}", 25, COLORS["ink"], "700"),
        text(width / 2, 80, f"epsilon={params['epsilon']}, alpha={params['alpha']}, gamma={params['gamma']}, initial Q={params['initial_q']}", 14, COLORS["muted"]),
    ]

    for r in range(config.height):
        for c in range(config.width):
            x = margin_x + c * cell
            y = margin_y + r * cell
            state = (r, c)
            q = agent.q[state]
            best = max(q, key=q.get)
            max_q = q[best]
            fill = color_for(max_q, low, high)
            if state == config.gold:
                fill = COLORS["gold"]
            elif state == config.bomb:
                fill = COLORS["bomb"]
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-8}" height="{cell-8}" rx="18" fill="{fill}" stroke="{COLORS["line"]}"/>')
            parts.append(text(x + 18, y + 26, f"{state}", 12, COLORS["muted"], anchor="start"))
            if state == config.gold:
                parts.append(text(x + cell / 2 - 4, y + 67, "GOLD", 22, COLORS["ink"], "700"))
            elif state == config.bomb:
                parts.append(text(x + cell / 2 - 4, y + 67, "BOMB", 22, "#fff7ed", "700"))
            else:
                parts.append(text(x + cell / 2 - 4, y + 48, ARROWS[best], 32, COLORS["best"], "700"))
                parts.append(text(x + cell / 2 - 4, y + 72, f"best {max_q:.2f}", 13, COLORS["ink"], "700"))
            small = [
                ("U", q["UP"], x + cell / 2 - 4, y + 96),
                ("D", q["DOWN"], x + cell / 2 - 4, y + 114),
                ("L", q["LEFT"], x + 29, y + 106),
                ("R", q["RIGHT"], x + cell - 37, y + 106),
            ]
            for label, value, tx, ty in small:
                weight = "700" if {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}[label] == best else "400"
                fill_text = "#fff7ed" if state == config.bomb else COLORS["ink"]
                parts.append(text(tx, ty, f"{label}:{value:.1f}", 11, fill_text, weight))

    legend_y = margin_y + cell * config.height + 35
    parts.append(text(margin_x, legend_y, "Each state shows best action and Q-values for U/D/L/R. Green = higher learned value, red = lower value.", 14, COLORS["muted"], anchor="start"))
    parts.append(text(margin_x, legend_y + 26, f"Last 100 episodes: mean reward {sum(rewards[-100:]) / 100:.2f}, mean steps {sum(steps[-100:]) / 100:.2f}", 14, COLORS["muted"], anchor="start"))
    parts.append("</svg>\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"gridworld_q_table_{name}.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    out = Path("docs/images")
    for name, params in SETTINGS:
        render_setting(name, params, out)
    print(f"wrote {len(SETTINGS)} SVG files to {out}")


if __name__ == "__main__":
    main()
