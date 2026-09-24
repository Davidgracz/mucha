from __future__ import annotations

from datetime import datetime
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _bar(value: float, width: int = 18) -> str:
    value = max(0.0, min(1.0, float(value)))
    full = int(round(value * width))
    return "█" * full + "░" * (width - full)


class ConsoleBrainUI:
    """Terminal dashboard for live connectome state."""

    def __init__(self, mode: str = "dashboard", top_neurons: int = 8):
        self.mode = (mode or "dashboard").strip().lower()
        self.top_neurons = max(1, int(top_neurons))
        self.console = Console()
        self.live: Live | None = None
        self.started = False

    def start(self) -> None:
        if self.started or self.mode == "off":
            return
        self.started = True
        if self.mode == "dashboard":
            self.live = Live(
                self._waiting_renderable(),
                console=self.console,
                refresh_per_second=4,
                transient=False,
                screen=False,
            )
            self.live.start(refresh=True)

    def stop(self) -> None:
        if self.live is not None:
            self.live.stop()
            self.live = None
        self.started = False

    def _waiting_renderable(self):
        return Panel("Łączenie z Discordem i inicjalizacja mózgu…", title="🪰 MUCHA BRAIN")

    def update(self, snap: dict[str, Any]) -> None:
        if self.mode == "off":
            return
        if not self.started:
            self.start()

        if self.mode == "simple":
            scores = snap["scores"]
            dominant = max(scores, key=scores.get)
            self.console.print(
                f"[MUCHA] active={snap['diag']['active_abs_gt_0_1']:,} "
                f"mean={snap['diag']['mean_abs']:.4f} "
                f"reward={snap['diag']['reward_trace']:+.3f} "
                f"dominant={dominant}:{scores[dominant]:.2f} "
                f"event={escape(str(snap.get('last_event', '-')))}"
            )
            return

        if self.live is not None:
            self.live.update(self._render(snap), refresh=True)

    def _render(self, snap: dict[str, Any]):
        diag = snap["diag"]
        scores = snap["scores"]
        top = snap["top_neurons"]
        dominant = max(scores, key=scores.get)

        header = Text()
        header.append("🪰 MUCHA BRAIN  ", style="bold")
        header.append("LIVE", style="bold green")
        if snap.get("paused"):
            header.append("  ⏸ PAUSED", style="bold yellow")
        header.append(f"   {datetime.now().strftime('%H:%M:%S')}")
        header.append("\n")
        header.append(str(snap.get("source", "unknown")), style="dim")

        stats = Table(box=box.SIMPLE, show_header=False, expand=True, pad_edge=False)
        stats.add_column("metric")
        stats.add_column("value", justify="right")
        stats.add_row("neurony", f"{diag['neurons']:,}")
        stats.add_row("połączenia", f"{diag['connections']:,}")
        stats.add_row("aktywne |a| > 0.1", f"{diag['active_abs_gt_0_1']:,}")
        stats.add_row("średnia |a|", f"{diag['mean_abs']:.5f}")
        stats.add_row("maks. |a|", f"{diag['max_abs']:.5f}")
        stats.add_row("reward trace", f"{diag['reward_trace']:+.4f}")
        stats.add_row("tick", f"{diag['ticks']:,}")

        actions = Table(box=box.SIMPLE, show_header=True, header_style="bold", expand=True, pad_edge=False)
        actions.add_column("readout", no_wrap=True)
        actions.add_column("aktywność", no_wrap=True)
        actions.add_column("", justify="right", width=6)
        order = ("speak", "react", "voice_join", "voice_move", "voice_leave", "explore", "stay")
        for name in order:
            value = float(scores[name])
            prefix = "▶ " if name == dominant else "  "
            actions.add_row(prefix + name, _bar(value), f"{value:.3f}")

        lang = Table(box=box.SIMPLE, show_header=False, expand=True, pad_edge=False)
        lang.add_column("field", no_wrap=True)
        lang.add_column("value")
        lang.add_row("język", f"{snap['language_tokens']:,} tokenów / {snap['language_unique']:,} unikalnych")
        lang.add_row("gotowa pisać", "TAK" if snap["language_ready"] else "nie")
        lang.add_row("voice", escape(str(snap.get("voice", "poza voice"))))
        lang.add_row("ostatni bodziec", escape(str(snap.get("last_event", "-"))))
        lang.add_row("ostatnia akcja", escape(str(snap.get("last_action", "-"))))

        neurons = Table(box=box.SIMPLE, show_header=True, header_style="bold", expand=True, pad_edge=False)
        neurons.add_column("#", justify="right", width=3)
        neurons.add_column("FlyWire root_id", no_wrap=True)
        neurons.add_column("activation", justify="right")
        neurons.add_column("|a|", justify="right")
        for i, (root_id, activation) in enumerate(top, start=1):
            sign = "+" if activation >= 0 else ""
            neurons.add_row(str(i), str(root_id), f"{sign}{activation:.5f}", f"{abs(activation):.5f}")

        left = Panel(stats, title="Stan mózgu", border_style="bright_blue")
        right = Panel(actions, title=f"Wyjścia • dominant: {dominant}", border_style="bright_magenta")
        info = Panel(lang, title="Środowisko / pamięć", border_style="bright_cyan")
        top_panel = Panel(neurons, title=f"Top {len(top)} aktywnych neuronów", border_style="bright_green")

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=2)
        grid.add_row(left, right)

        footer = Text("Tryby: console_ui.mode = dashboard | simple | off", style="dim")
        return Group(Panel(header, border_style="white"), grid, info, top_panel, footer)
