import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

from .theme import COLORS as C


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    unit: str
    low: float
    high: float
    default: float
    step: float
    fmt: str = "{:g}"


class SliderRow(tk.Frame):
    def __init__(self, parent, spec: Spec, on_change=None):
        super().__init__(parent, bg=C["panel"])
        self.spec = spec
        self.on_change = on_change
        self._value = spec.default
        self._silent = False
        self.text = tk.StringVar(value=spec.fmt.format(spec.default))

        head = tk.Frame(self, bg=C["panel"])
        head.pack(fill="x")
        ttk.Label(head, text=spec.label).pack(side="left")
        ttk.Label(head, text=spec.unit, style="Muted.TLabel").pack(side="right", padx=(4, 0))
        self.entry = ttk.Entry(head, textvariable=self.text, width=8, justify="right")
        self.entry.pack(side="right")
        self.entry.bind("<Return>", self._from_entry)
        self.entry.bind("<FocusOut>", self._from_entry)

        self.scale = ttk.Scale(self, from_=spec.low, to=spec.high, command=self._from_scale)
        self.scale.set(spec.default)
        self.scale.pack(fill="x", pady=(4, 0))

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._silent = True
        self._value = self._snap(value)
        self.scale.set(self._value)
        self.text.set(self.spec.fmt.format(self._value))
        self._silent = False

    def _snap(self, value: float) -> float:
        s = self.spec
        value = min(max(value, s.low), s.high)
        return round(round((value - s.low) / s.step) * s.step + s.low, 6)

    def _from_scale(self, raw):
        if self._silent:
            return
        value = self._snap(float(raw))
        if value != self._value:
            self._value = value
            self.text.set(self.spec.fmt.format(value))
            self._notify()

    def _from_entry(self, _event=None):
        try:
            value = self._snap(float(self.text.get().replace(",", "")))
        except ValueError:
            value = self._value
        changed = value != self._value
        self.set(value)
        if changed:
            self._notify()

    def _notify(self):
        if self.on_change:
            self.on_change()


class StatCard(tk.Frame):
    def __init__(self, parent, label: str, unit: str = ""):
        super().__init__(parent, bg=C["card"], highlightthickness=1, highlightbackground=C["line"])
        self.value = tk.StringVar(value="—")
        ttk.Label(self, text=label, style="CardMuted.TLabel").pack(anchor="w", padx=10, pady=(8, 0))
        row = tk.Frame(self, bg=C["card"])
        row.pack(anchor="w", padx=10, pady=(0, 8))
        ttk.Label(row, textvariable=self.value, style="CardValue.TLabel").pack(side="left")
        ttk.Label(row, text=unit, style="CardMuted.TLabel").pack(side="left", padx=(4, 0), pady=(5, 0))


class Segmented(tk.Frame):
    def __init__(self, parent, options, variable, command=None, bg=None):
        super().__init__(parent, bg=bg or C["bg"])
        self.buttons = {}
        for value, text in options:
            btn = ttk.Radiobutton(self, text=text, value=value, variable=variable,
                                  command=command, style="Seg.Toolbutton")
            btn.pack(side="left")
            self.buttons[value] = btn

    def enable(self, value, flag: bool) -> None:
        self.buttons[value].state(["!disabled"] if flag else ["disabled"])


class Collapsible(tk.Frame):
    def __init__(self, parent, title: str, opened: bool = False):
        super().__init__(parent, bg=C["panel"])
        self.title = title
        self.opened = opened
        self.toggle_btn = ttk.Button(self, style="Quiet.TButton", command=self.toggle)
        self.toggle_btn.pack(anchor="w")
        self.body = tk.Frame(self, bg=C["panel"])
        self._sync()

    def toggle(self):
        self.opened = not self.opened
        self._sync()

    def _sync(self):
        self.toggle_btn.configure(text=("−  " if self.opened else "+  ") + self.title)
        if self.opened:
            self.body.pack(fill="x", pady=(4, 0))
        else:
            self.body.pack_forget()
