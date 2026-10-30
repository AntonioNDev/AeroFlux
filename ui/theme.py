from tkinter import font as tkfont
from tkinter import ttk

COLORS = {
    "bg": "#101216",
    "panel": "#171a20",
    "card": "#1e222a",
    "line": "#2a2f39",
    "text": "#e8ebf1",
    "muted": "#8d95a5",
    "accent": "#4f8cff",
    "accent_hover": "#6a9fff",
    "warn": "#f0b44c",
    "danger": "#ff6b6b",
    "flow": "#7fd0ff",
    "flow_hot": "#e6f6ff",
}

FAMILIES = ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Ubuntu", "DejaVu Sans")


def apply(root) -> dict:
    c = COLORS
    available = set(tkfont.families(root))
    family = next((f for f in FAMILIES if f in available), "Helvetica")
    fonts = {
        "base": (family, 10),
        "small": (family, 9),
        "bold": (family, 10, "bold"),
        "title": (family, 15, "bold"),
        "value": (family, 15, "bold"),
    }

    root.configure(bg=c["bg"])
    root.option_add("*Font", fonts["base"])
    root.option_add("*TCombobox*Listbox.background", c["card"])
    root.option_add("*TCombobox*Listbox.foreground", c["text"])
    root.option_add("*TCombobox*Listbox.selectBackground", c["accent"])
    root.option_add("*Menu.background", c["card"])
    root.option_add("*Menu.foreground", c["text"])

    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=c["panel"], foreground=c["text"], bordercolor=c["line"],
                lightcolor=c["panel"], darkcolor=c["panel"], focuscolor=c["panel"], font=fonts["base"])

    s.configure("TFrame", background=c["panel"])
    s.configure("TLabel", background=c["panel"], foreground=c["text"])
    s.configure("Muted.TLabel", foreground=c["muted"], font=fonts["small"])
    s.configure("Title.TLabel", font=fonts["title"])
    s.configure("Section.TLabel", foreground=c["muted"], font=fonts["small"])
    s.configure("Card.TLabel", background=c["card"])
    s.configure("CardMuted.TLabel", background=c["card"], foreground=c["muted"], font=fonts["small"])
    s.configure("CardValue.TLabel", background=c["card"], font=fonts["value"])
    s.configure("Stage.TFrame", background=c["bg"])
    s.configure("Stage.TLabel", background=c["bg"], foreground=c["muted"], font=fonts["small"])
    s.configure("Warn.TLabel", background=c["panel"], foreground=c["warn"], font=fonts["small"])

    s.configure("TButton", background=c["card"], foreground=c["text"], bordercolor=c["line"],
                lightcolor=c["card"], darkcolor=c["card"], padding=(12, 7), relief="flat")
    s.map("TButton", background=[("disabled", c["panel"]), ("pressed", c["line"]), ("active", c["line"])],
          foreground=[("disabled", c["muted"])])
    s.configure("Card.TButton", background=c["line"], bordercolor=c["line"], lightcolor=c["line"],
                darkcolor=c["line"], padding=(12, 6))
    s.map("Card.TButton", background=[("active", c["accent"]), ("pressed", c["accent"])],
          foreground=[("active", "#ffffff"), ("disabled", c["muted"])])
    s.configure("Accent.TButton", background=c["accent"], foreground="#ffffff", bordercolor=c["accent"],
                lightcolor=c["accent"], darkcolor=c["accent"], padding=(12, 10), font=fonts["bold"])
    s.map("Accent.TButton", background=[("disabled", c["line"]), ("pressed", c["accent"]), ("active", c["accent_hover"])],
          foreground=[("disabled", c["muted"])])
    s.configure("Quiet.TButton", background=c["panel"], foreground=c["muted"], padding=(6, 4), bordercolor=c["panel"],
                lightcolor=c["panel"], darkcolor=c["panel"], font=fonts["small"])
    s.map("Quiet.TButton", foreground=[("active", c["text"])], background=[("active", c["panel"])])

    s.configure("Seg.Toolbutton", background=c["card"], foreground=c["muted"], padding=(12, 6),
                bordercolor=c["line"], lightcolor=c["card"], darkcolor=c["card"], relief="flat", font=fonts["small"])
    s.map("Seg.Toolbutton",
          background=[("selected", c["accent"]), ("active", c["line"])],
          foreground=[("selected", "#ffffff"), ("disabled", c["line"])])

    s.configure("TEntry", fieldbackground=c["card"], foreground=c["text"], bordercolor=c["line"],
                lightcolor=c["card"], darkcolor=c["card"], insertcolor=c["text"], padding=4)
    s.map("TEntry", bordercolor=[("focus", c["accent"])])
    s.configure("TCombobox", fieldbackground=c["card"], background=c["card"], foreground=c["text"],
                arrowcolor=c["muted"], bordercolor=c["line"], lightcolor=c["card"], darkcolor=c["card"], padding=5)
    s.map("TCombobox", fieldbackground=[("readonly", c["card"])], foreground=[("readonly", c["text"])],
          bordercolor=[("focus", c["accent"])])

    s.configure("Horizontal.TScale", background=c["panel"], troughcolor=c["line"],
                bordercolor=c["panel"], lightcolor=c["accent"], darkcolor=c["accent"])
    s.configure("Horizontal.TProgressbar", background=c["accent"], troughcolor=c["line"],
                bordercolor=c["panel"], lightcolor=c["accent"], darkcolor=c["accent"], thickness=4)
    s.configure("TCheckbutton", background=c["bg"], foreground=c["muted"], font=fonts["small"])
    s.map("TCheckbutton", background=[("active", c["bg"])], foreground=[("active", c["text"])])
    return fonts
