import sys
from formulas.placeholder import PlaceholderSolver

def main() -> None:
    if sys.platform == "win32":
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    from ui.app import App

    App(solver=PlaceholderSolver()).mainloop()


if __name__ == "__main__":
    main()
