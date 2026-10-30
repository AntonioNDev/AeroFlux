import importlib.util
import threading
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


class MatlabView:
    """Owns one MATLAB engine. Every call blocks, so run them from a worker thread."""

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    @staticmethod
    def installed() -> bool:
        return importlib.util.find_spec("matlab") is not None

    def warm_up(self) -> None:
        self._ensure()

    def show(self, mat_file) -> None:
        # background=True returns at once, the figure keeps running inside MATLAB
        self._ensure().render_session(str(mat_file), nargout=0, background=True)

    def close(self) -> None:
        with self._lock:
            engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    def _ensure(self):
        with self._lock:
            if self._engine is None:
                import matlab.engine
                self._engine = matlab.engine.start_matlab()
                self._engine.addpath(str(SCRIPT_DIR), nargout=0)
            return self._engine
