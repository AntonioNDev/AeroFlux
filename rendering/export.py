import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat

from formulas.pipeline import TestResult

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def write_run(result: TestResult, root: Path = OUTPUT_DIR) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(root) / f"{result.model.name}_{stamp}"
    folder.mkdir(parents=True, exist_ok=True)
    write_mat(result, folder / "run.mat")
    write_json(result, folder / "summary.json")
    return folder


def export_file(result: TestResult, path) -> None:
    path = Path(path)
    writers = {".csv": write_csv, ".json": write_json, ".mat": write_mat}
    if path.suffix.lower() not in writers:
        raise ValueError(f"Cannot export {path.suffix!r}")
    writers[path.suffix.lower()](result, path)


def write_mat(result: TestResult, path) -> None:
    model = result.model
    savemat(path, {
        "vertices": model.vertices,
        "faces": (model.faces + 1).astype(float),
        "pressure": result.pressure.reshape(-1, 1),
        "cp": result.cp.reshape(-1, 1),
        "wall_temperature": result.wall_temperature.reshape(-1, 1),
        "streamlines": result.streamlines,
        "summary": {**result.summary, "model": model.name},
        "settings": result.settings.as_dict(),
    })


def write_json(result: TestResult, path) -> None:
    body = {
        "model": result.model.name,
        "solver": result.solver,
        "settings": result.settings.as_dict(),
        "summary": result.summary,
        "advice": result.advice,
    }
    Path(path).write_text(json.dumps(body, indent=2))


def write_csv(result: TestResult, path) -> None:
    m = result.model
    table = np.column_stack([
        np.arange(len(m.faces)), m.face_centroids, m.face_areas,
        result.pressure, result.cp, result.wall_temperature,
    ])
    header = "face,x,y,z,area_m2,pressure_pa,cp,wall_temperature_k"
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.6g")
