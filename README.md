# Wind Tunnel

    conda env create -f requirements.yaml
    conda activate windtunnel
    python main.py

    ui/          window, theme, widgets, 2-thread job queue
    rendering/   embedded matplotlib view, MATLAB launcher, render_session.m, file export
    formulas/    mesh loading, TestSettings/TestResult/Solver contract, placeholder solver
    output/      every run is saved here automatically (run.mat + summary.json)

Shortcuts: Ctrl/Cmd+O import, Ctrl/Cmd+S save data, Ctrl/Cmd+R restart, F5 start, Space play/pause.
Mouse in the 3D view: left drag rotates, right drag zooms, middle drag pans, wheel zooms.
