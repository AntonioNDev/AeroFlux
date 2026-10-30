function render_session(matFile)
% Interactive view of one run, opened from the Python side.
d = load(matFile);

fig = figure('Name', ['Wind tunnel - ' d.summary.model], 'NumberTitle', 'off', ...
    'Color', [0.07 0.08 0.09], 'Position', [120 120 1000 700]);
ax = axes(fig, 'Color', 'none', 'Position', [0.02 0.02 0.84 0.96]);
hold(ax, 'on');

fields = {'pressure', 'cp', 'wall_temperature'};
labels = {'Pressure (Pa)', 'Cp', 'Wall temperature (K)'};
maps = {@parula, @cool, @hot};

body = patch(ax, 'Faces', d.faces, 'Vertices', d.vertices, ...
    'FaceVertexCData', d.pressure, 'FaceColor', 'flat', 'EdgeColor', 'none');
axis(ax, 'equal', 'off');
view(ax, 3);
camlight(ax, 'headlight');
material(body, 'dull');
cb = colorbar(ax, 'Color', [0.8 0.82 0.86], 'Position', [0.89 0.2 0.02 0.55]);

uicontrol(fig, 'Style', 'popupmenu', 'String', labels, ...
    'Units', 'normalized', 'Position', [0.02 0.94 0.25 0.04], ...
    'Callback', @(src, ~) setField(src.Value));
rotate3d(ax, 'on');
setField(1);

S = d.streamlines;
n = size(S, 1);
steps = size(S, 2);
trail = gobjects(n, 1);
for i = 1:n
    trail(i) = plot3(ax, NaN, NaN, NaN, 'Color', [0.45 0.8 1 0.55], 'LineWidth', 1);
end

stride = max(1, ceil(steps / 120));
for k = 1:stride:steps
    if ~isvalid(fig), return; end
    for i = 1:n
        set(trail(i), 'XData', S(i, 1:k, 1), 'YData', S(i, 1:k, 2), 'ZData', S(i, 1:k, 3));
    end
    drawnow limitrate;
    pause(0.02);
end

    function setField(idx)
        body.FaceVertexCData = d.(fields{idx});
        colormap(ax, maps{idx}(256));
        cb.Label.String = labels{idx};
        clim(ax, [min(body.FaceVertexCData) max(body.FaceVertexCData) + eps]);
    end
end
