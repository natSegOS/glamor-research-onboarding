"""Inline-SVG chart builders and heatmap color math for the HIVE report.

Every chart is theme-aware: geometry is fixed, colors come from CSS custom properties
(--series-*, --ink-*, --grid, --diverging poles) defined in the report stylesheet, so
the same SVG renders correctly in light and dark mode. Heatmap cells precompute one
background per mode (``--cell-light`` / ``--cell-dark``) because ramp interpolation
cannot be expressed in CSS variables alone.
"""

from __future__ import annotations

CHART_FONT = "font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
LABEL_FONT_SIZE = 12
ROW_HEIGHT = 26
BAR_THICKNESS = 12
MARKER_RADIUS = 5

LIGHT_NEUTRAL, DARK_NEUTRAL = "#f0efec", "#383835"
LIGHT_NEGATIVE_POLE, DARK_NEGATIVE_POLE = "#e34948", "#e66767"
LIGHT_POSITIVE_POLE, DARK_POSITIVE_POLE = "#2a78d6", "#3987e5"
LIGHT_SEQUENTIAL_POLE, DARK_SEQUENTIAL_POLE = "#256abf", "#6da7ec"
INK_THRESHOLD = 0.62


def hex_to_rgb(hex_color):
    return tuple(int(hex_color[position:position + 2], 16) for position in (1, 3, 5))


def blend(from_hex, to_hex, strength):
    from_rgb, to_rgb = hex_to_rgb(from_hex), hex_to_rgb(to_hex)
    mixed = [round(f + (t - f) * strength) for f, t in zip(from_rgb, to_rgb)]
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def diverging_cell_style(value, full_scale):
    """Per-cell style carrying both modes' backgrounds and an ink override when the
    light-mode background gets deep enough to need white text."""
    strength = min(1.0, abs(value) / full_scale)
    light_pole = LIGHT_NEGATIVE_POLE if value < 0 else LIGHT_POSITIVE_POLE
    dark_pole = DARK_NEGATIVE_POLE if value < 0 else DARK_POSITIVE_POLE
    light_background = blend(LIGHT_NEUTRAL, light_pole, strength)
    dark_background = blend(DARK_NEUTRAL, dark_pole, strength)
    ink_override = ";--cell-ink-light:#ffffff" if strength > INK_THRESHOLD else ""
    return f"--cell-light:{light_background};--cell-dark:{dark_background}{ink_override}"


def sequential_cell_style(value, full_scale):
    strength = min(1.0, max(0.0, value / full_scale))
    light_background = blend("#ffffff", LIGHT_SEQUENTIAL_POLE, strength * 0.85)
    dark_background = blend("#1a1a19", DARK_SEQUENTIAL_POLE, strength * 0.55)
    ink_override = ";--cell-ink-light:#ffffff" if strength > INK_THRESHOLD else ""
    return f"--cell-light:{light_background};--cell-dark:{dark_background}{ink_override}"


def heatmap_table(row_labels, column_labels, values_by_row, cell_style, cell_format,
                  row_group_of=None, partial_cells=frozenset(), cell_title=None,
                  row_label_html=None):
    """An HTML-table heatmap. values_by_row maps row label -> {column label: value};
    missing cells render as em-dashes. row_group_of optionally maps a row label to a
    group heading inserted when the group changes. Cells in partial_cells (a set of
    (row label, column label)) render uncolored with a dagger — their value comes from
    an incomplete cell and must not read as a finding. cell_title(value, row, column)
    supplies a hover explanation per cell; row_label_html(row_label) lets callers wrap
    row labels (e.g. in glossary term spans)."""
    header_cells = "".join(f"<th>{column}</th>" for column in column_labels)
    body_rows, current_group = [], None
    for row_label in row_labels:
        group = row_group_of(row_label) if row_group_of else None
        if group is not None and group != current_group:
            body_rows.append(
                f'<tr class="heatmap-group"><td colspan="{len(column_labels) + 1}">{group}</td></tr>')
            current_group = group
        cells = []
        for column in column_labels:
            value = values_by_row.get(row_label, {}).get(column)
            title = (f' title="{cell_title(value, row_label, column)}"'
                     if cell_title and value is not None else "")
            if value is None:
                cells.append('<td class="heatmap-cell heatmap-missing">—</td>')
            elif (row_label, column) in partial_cells:
                cells.append(f'<td class="heatmap-cell heatmap-partial"{title}>'
                             f'{cell_format(value)}†</td>')
            else:
                cells.append(
                    f'<td class="heatmap-cell" style="{cell_style(value)}"{title}>'
                    f'{cell_format(value)}</td>')
        label = row_label_html(row_label) if row_label_html else row_label
        body_rows.append(f'<tr><th class="heatmap-row-label">{label}</th>{"".join(cells)}</tr>')
    return (
        '<div class="table-scroll"><table class="heatmap">'
        f'<thead><tr><th></th>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table></div>')


def _svg_open(width, height):
    return (f'<svg viewBox="0 0 {width} {height}" role="img" '
            f'style="max-width:100%;height:auto;{CHART_FONT}">')


def _text(x, y, content, anchor="start", css_class="chart-label", size=LABEL_FONT_SIZE):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{css_class}" '
            f'font-size="{size}">{content}</text>')


def _value_scale(minimum, maximum, pixel_start, pixel_end):
    span = (maximum - minimum) or 1.0
    return lambda value: pixel_start + (value - minimum) / span * (pixel_end - pixel_start)


def _axis_ticks(scale, tick_values, y, tick_format):
    parts = [f'<line x1="{scale(min(tick_values)):.1f}" x2="{scale(max(tick_values)):.1f}" '
             f'y1="{y}" y2="{y}" class="chart-baseline"/>']
    parts += [
        _text(scale(tick), y + 16, tick_format(tick), anchor="middle", css_class="chart-muted")
        for tick in tick_values
    ]
    return parts


def diverging_break_fix_bars(rows, width=760):
    """rows: [{label, break_pct, fix_pct, churn_pct, churn_note}]. Breaks extend left in
    the negative pole color, fixes right in the positive pole color; churn is the direct
    label on the far right."""
    label_gutter, right_gutter, top = 130, 128, 26
    value_label_reserve = 42
    plot_left, plot_right = label_gutter, width - right_gutter
    largest = max(max(row["break_pct"], row["fix_pct"]) for row in rows)
    center = (plot_left + plot_right) / 2
    scale = ((plot_right - plot_left) / 2 - value_label_reserve) / largest
    height = top + len(rows) * (ROW_HEIGHT + 8) + 34
    parts = [_svg_open(width, height)]
    parts.append(_text(center - 6, 14, "breaks ←", anchor="end", css_class="chart-muted"))
    parts.append(_text(center + 6, 14, "→ fixes", css_class="chart-muted"))
    parts.append(_text(width - 4, 14, "churn", anchor="end", css_class="chart-muted"))
    for index, row in enumerate(rows):
        y = top + index * (ROW_HEIGHT + 8)
        bar_y = y + (ROW_HEIGHT - BAR_THICKNESS) / 2
        break_width = row["break_pct"] * scale
        fix_width = row["fix_pct"] * scale
        parts.append(_text(label_gutter - 10, y + ROW_HEIGHT / 2 + 4, row["label"], anchor="end"))
        parts.append(
            f'<rect x="{center - break_width:.1f}" y="{bar_y:.1f}" width="{break_width:.1f}" '
            f'height="{BAR_THICKNESS}" rx="4" class="mark-negative">'
            f'<title>{row["label"]}: {row["break_pct"]:.2f}% break</title></rect>')
        parts.append(
            f'<rect x="{center:.1f}" y="{bar_y:.1f}" width="{fix_width:.1f}" '
            f'height="{BAR_THICKNESS}" rx="4" class="mark-positive">'
            f'<title>{row["label"]}: {row["fix_pct"]:.2f}% fix</title></rect>')
        parts.append(_text(center - break_width - 6, y + ROW_HEIGHT / 2 + 4,
                           f'{row["break_pct"]:.1f}', anchor="end", css_class="chart-muted"))
        parts.append(_text(center + fix_width + 6, y + ROW_HEIGHT / 2 + 4,
                           f'{row["fix_pct"]:.1f}', css_class="chart-muted"))
        parts.append(_text(width - 4, y + ROW_HEIGHT / 2 + 4,
                           f'{row["churn_pct"]:.1f}%{row.get("churn_note", "")}', anchor="end"))
    parts.append(f'<line x1="{center:.1f}" x2="{center:.1f}" y1="{top - 4}" '
                 f'y2="{height - 30}" class="chart-baseline"/>')
    parts.append("</svg>")
    return "".join(parts)


def dumbbell_chart(rows, minimum, maximum, tick_values, tick_format,
                   from_legend, to_legend, width=760, zero_line=None, label_gutter=190):
    """rows: [{label, from_value, to_value, from_title, to_title}]. A gray 'from' dot
    connected to a colored 'to' dot on a shared horizontal scale."""
    right_gutter, top = 30, 34
    scale = _value_scale(minimum, maximum, label_gutter, width - right_gutter)
    height = top + len(rows) * ROW_HEIGHT + 40
    parts = [_svg_open(width, height)]
    parts.append(f'<circle cx="{label_gutter}" cy="12" r="{MARKER_RADIUS}" class="mark-reference"/>')
    parts.append(_text(label_gutter + 10, 16, from_legend, css_class="chart-muted"))
    legend_second_x = label_gutter + 10 + 7 * len(from_legend) + 30
    parts.append(f'<circle cx="{legend_second_x}" cy="12" r="{MARKER_RADIUS}" class="mark-accent"/>')
    parts.append(_text(legend_second_x + 10, 16, to_legend, css_class="chart-muted"))
    if zero_line is not None:
        parts.append(f'<line x1="{scale(zero_line):.1f}" x2="{scale(zero_line):.1f}" '
                     f'y1="{top - 6}" y2="{height - 34}" class="chart-zeroline"/>')
    for index, row in enumerate(rows):
        y = top + index * ROW_HEIGHT + ROW_HEIGHT / 2
        from_x, to_x = scale(row["from_value"]), scale(row["to_value"])
        parts.append(_text(label_gutter - 10, y + 4, row["label"], anchor="end"))
        parts.append(f'<line x1="{from_x:.1f}" x2="{to_x:.1f}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'class="dumbbell-connector"/>')
        parts.append(f'<circle cx="{from_x:.1f}" cy="{y:.1f}" r="{MARKER_RADIUS}" '
                     f'class="mark-reference"><title>{row["from_title"]}</title></circle>')
        parts.append(f'<circle cx="{to_x:.1f}" cy="{y:.1f}" r="{MARKER_RADIUS}" '
                     f'class="mark-accent"><title>{row["to_title"]}</title></circle>')
    parts += _axis_ticks(scale, tick_values, height - 28, tick_format)
    parts.append("</svg>")
    return "".join(parts)


def dot_with_interval_chart(rows, minimum, maximum, tick_values, tick_format,
                            width=760, zero_line=0.0):
    """rows: [{label, value, low, high, title}]. A dot with a horizontal 95% interval
    whisker and a zero reference line."""
    label_gutter, right_gutter, top = 190, 30, 16
    scale = _value_scale(minimum, maximum, label_gutter, width - right_gutter)
    height = top + len(rows) * ROW_HEIGHT + 40
    parts = [_svg_open(width, height)]
    if zero_line is not None:
        parts.append(f'<line x1="{scale(zero_line):.1f}" x2="{scale(zero_line):.1f}" '
                     f'y1="{top - 6}" y2="{height - 34}" class="chart-zeroline"/>')
    for index, row in enumerate(rows):
        y = top + index * ROW_HEIGHT + ROW_HEIGHT / 2
        parts.append(_text(label_gutter - 10, y + 4, row["label"], anchor="end"))
        parts.append(f'<line x1="{scale(row["low"]):.1f}" x2="{scale(row["high"]):.1f}" '
                     f'y1="{y:.1f}" y2="{y:.1f}" class="interval-whisker"/>')
        parts.append(f'<circle cx="{scale(row["value"]):.1f}" cy="{y:.1f}" r="{MARKER_RADIUS}" '
                     f'class="mark-accent"><title>{row["title"]}</title></circle>')
        parts.append(_text(scale(row["high"]) + 8, y + 4,
                           f'{row["value"]:+.2f}', css_class="chart-muted"))
    parts += _axis_ticks(scale, tick_values, height - 28, tick_format)
    parts.append("</svg>")
    return "".join(parts)
