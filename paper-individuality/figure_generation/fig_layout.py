"""
LAYOUT HELPERS SHARED BY THE ASSEMBLED FIGURES IN figure_generation/
=====================================================================
Every assembled figure is built the same way: one 180 mm matplotlib figure, panels drawn by
`draw_X(fig, spec)` functions into a nested GridSpec, then

    align_left(fig, [axes of the panels in the left column])     # one left edge
    place_letters(fig, panels, rows, left=[...])                  # one letter style

so the margins, the left edge and the panel letters follow one convention across figures:

* MARGINS: `fig.add_gridspec(..., **MARGINS)`, row gap `ROW_HSPACE`.
* LEFT EDGE: the leftmost axes of every row start at the same x (the rightmost of their
  current starts, so every y label still fits).
* LETTERS: bold, font size + 3, bottom-aligned just above the TOP of the row's content
  (titles included), so a row's letters share one height whatever the panels carry above
  their axes; letters of the left column share one x, just left of the widest y label.
"""
import matplotlib.pyplot as plt

FIG_WIDTH = 7.09                    # in; 180 mm, double column
MARGINS = dict(left=0.075, right=0.985, top=0.97, bottom=0.03)
ROW_HSPACE = 0.38
LETTER_PAD = 0.004                  # figure fraction between a row's top and its letters


def _bbox(fig, ax):
    return ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())


def align_left(fig, axes):
    """Give `axes` one left edge (the rightmost of their current ones), keeping each right edge."""
    x0 = max(a.get_position().x0 for a in axes)
    for a in axes:
        b = a.get_position()
        a.set_position([x0, b.y0, b.x1 - x0, b.height])
    fig.canvas.draw()
    return x0


def place_letters(fig, panels, rows, left=(), letters=None, pad=LETTER_PAD, x_groups=()):
    """Draw the panel letters.

    panels    {key: [axes]}
    rows      list of strings/lists of keys that share one letter height, e.g. ['AB', 'CDG']
    left      keys whose letters share the left-column x
    letters   {key: text}; default key.lower()
    x_groups  further groups of keys that share one x (e.g. a right-hand column, ['JK'])
    """
    fig.canvas.draw()
    letters = letters or {k: k.lower() for k in panels}
    y = {}
    for row in rows:
        top = max(_bbox(fig, a).y1 for k in row for a in panels[k])
        for k in row:
            y[k] = top + pad
    x = {k: max(min(_bbox(fig, a).x0 for a in panels[k]) - 0.005, 0.003) for k in panels}
    for group in [left, *x_groups]:
        if group:
            gx = min(x[k] for k in group)
            for k in group:
                x[k] = gx
    for k in panels:
        fig.text(x[k], y.get(k, max(_bbox(fig, a).y1 for a in panels[k]) + pad), letters[k],
                 fontsize=plt.rcParams['font.size'] + 3, fontweight='bold', ha='left', va='bottom')


def free_corner(ax, x, y, frac=(0.45, 0.3)):
    """The axes corner ('upper left', 'upper right', 'lower left', 'lower right') holding the fewest data
    points in a box of `frac` (width, height) of the axes -- where a statistics block covers least data.
    Call after the limits are final."""
    import numpy as np
    pts = ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    pts = ax.transAxes.inverted().transform(pts)
    fx, fy = frac
    boxes = {'upper left': (pts[:, 0] < fx) & (pts[:, 1] > 1 - fy),
             'upper right': (pts[:, 0] > 1 - fx) & (pts[:, 1] > 1 - fy),
             'lower left': (pts[:, 0] < fx) & (pts[:, 1] < fy),
             'lower right': (pts[:, 0] > 1 - fx) & (pts[:, 1] < fy)}
    return min(boxes, key=lambda k: boxes[k].sum())


def stats_text(ax, corner, r, p, n, q=None, rho=False, p_floor=None):
    """The correlation block of ps.corr_panel ('r = .. / p = .. stars / n = ..'), optionally with the FDR q,
    placed inside the axes at `corner`."""
    import paper_style as ps
    sym = 'ρ' if rho else 'r'
    sig = (q if q is not None else p) < 0.05
    mark = ps.stars(max(q if q is not None else p, 1e-12)) if sig else 'n.s.'
    # ONE p-value: the FDR-corrected q when the test belongs to a corrected family, the raw p otherwise
    if q is not None:
        val, name = q, 'p(FDR)'
    else:
        val, name = p, 'p'
    v_txt = f'{name} < {p_floor:g}' if (p_floor is not None and val <= 0) else f'{name} = {val:.1e}'
    lines = [f'{sym} = {r:.2f}', f'{v_txt} {mark}']
    lines.append(f'n = {n}')
    x, ha = (0.04, 'left') if 'left' in corner else (0.96, 'right')
    y, va = (0.97, 'top') if 'upper' in corner else (0.04, 'bottom')
    return ax.text(x, y, '\n'.join(lines), transform=ax.transAxes, ha=ha, va=va, zorder=5,
                   fontsize=plt.rcParams['font.size'] * 0.75, color='0.2', linespacing=1.2)


def stats_auto(ax, x, y, r, p, n, q=None, rho=False, p_floor=None, extend=False):
    """stats_text in the corner where its text box covers the fewest data points. The axes stay fitted to the data
    (the text may overlap points); extend=True instead grows the y axis so the block sits above the data."""
    import numpy as np
    fig = ax.figure
    x, y = np.asarray(x, float), np.asarray(y, float)

    def count(corner):
        fig.canvas.draw()
        pts = ax.transData.transform(np.column_stack([x, y]))
        t = stats_text(ax, corner, r, p, n, q=q, rho=rho, p_floor=p_floor)
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).expanded(1.05, 1.1)
        k = int(((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) & (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)).sum())
        h = bb.height / ax.get_window_extent().height
        t.remove()
        return k, h

    corners = ('upper left', 'upper right', 'lower left', 'lower right')
    res = {c: count(c) for c in corners}
    best = min(corners, key=lambda c: res[c][0])
    if extend and res[best][0] > 0:
        # make room: the data keep the bottom (1 - h) of the axes, the block takes the top
        h = max(v[1] for v in res.values()) + 0.03
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * h / (1 - h))
        best = min(('upper left', 'upper right'), key=lambda c: count(c)[0])
    return stats_text(ax, best, r, p, n, q=q, rho=rho, p_floor=p_floor)
