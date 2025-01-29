from typing import Dict, List, Tuple, Literal
import numpy as np
import plotly.graph_objects as go
import branca.colormap as cm

from mescal.utils.color_utils.conversion import to_hex

# TODO: remove this file

def _sorted_dict(d):
    return {k: d[k] for k in sorted(d.keys())}


def color_continuous_segments_to_colors_and_index(
        segments: Dict[Tuple[int, int], List[str]],
) -> Tuple[List[str], List[float]]:

    sorted_segments = _sorted_dict(segments)
    sorted_keys = list(sorted_segments.keys())

    vmax = sorted_keys[-1][-1]

    for k in sorted_keys:
        sorted_segments[k] = [to_hex(c) for c in segments[k]]

    colors, index = [], []
    for (start, end), color_segment in sorted_segments.items():
        num_colors = len(color_segment)

        if num_colors == 1:
            index.append(start)
            colors.append(color_segment[0])
            index.append(end if end == vmax else end - 1e-6)
            colors.append(color_segment[-1])
        else:
            for i, c in zip(np.linspace(start, end, num_colors), color_segment):
                if i == vmax:
                    _idx = i
                elif i == end:
                    _idx = i - 1e-6
                else:
                    _idx = i
                index.append(_idx)
                colors.append(c)
    return colors, index


def get_segmented_color_scale_figure(
        segments: Dict[Tuple[int, int], List[str]],
        orientation: Literal['horizontal', 'vertical'] = 'horizontal',
        tick_side: Literal['top', 'bottom', 'left', 'right'] = None,
        parts_per_segment: int = 50
) -> go.Figure:

    if tick_side is None:
        tick_side = 'right' if orientation == 'vertical' else 'bottom'

    thickness = 200
    overlap = 1e-2
    bgcolor = 'white'

    sorted_segments = _sorted_dict(segments)
    colors, index = color_continuous_segments_to_colors_and_index(segments)
    branca_cmap = cm.LinearColormap(colors, index)

    num_segments = len(segments)
    axis_segments = np.linspace(0, 1, num_segments + 1)
    fig = go.Figure()

    for i, k in enumerate(sorted_segments.keys()):
        _start = k[0]
        _end = k[1]
        start = axis_segments[i]
        end = axis_segments[i + 1]
        absolute_linspace = np.linspace(_start, _end, parts_per_segment)
        relative_linspace = np.linspace(start, end, parts_per_segment)
        for j in range(len(absolute_linspace) - 1):
            value = absolute_linspace[j]
            color = to_hex(branca_cmap(value))
            if orientation == 'vertical':
                fig.add_trace(go.Scatter(
                    x=[0, 0],
                    y=[relative_linspace[j], relative_linspace[j + 1] + overlap],
                    mode='lines',
                    line=dict(color=color, width=thickness),
                    showlegend=False
                ))
            else:  # horizontal
                fig.add_trace(go.Scatter(
                    x=[relative_linspace[j], relative_linspace[j + 1] + overlap],
                    y=[0, 0],
                    mode='lines',
                    line=dict(color=color, width=thickness),
                    showlegend=False
                ))

    tickvals = axis_segments
    ticktext = [k[0] for k in sorted_segments.keys()] + [list(sorted_segments.keys())[-1][1]]

    if orientation == 'vertical':
        fig.update_layout(
            xaxis=dict(showticklabels=False, range=[-thickness, thickness]),
            yaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                range=[0, 1],
                side=tick_side,
            ),
            xaxis_title=None,
            yaxis_title=None,
            width=thickness,
            plot_bgcolor=bgcolor
        )
    else:  # horizontal
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                range=[0, 1],
                side=tick_side,
            ),
            yaxis=dict(showticklabels=False, range=[-thickness, thickness]),
            xaxis_title=None,
            yaxis_title=None,
            height=thickness,
            plot_bgcolor=bgcolor
        )

    return fig


if __name__ == '__main__':
    import plotly.express as px

    color_continuous_segments = {
        (5000, 10000): ['#FF0000'],
        (1000, 5000): ['#FF00FF'],
        (500, 1000): px.colors.sequential.Pinkyl,
        (0, 500): px.colors.sequential.Tealgrn,
        (-500, 0): px.colors.sequential.BuPu_r[1:-4],
        (-1000, -500): ['#0080FF'],
        (-5000, -1000): ['#00ffff'],
        (-10000, -5000): ['#000000'],
    }

    colorss, indexx = color_continuous_segments_to_colors_and_index(color_continuous_segments)
    colormap = cm.LinearColormap(colorss, indexx)

    fig_horizontal_bottom = get_segmented_color_scale_figure(color_continuous_segments, orientation='horizontal', tick_side='bottom')
    fig_horizontal_bottom.show()

    fig_horizontal_top = get_segmented_color_scale_figure(color_continuous_segments, orientation='horizontal', tick_side='top')
    fig_horizontal_top.show()

    fig_vertical_right = get_segmented_color_scale_figure(color_continuous_segments, orientation='vertical', tick_side='right')
    fig_vertical_right.show()

    fig_vertical_left = get_segmented_color_scale_figure(color_continuous_segments, orientation='vertical', tick_side='left')
    fig_vertical_left.show()
