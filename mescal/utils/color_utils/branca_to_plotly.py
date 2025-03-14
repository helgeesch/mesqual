from branca.colormap import LinearColormap

from mescal.utils.color_utils.conversion import to_hex


def branca_to_plotly(branca_colormap: LinearColormap):
    colors = branca_colormap.colors
    positions = branca_colormap.index

    min_pos = min(positions)
    max_pos = max(positions)
    normalized_positions = [(pos - min_pos) / (max_pos - min_pos) for pos in positions]

    plotly_colorscale = [(pos, to_hex(color)) for pos, color in zip(normalized_positions, colors)]

    return plotly_colorscale


if __name__ == '__main__':
    import plotly.graph_objects as go
    branca_cmap = LinearColormap(['blue', 'green', 'yellow', 'red'], vmin=0, vmax=100)

    plotly_cscale = branca_to_plotly(branca_cmap)

    print(plotly_cscale)

    fig = go.Figure(data=go.Heatmap(
        z=[[1, 20, 30], [20, 1, 60], [30, 60, 1]],
        colorscale=plotly_cscale
    ))
    fig.show()
