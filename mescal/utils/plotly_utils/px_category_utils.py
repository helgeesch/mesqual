import plotly.graph_objects as go
import plotly.express as px


def get_x_y_axis_for_category(fig: go.Figure, category_args: dict[str, str]) -> tuple[str, str]:
    keys = [f'{k}={i}' for k, i in category_args.items()]
    for trace in fig.data:
        if all(k in trace.hovertemplate for k in keys):
            x_axis = trace.xaxis if 'xaxis' in trace else 'x'
            y_axis = trace.yaxis if 'yaxis' in trace else 'y'
            return x_axis, y_axis
    raise KeyError(f'No trace with matching key: value pairs {keys} found in any hovertemplate.')


def get_all_x_axis_names(fig: go.Figure) -> list[str]:
    return [attr for attr in fig.layout if attr.startswith('xaxis')]


def get_all_y_axis_names(fig: go.Figure) -> list[str]:
    return [attr for attr in fig.layout if attr.startswith('yaxis')]


def get_row_col_for_x_y_axis(fig: go.Figure, x_axis: str, y_axis: str) -> tuple[int, int]:
    all_x_axis_names = get_all_x_axis_names(fig)
    all_y_axis_names = get_all_y_axis_names(fig)

    x_domains = list(sorted(set([tuple(fig.layout[x].domain) for x in all_x_axis_names])))
    y_domains = list(sorted(set([tuple(fig.layout[y].domain) for y in all_y_axis_names])))

    if 'axis' not in x_axis:
        x_axis = f'xaxis{x_axis[1:]}'
    if 'axis' not in y_axis:
        y_axis = f'yaxis{y_axis[1:]}'

    if x_axis not in all_x_axis_names:
        raise KeyError(f'No matching axis found for {x_axis}.')
    if y_axis not in all_y_axis_names:
        raise KeyError(f'No matching axis found for {y_axis}.')

    col = x_domains.index(tuple(fig.layout[x_axis].domain)) + 1
    row = y_domains.index(tuple(fig.layout[y_axis].domain)) + 1
    return row, col


def get_subplot_row_and_col_for_category(fig: go.Figure, category_args: dict[str, str]) -> tuple[int, int]:
    x_axis, y_axis = get_x_y_axis_for_category(fig, category_args)
    row, col = get_row_col_for_x_y_axis(fig, x_axis, y_axis)
    return row, col


def get_index_for_category_on_axis(fig: go.Figure, axis: str, category_value: str) -> int:
    if not isinstance(category_value, str):
        raise TypeError(
            'Method only works with string categories. '
            'Sure you need this? In case you already have an int / float, just use the value as an index directly.'
        )
    if 'axis' not in axis:
        if axis.startswith('x'):
            axis = f'xaxis{axis[1:]}'
        elif axis.startswith('y'):
            axis = f'yaxis{axis[1:]}'
        else:
            raise KeyError(f'Unknown axis {axis}')
    cat_array = fig.layout[axis].categoryarray
    if category_value not in cat_array:
        raise KeyError(f'Unknown category {category_value}. Recognised categories for axis: {cat_array}')
    index = cat_array.index(category_value) + 1
    return index


if __name__ == '__main__':

    data = px.data.tips().groupby(['sex', 'time', 'day', 'smoker'])['tip'].mean().to_frame().reset_index()
    figg = px.bar(
        data,
        x="time",
        y="tip",
        color="day",
        facet_col="sex",
        facet_row="smoker",
        barmode='group',
        title="Sample Bar Chart with Facets"
    )
    data_mean = px.data.tips().groupby(['sex', 'smoker'])['tip'].mean()
    # for (sex, smoker), value in data_mean.items():
    #     row, col = get_subplot_row_and_col_for_category(fig, category_args={'sex': sex, 'smoker': smoker})
    #     figg.add_trace(
    #         go.Scatter(
    #             x=[1, 2],
    #             y=[value, value],
    #             mode='lines',
    #             line=dict(color='black'),
    #             hovertext=f'mean for sex={sex} and smoker={smoker}'
    #         ),
    #         row=row,
    #         col=col
    #     )

    data_mean = px.data.tips().groupby(['sex', 'smoker', 'time'])['tip'].mean()
    # for (sex, smoker, time), value in data_mean.items():
    #     xax, yax = get_x_y_axis_for_category(fig, category_args={'sex': sex, 'smoker': smoker})
    #     x_index = get_index_for_category_on_axis(fig, xax, time)
    #     figg.add_trace(
    #         go.Scatter(
    #             x=[x_index-0.4, x_index+0.4],
    #             y=[value, value],
    #             mode='lines',
    #             line=dict(color='blue'),
    #             xaxis=xax,
    #             yaxis=yax,
    #             hovertext=f'mean for sex={sex} and smoker={smoker} and time={time}'
    #         )
    #     )

    figg.write_html('tmp/tmp.html')