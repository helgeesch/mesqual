import plotly.graph_objects as go


def set_title(fig: go.Figure, title: str):
    title = f'<b>{title}</b>'
    fig.update_layout(title_text=title, title_x=0.5)


def remove_category_in_annotations(fig: go.Figure):
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))


def make_annotations_bold(fig: go.Figure):
    fig.for_each_annotation(lambda a: a.update(text='<b>' + a.text + '</b>'))


def unmatch_xaxes(fig: go.Figure):
    fig.update_xaxes(matches=None)


def unmatch_yaxes(fig: go.Figure):
    fig.update_yaxes(matches=None)


def reverse_legend_traceorder(fig: go.Figure):
    fig.update_layout(legend_traceorder="reversed")


def add_datetime_rangeslider(fig: go.Figure):
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_layout(yaxis_fixedrange=False)
    return fig
