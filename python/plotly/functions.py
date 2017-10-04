import pandas as pd
import plotly.graph_objs as go



def df_plotly_histograms(df, feature, category, nbins=20):
    """Plot overlay histograms give a category"""
    start = df[feature].min()
    end = df[feature].max()
    size = (end - start)/ float(nbins)
    xbins = {'start':start, 'end':end, 'size':size}

    traces = [(t, g[feature].values) for t, g in df.groupby(category)]
    traces = [go.Histogram(x=g, opacity=0.5, name=t, xbins=xbins, autobinx=False) for t, g in traces]
    layout = go.Layout(barmode='overlay')
    fig = go.Figure(data=traces, layout=layout)

    return fig
