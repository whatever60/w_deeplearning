from typing import Dict, List

import numpy as np
from plotly import graph_objects as go


def autorange(X: np.ndarray):
    return zip(X.min(axis=0), X.max(axis=0))
    

def plot_animation(data: List[List[Dict]], names: List[str], xrange=None, yrange=None, zrange=None):
    redraw = False if ('type' not in data[0][0] or data[0][0]['type'] == 'scatter') else True
    # make figure
    layout=dict(
        scene=dict(
            xaxis=dict(
                range=xrange,
                # title='Life Expectancy',
                autorange=False
            ),
            yaxis=dict(
                range=yrange,
                # title='GDP per Capita',
                # type='log',
                autorange=False
            ),
            zaxis=dict(
                range=zrange,
                autorange=False
            )
        ),
        hovermode='closest',
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[
                            None,
                            dict(
                                frame=dict(duration=500, redraw=redraw),
                                fromcurrent=True,
                                transition=dict(duration=300, easing='quadratic-in-out')
                            )
                        ],
                        label='Play',
                        method='animate'
                    ),
                    dict(
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=redraw),
                                mode='immediate',
                                transition=dict(duration=0)
                            )
                        ],
                        label='Pause',
                        method='animate'
                    )
                ],
                direction='left',
                pad=dict(r=10, t=87),
                showactive=False,
                type='buttons',
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='top'
            )
        ],
        sliders=[
            dict(
                active=0,
                xanchor='left',
                yanchor='top',
                currentvalue=dict(
                    font=dict(size=20),
                    prefix='Year:',
                    visible=True,
                    xanchor='right'
                ),
                transition=dict(duration=300, easing='cubic-in-out'),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
            )
        ]
    )
    layout['sliders'][0]['steps'] = [
        dict(
            args=[
                [name],
                dict(
                    frame=dict(duration=300, redraw=redraw),
                    mode='immediate',
                    transition=dict(duration=300)
                )
            ],
            label=name,
            method='animate'
        )
        for name in names
    ]
    frames = [dict(data=d, name=name) for d, name in zip(data, names)]
    fig = go.Figure(dict(
        data=frames[0]['data'],
        layout=layout,
        frames=frames,
    ))
    fig.show(renderer='notebook')
    # return fig