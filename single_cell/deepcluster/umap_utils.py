from umap import UMAP

from plotly import graph_objects as go
from babyplots import Babyplot


def umap(X, n_neighbors=50, n_components=3, min_dist=0.0):
    embedding = UMAP(
        n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
    ).fit(X)
    return embedding, embedding.transform(X)



def plot(
    u,
    labels=None,
    colorscale="Viridis",
    backend="plotly",
    xrange=None,
    yrange=None,
    zrange=None,
    folded=False,
    folded_embedding=None,
):
    assert len(u.shape) == 2
    assert u.shape[1] in (2, 3)
    assert backend in ("plotly", "babyplots")

    if backend == "plotly":
        if u.shape[1] == 2:
            fig = go.Figure(
                data=go.Scattergl(
                    x=u[:, 0],
                    y=u[:, 1],
                    # z=u[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2, color=labels, colorscale=colorscale, opacity=0.9
                    ),
                )
            )
        elif u.shape[1] == 3:
            fig = go.Figure(
                data=go.Scatter3d(
                    x=u[:, 0],
                    y=u[:, 1],
                    z=u[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2, color=labels, colorscale=colorscale, opacity=0
                    ),
                )
            )
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=xrange),
                yaxis=dict(range=yrange),
                zaxis=dict(range=zrange),
            )
        )
    elif backend == "babyplots":
        if labels is None:
            labels = np.zeros(len(u))
        fig = Babyplot()
        options = dict(
            # shape='sphere',
            color_scale=colorscale,
            show_axes=[True, True, True],
            show_legend=True,
            folded=folded,
        )
        options["show_axes"] = [True] * u.shape[1]
        if folded is True:
            options["folded_embedding"] = folded_embedding
        fig.add_plot(u, "point_cloud", "values", labels, options)
    return fig

