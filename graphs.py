import planarity
import numpy as np
import pandas as pd
import networkx as nx

def compute_distance(mat):
    return np.sqrt( 2 * np.clip( 1. - mat, 0., 2.) ) 

def construct_pmfg(df_corr_matrix):
    df_distance = compute_distance(df_corr_matrix) 
    dist_matrix = df_distance.values
    index_upper_triangular = np.triu_indices(dist_matrix.shape[0],1)
    isort = np.argsort( dist_matrix[index_upper_triangular] )
    G = nx.Graph()
    for k in range(0,len(isort)):
        u = index_upper_triangular[0][isort[k]]
        v = index_upper_triangular[1][isort[k]]
        if dist_matrix[u,v] > .0: # remove perfect correlation because of diagonal FIXME
            G.add_edge(u, v, {'weight': float(dist_matrix[u,v])})
            if not planarity.is_planar(G):
                G.remove_edge(u,v)
    return G

def construct_mst(df, threshold=0.1):
    g = nx.Graph()
    names = df.columns.unique()
    df_distance = compute_distance(df)
    for i, n in enumerate(names):
        for j, m in enumerate(names):
            if j >= i: break
            val = df_distance.loc[n,m]
            g.add_edge(n, m, weight=val)
            # Threshold-based filtering
            # if np.abs(val) > threshold / len(names):
                # g.add_edge(n, m, weight=val)
    return nx.minimum_spanning_tree(g)


from bokeh.models import ColumnDataSource, HoverTool, HBox, VBoxForm
from bokeh.plotting import figure, show
import seaborn as sns

def prepare_datasources(G, **kwargs):
    layout = nx.layout.spring_layout(G, **kwargs)
    import community
    partition = community.best_partition(G)
    df_ = pd.DataFrame(layout).T
    df_.columns=['x', 'y']
    df_['community'] = pd.DataFrame.from_dict({'nodes': list(partition.keys()), 'community': list(partition.values())}).set_index('nodes')
    n_communities = df_.community.max() + 1
    df_['color'] = df_.community.apply(hex_palette(n_communities))

    ctr = compute_centrality_metrics(G)
    df_['size'] = 8 * ctr.XpY
    df_['line_color'] = 'navy'
    chosen = ctr.XpY >= ctr.XpY.quantile((ctr.XpY.size - 30.) / ctr.XpY.size)
    df_['color'][chosen] = 'red'
    ds_nodes = ColumnDataSource(df_)

    
    df_edges = pd.DataFrame((df_.x[e[0]], df_.y[e[0]], df_.x[e[1]], df_.y[e[1]]) for e in G.edges_iter())
    df_edges.columns = ['x0', 'y0', 'x1', 'y1']
    ds_edges = ColumnDataSource(df_edges)
    
    return ds_nodes, ds_edges


def plot_graph(p, ds_edges, ds_nodes):
    segments = p.segment('x0', 'y0', 'x1', 'y1', color='grey', line_width=1, alpha=0.3, source=ds_edges)
    circles = p.scatter('x', 'y', marker='circle', size='size', line_color="navy", fill_color="color", alpha=0.8, source=ds_nodes)
    return circles, segments

def hex_palette(n):
    pal = sns.hls_palette(n).as_hex()
    return lambda c: pal[c]



def calc_metric(metric, metric_name, g, **kwargs):
    return pd.DataFrame.from_dict( metric(g, **kwargs), orient='index').rename(columns={0: metric_name})

def compute_rank(col, order='ascending'):
    return col.rank(ascending=(order == 'ascending'))


def compute_rank(col, order='ascending'):
    u, v = np.unique(col if order == 'ascending' else -col, return_inverse=True)
    return (np.cumsum(np.bincount(v, minlength=u.size)) - 1)[v]

def betweenness_centrality(g, weighted=True):
    _name = 'betweenness' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.betweenness_centrality, _name, g, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='descending')
    return _d

def degree_centrality(g, weighted=True):
    # TODO weight not yet taken into account, make own function
    _name = 'degree' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.degree_centrality, _name, g ) #, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='descending')
    return _d


def eigenvector_centrality(g, weighted=True):
    _name = 'eigenvector' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.eigenvector_centrality_numpy, _name, g, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='descending')
    return _d

def eccentricity(g, weighted=True):
    ### weight not yet taken into accoutn
    _name = 'eccentricity' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.distance_measures.eccentricity, _name, g) #, weight='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='ascending')
    return _d

def closeness(g, weighted=True):
    def calc2(metric, metric_name, g, distance='weight', **kwargs):
        return pd.DataFrame.from_dict( metric(g, distance=distance), orient='index').rename(columns={0: metric_name})
    _name = 'closeness' + ('_weighted' if weighted else '')
    _d = calc_metric(nx.algorithms.centrality.closeness.closeness_centrality, _name, g, distance='weight' if weighted else None)
    _d[_name] = compute_rank(_d[_name], order='ascending')
    return _d



def compute_centrality_metrics(g):
    """  Computes the centrality metrics according to ... """
    c = betweenness_centrality(g, weighted=True)
    # c['betweenness']
    c['betweenness'] = betweenness_centrality(g, weighted=False)
    c['degree'] = degree_centrality(g, weighted=False)
    c['degree_weighted'] = degree_centrality(g, weighted=True)
    c['X'] = (c.betweenness + c.betweenness_weighted + c.degree + c.degree_weighted  - 4) / 4 / len(c.index)

    c['eigenvector_weighted'] = eigenvector_centrality(g)
    c['eigenvector'] = eigenvector_centrality(g, weighted=False)
    c['closeness_weighted'] = closeness(g)
    c['closeness'] = closeness(g, weighted=False)
    c['eccentricity_weighted'] = eccentricity(g)
    c['eccentricity'] = eccentricity(g, weighted=False)
    c['Y'] = (c.eccentricity_weighted + c.eigenvector + c.closeness + c.closeness_weighted + c.eccentricity + c.eccentricity_weighted - 6) / 6 / len(c.index)
    c['XmY'] = c.Y - c.X
    c['XpY'] = c.Y + c.X
    return c