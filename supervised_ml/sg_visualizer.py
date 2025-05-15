# sg_visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox

def plot_truck_routes_comparison(graph, task_edges,
                                 pred_route_dict,
                                 gt_route_dict,
                                 depots):
    """
    Visualize predicted vs. ground truth routes on the *subgraph* induced by task_edges,
    using OSMnx’s plot_graph with a true black background.

    Parameters
    ----------
    graph : networkx.Graph or MultiDiGraph
        The full OSMnx graph with node attributes 'x','y'.
    task_edges : list of (u,v)
        The edges defining which part of `graph` to extract.
    pred_route_dict : dict[int -> list of node IDs]
    gt_route_dict   : dict[int -> list of node IDs]
    depots : dict[int -> node ID]
    """
    # 1) Build node set and induce subgraph
    nodes = {u for (u, v) in task_edges} | {v for (u, v) in task_edges}
    G_sub = graph.subgraph(nodes).copy()

    # 2) Grab positions
    pos = {nid: (data['x'], data['y'])
           for nid, data in G_sub.nodes(data=True)}

    trucks = list(pred_route_dict.keys())
    if not trucks:
        print("No routes to plot.")
        return

    # 3) Create figure & axes
    rows, cols = len(trucks), 2
    fig, axes = plt.subplots(rows, cols,
                             figsize=(14, 7 * rows),
                             squeeze=False)
    # force the figure background to black
    fig.patch.set_facecolor('black')

    # 4) Loop over each truck
    for i, truck in enumerate(trucks):
        depot_node = depots.get(truck)

        # ---- Predicted panel ----
        ax_pred = axes[i][0]
        ax_pred.set_facecolor('black')

        # identify edges in this truck’s predicted route
        seq_pred = pred_route_dict[truck]
        service_pred = {frozenset((a, b)) for a, b in zip(seq_pred, seq_pred[1:])}

        # prepare colors & widths
        edge_colors, edge_widths = [], []
        for u, v, key, data in G_sub.edges(keys=True, data=True):
            if frozenset((u, v)) in service_pred:
                edge_colors.append('red')
                edge_widths.append(2.0)
            else:
                edge_colors.append('lightgray')
                edge_widths.append(1.0)

        # plot with OSMnx, forcing black background
        ox.plot_graph(G_sub, ax=ax_pred,
                      node_size=20,
                      edge_color=edge_colors,
                      edge_linewidth=edge_widths,
                      bgcolor='black',
                      show=False, close=False)

        # draw depot
        if depot_node in pos:
            ax_pred.scatter(*pos[depot_node],
                            marker='*', s=200,
                            c='yellow', edgecolors='black', zorder=5)

        ax_pred.set_title(f"Truck {truck} — Predicted", color='white')
        ax_pred.axis('off')

        # ---- Ground-truth panel ----
        ax_gt = axes[i][1]
        ax_gt.set_facecolor('black')

        seq_gt = gt_route_dict[truck]
        service_gt = {frozenset((a, b)) for a, b in zip(seq_gt, seq_gt[1:])}

        edge_colors, edge_widths = [], []
        for u, v, key, data in G_sub.edges(keys=True, data=True):
            if frozenset((u, v)) in service_gt:
                edge_colors.append('green')
                edge_widths.append(2.0)
            else:
                edge_colors.append('lightgray')
                edge_widths.append(1.0)

        ox.plot_graph(G_sub, ax=ax_gt,
                      node_size=20,
                      edge_color=edge_colors,
                      edge_linewidth=edge_widths,
                      bgcolor='black',
                      show=False, close=False)

        if depot_node in pos:
            ax_gt.scatter(*pos[depot_node],
                          marker='*', s=200,
                          c='yellow', edgecolors='black', zorder=5)

        ax_gt.set_title(f"Truck {truck} — Ground Truth", color='white')
        ax_gt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_truck_routes_from_model(graph, route_dict, depots):
    """
    Visualize truck routes on the given graph.
    
    Parameters:
      graph      : a networkx graph (e.g., from OSMnx)
      route_dict : dictionary mapping truck id to a route list.
                   Each route list is of the form:
                     [depot, (u,v), (u,v), ..., depot]
                   where the first and last values are depot nodes (ints),
                   and the intermediate tuples represent edges to be highlighted.
      depots     : dictionary mapping truck id to depot node.
    """
    trucks = list(route_dict.keys())
    if not trucks:
        print("No routes found; nothing to plot.")
        return
    
    n = len(trucks)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    
    # Normalize axes.
    if n == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]
    
    for idx, truck in enumerate(trucks):
        ax = axes_flat[idx]
        G_copy = graph.copy()
        
        # Build a set of service edges from the route (only include the tuples)
        service_edges = set()
        route = route_dict[truck]
        for item in route:
            if isinstance(item, tuple):
                service_edges.add(item)
        
        # Color each edge in the graph: if it is in the service_edges, color it red; otherwise, gray.
        for (u, v, key, data) in G_copy.edges(keys=True, data=True):
            if (u, v) in service_edges:
                data["color"] = "red"
                data["linewidth"] = 3.0
            else:
                data["color"] = "gray"
                data["linewidth"] = 1.0
        
        # Prepare lists of colors and line widths.
        edge_colors = [data.get("color", "gray") for (_, _, _, data) in G_copy.edges(keys=True, data=True)]
        edge_linewidths = [data.get("linewidth", 1.0) for (_, _, _, data) in G_copy.edges(keys=True, data=True)]
        
        # Plot the graph with the colored edges.
        ox.plot_graph(G_copy, ax=ax, node_size=20,
                      edge_color=edge_colors, edge_linewidth=edge_linewidths,
                      show=False, close=False)
        ax.set_title(f"Truck {truck}")
        
        # Mark the depot with a star.
        depot_node = depots[truck]
        sx = G_copy.nodes[depot_node]['x']
        sy = G_copy.nodes[depot_node]['y']
        ax.scatter(sx, sy, marker='*', c='yellow', s=300, edgecolors='black', zorder=5)
        ax.set_facecolor('black')
    
    # Turn off any unused subplots.
    for idx in range(n, rows * cols):
        axes_flat[idx].axis("off")
    
    plt.tight_layout()
    plt.show()
