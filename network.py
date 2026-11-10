import math
import random
from collections import defaultdict
from colorsys import hsv_to_rgb
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


class SupplyChainNetwork:
    """Graph class to model supply chain networks."""

    _GRAPH_BUILDERS = {
        "random": "create_random_graph",
        "scale_free": "create_scale_free_graph",
    }

    def __init__(self):
        self._subnetworks = None
        # Sampler state
        self._sampled_subnetworks: dict = {}   # edge_key -> [H, log_w, hits]
        self._incoming_colours_cache = None


    @classmethod
    def from_file(cls, filename: str, seed: int | None = None):
        obj = cls()
        num_colors = obj._build_network(filename)
        num_nodes = obj.G.number_of_nodes()
        num_edges = obj.G.number_of_edges()
        method = "file"
        rho = 1
        obj._initialize_parameters(num_nodes, num_edges, num_colors, method, rho, seed)
        obj._initialize_palette(num_colors)
        return obj


    @classmethod
    def generate(
        cls,
        *,
        num_nodes: int,
        num_edges: int,
        num_colors: int,
        method: str = "random",
        seed: int | None = None,
        rho: int = 2,
    ):
        obj = cls()
        obj._initialize_parameters(num_nodes, num_edges, num_colors, method, rho, seed)
        obj._initialize_palette(num_colors)
        obj._build_generated_graph(method)
        return obj


    def _initialize_parameters(self, num_nodes, num_edges, num_colors, method, rho, seed):
        self._validate_parameters(num_nodes, num_edges, num_colors, method, rho)
        self.method = method
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_colors = num_colors
        self.rho = rho
        self.rng = random.Random(seed)


    def _validate_parameters(self, num_nodes, num_edges, num_colors, method, rho):
        if num_nodes < 1:
            raise ValueError("Number of nodes must be positive.")
        if method == "random":
            max_edges = num_colors * num_nodes * (num_nodes - 1)
            if not (num_nodes <= num_edges <= max_edges):
                raise ValueError(
                    "Strongly connected graph is not possible with given parameters."
                )
        if rho <= 0:
            raise ValueError("Maximum edge weight must be positive.")


    def _initialize_palette(self, num_colors: int):
        self.palette = [
            self._color_from_index(i, num_colors)
            for i in range(num_colors)
        ]


    @staticmethod
    def _color_from_index(i: int, n: int) -> str:
        h = i / n
        r, g, b = hsv_to_rgb(h, s=0.7, v=0.9)
        return "#{:02x}{:02x}{:02x}".format(int(255*r), int(255*g), int(255*b))


    def _build_generated_graph(self, method):
        try:
            builder_name = self._GRAPH_BUILDERS[method]
        except KeyError:
            raise ValueError(f"Unknown graph generation method: {method}")
        getattr(self, builder_name)()


    @property
    def subnetworks(self):
        if self._subnetworks is None:
            self._subnetworks = self.calculate_subnetworks()
        return self._subnetworks


    def _build_network(self, filename: str) -> int:
        import csv
        self.G = nx.MultiDiGraph()
        colors = []
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                u, v, business = row
                start = business.find(":") + 1
                end = business.find("}", start)
                business_type = business[start:end].strip()
                if business_type not in colors:
                    colors.append(business_type)
                c = colors.index(business_type)
                self.G.add_edge(u, v, key=c, color=c, weight=1)
        return len(colors)


    def create_random_graph(self):
        self.G = nx.MultiDiGraph()
        used = set()
        for v in range(self.num_nodes):
            while True:
                u = self.rng.randrange(self.num_nodes)
                if u == v:
                    continue
                c = self.rng.randrange(self.num_colors)
                if (u, v, c) in used:
                    continue
                w = self.rng.uniform(0, self.rho)
                self.G.add_edge(u, v, key=c, color=c, weight=w)
                used.add((u, v, c))
                break
        remaining = self.num_edges - self.num_nodes
        while remaining > 0:
            u = self.rng.randrange(self.num_nodes)
            v = self.rng.randrange(self.num_nodes)
            if u == v:
                continue
            c = self.rng.randrange(self.num_colors)
            if (u, v, c) in used:
                continue
            w = self.rng.uniform(0, self.rho)
            self.G.add_edge(u, v, key=c, color=c, weight=w)
            used.add((u, v, c))
            remaining -= 1


    def create_scale_free_graph(self, rng: random.Random):
        base_G = nx.scale_free_graph(self.num_nodes, seed=self.rng.randint(0, 10**6))
        self.G = nx.MultiDiGraph()
        for u, v in base_G.edges():
            c = rng.randint(0, self.num_colors - 1)
            w = rng.uniform(0, self.rho)
            self.G.add_edge(u, v, key=c, color=c, weight=w)
        if not nx.is_weakly_connected(self.G):
            order = list(self.G.nodes)
            rng.shuffle(order)
            for i in range(len(order)):
                u = order[i]
                v = order[(i + 1) % len(order)]
                c = rng.randint(0, self.num_colors - 1)
                w = rng.uniform(0, self.rho)
                self.G.add_edge(u, v, key=c, color=c, weight=w)
        self.num_edges = self.G.number_of_edges()


    def diversify(self, p: float | int, lam: float = 0.5):
        if not (0 < p <= 1):
            raise ValueError("Percentage of nodes must be 0 < p <= 1")
        if not (0 < lam < 1):
            raise ValueError("Diversification factor must be 0 < lam < 1")

        nodes_with_color = defaultdict(set)
        edges = list(self.G.edges(keys=True, data=True))
        for _, v, _, data in edges:
            nodes_with_color[data["color"]].add(v)

        num_edges = round(p * self.G.number_of_edges())
        edges_to_diversify = self.rng.sample(edges, num_edges)
        diversified = set()

        for u, v, key, data in edges_to_diversify:
            if (u, v, key) in diversified:
                continue
            diversified.add((u, v, key))
            color = data["color"]
            weight = data["weight"]
            destinations = nodes_with_color[color]
            candidates = destinations - {v}
            if u in destinations:
                candidates = candidates | {u}
            if not candidates:
                continue
            v2 = self.rng.choice(tuple(candidates))
            new_weight = (1 - lam) * weight
            for _, d2 in self.G[u][v2].items() if self.G.has_edge(u, v2) else []:
                if d2["color"] == color:
                    d2["weight"] += new_weight
                    break
            else:
                self.G.add_edge(u, v2, key=color, color=color, weight=new_weight)
            self.G[u][v][key]['weight'] = lam * weight

        self.num_edges = self.G.number_of_edges()
        self._subnetworks = None
        # Invalidate sampler state — graph has changed
        self._sampled_subnetworks = {}
        self._incoming_colours_cache = None


    def _materialize_subgraph(self, star_defs, chosen_star_indices):
        H = nx.MultiDiGraph()
        for idx in chosen_star_indices:
            target, color, sources = star_defs[idx]
            for source in sources:
                for key, data in self.G[source][target].items():
                    if data["color"] == color:
                        H.add_edge(source, target, key=key, **data)
                        break
        return H


    def calculate_subnetworks(self):
        """Return all minimal SCSs via exhaustive DFS."""
        G = self.G
        star_defs = []
        star_of = defaultdict(list)

        for v in G.nodes:
            by_color = defaultdict(list)
            for u, _, data in G.in_edges(v, data=True):
                by_color[data["color"]].append(u)
            for c, sources in by_color.items():
                idx = len(star_defs)
                star_defs.append((v, c, tuple(sources)))
                star_of[v].append(idx)

        results = []
        seen_edge_sets = set()

        def dfs(chosen, color_of, frontier):
            while frontier and frontier[-1] in color_of:
                frontier.pop()
            if not frontier:
                H = self._materialize_subgraph(star_defs, chosen)
                key = frozenset(H.edges(keys=True))
                if key not in seen_edge_sets:
                    seen_edge_sets.add(key)
                    results.append(H)
                return
            w = frontier.pop()
            for idx in star_of.get(w, []):
                _, c, sources = star_defs[idx]
                chosen.add(idx)
                color_of[w] = c
                new_nodes = [u for u in sources if u not in color_of]
                dfs(chosen, color_of, frontier + new_nodes)
                chosen.discard(idx)
                del color_of[w]

        for seed_idx, (v, c, sources) in enumerate(star_defs):
            chosen = {seed_idx}
            color_of = {v: c}
            frontier = list(sources)
            dfs(chosen, color_of, frontier)

        if not results:
            return results
        edge_sets = [frozenset(H.edges(keys=True)) for H in results]
        return [
            results[i] for i, es_i in enumerate(edge_sets)
            if not any(es_j < es_i for j, es_j in enumerate(edge_sets) if i != j)
        ]


    def visualize_supply_chain(self, pos=None, weight_range=(1.2, 4.0)):
        self.visualize_network(self.G, pos=pos, weight_range=weight_range)


    def visualize_subnetworks(self):
        for i, H in enumerate(self.subnetworks):
            self.visualize_network(H, title=f"Subnetwork {i + 1}")


    def visualize_network(self, G: nx.MultiDiGraph, title=None, pos=None, weight_range=(1.2, 4.0)):
        if pos is None:
            pos = nx.spring_layout(G, seed=0)

        cmap = mcolors.ListedColormap(self.palette)
        vmin, vmax = 0, self.num_colors - 1
        fig, ax = plt.subplots(figsize=(7, 7))

        nx.draw_networkx_nodes(
            G, pos, node_size=350, node_color="white",
            edgecolors="black", linewidths=1.2, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, labels={v: str(v) for v in G}, font_size=10, ax=ax
        )

        w_raw = [d["weight"] for _, _, _, d in G.edges(keys=True, data=True)]
        wr_min, wr_max = (min(w_raw), max(w_raw)) if w_raw else (1, 1)
        wp_min, wp_max = weight_range
        scale = ((wp_max - wp_min) / (wr_max - wr_min)) if wr_max != wr_min else 0

        def width(w):
            return wp_min + (w - wr_min) * scale if scale else (wp_min + wp_max) / 2

        slot = defaultdict(int)
        for u, nbrdict in G.adj.items():
            for v, keyedgedict in nbrdict.items():
                pair = frozenset((u, v))
                for _, data in keyedgedict.items():
                    k = slot[pair]
                    slot[pair] += 1
                    sign = 1 if k & 1 else -1
                    mag  = (k + 1) // 2
                    rad  = sign * 0.15 * mag
                    if u > v:
                        rad = -rad
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[(u, v)],
                        edge_color=self.palette[data["color"]],
                        edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax,
                        width=width(data["weight"]),
                        arrows=True, arrowstyle="-|>", arrowsize=15,
                        connectionstyle=f"arc3,rad={rad}", ax=ax
                    )

        present_cols = {d["color"] for _, _, _, d in G.edges(keys=True, data=True)}
        handles = [mpatches.Patch(color=self.palette[i], label=f"Color {i}")
                   for i in sorted(present_cols)]
        if handles:
            ax.legend(handles=handles, title="Edge Colors",
                      loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.axis("off")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        plt.show()
