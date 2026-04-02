"""
Visualizes the CI job dependency DAG and matrix heatmap.
Run directly: python tools/visualize_dag.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from core.matrix_parser import MatrixParser, JobConfig
from core.dependency_resolver import DependencyResolver


BACKEND_COLORS = {
    "cpu": "#4CAF50",
    "mpi": "#2196F3",
    "cuda-stub": "#FF9800",
}

OS_SHORT = {
    "ubuntu-22.04": "ubuntu",
    "macos-13": "macos",
    "windows-2022": "windows",
}


def build_graph(jobs: list[JobConfig], resolver: DependencyResolver) -> nx.DiGraph:
    G = nx.DiGraph()
    for job in jobs:
        G.add_node(
            job.job_id,
            backend=job.backend,
            os=job.os,
            python=job.python,
            excluded=job.excluded,
        )
    for edge in resolver.edges:
        G.add_edge(edge.source, edge.target, reason=edge.reason)
    return G


def plot_dag(
    parser: MatrixParser,
    resolver: DependencyResolver,
    output_path: Path = Path("outputs/dag.png"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_jobs = parser.active_jobs()
    G = build_graph(active_jobs, resolver)

    waves = resolver.topological_waves()
    wave_map = {}
    for wave_idx, wave in enumerate(waves):
        for job_id in wave:
            wave_map[job_id] = wave_idx

    # Layout: x = wave index, y = position within wave
    pos = {}
    for wave_idx, wave in enumerate(waves):
        n = len(wave)
        for rank, job_id in enumerate(sorted(wave)):
            x = wave_idx * 3.5
            y = (n - 1) / 2 - rank
            pos[job_id] = (x, y)

    fig, ax = plt.subplots(figsize=(16, 10))

    node_colors = [BACKEND_COLORS.get(G.nodes[n]["backend"], "#999") for n in G.nodes]
    node_labels = {n: "\n".join(n.split("-")[-2:]) for n in G.nodes}

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                            font_size=6, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555",
                           arrows=True, arrowsize=20,
                           connectionstyle="arc3,rad=0.1",
                           min_source_margin=30, min_target_margin=30)
    edge_labels = {(e.source, e.target): "cache\ndep" for e in resolver.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 ax=ax, font_size=6, font_color="#333")

    # Wave labels
    for wave_idx, wave in enumerate(waves):
        ax.text(wave_idx * 3.5, max(pos[n][1] for n in wave) + 1.0,
                f"Wave {wave_idx + 1}\n({len(wave)} parallel)",
                ha="center", fontsize=9, color="#333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eee", alpha=0.8))

    # Legend
    patches = [
        mpatches.Patch(color=c, label=b.upper())
        for b, c in BACKEND_COLORS.items()
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=9)

    ax.set_title(
        "gprMax CI Job Dependency DAG\n"
        "Nodes = jobs, Edges = dependencies (MPI waits for CPU cache)",
        fontsize=12, pad=20
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"DAG saved to {output_path}")
    plt.close()


def plot_matrix_heatmap(
    parser: MatrixParser,
    job_results: list | None = None,
    output_path: Path = Path("outputs/matrix_heatmap.png"),
):
    """
    3-panel heatmap: one per backend.
    Rows = OS, Columns = Python version.
    Color = pass (green) / fail (red) / excluded (grey) / not-run (white).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_jobs = parser.expand_matrix()
    axes = parser.get_axes()
    oses = axes["os"]
    pythons = axes["python"]
    backends = axes["backend"]

    # Build result lookup
    result_lookup: dict[str, str] = {}
    if job_results:
        for r in job_results:
            result_lookup[r.job_id] = "pass" if r.success else "fail"

    fig, axes_panels = plt.subplots(1, len(backends), figsize=(5 * len(backends), 4))
    if len(backends) == 1:
        axes_panels = [axes_panels]

    COLOR_MAP = {
        "pass": "#4CAF50",
        "fail": "#F44336",
        "excluded": "#9E9E9E",
        "not_run": "#EEEEEE",
    }

    for panel_idx, backend in enumerate(backends):
        ax = axes_panels[panel_idx]
        data = np.zeros((len(oses), len(pythons)))
        cell_colors = []

        for i, os_ in enumerate(oses):
            row_colors = []
            for j, python in enumerate(pythons):
                job_id = f"{os_}-py{python}-{backend}"
                job = next((jb for jb in all_jobs if jb.job_id == job_id), None)

                if job and job.excluded:
                    row_colors.append(COLOR_MAP["excluded"])
                elif job_id in result_lookup:
                    row_colors.append(COLOR_MAP[result_lookup[job_id]])
                else:
                    row_colors.append(COLOR_MAP["not_run"])
            cell_colors.append(row_colors)

        for i in range(len(oses)):
            for j in range(len(pythons)):
                ax.add_patch(plt.Rectangle(
                    (j, len(oses) - 1 - i), 1, 1,
                    color=cell_colors[i][j], ec="white", lw=2
                ))
                label_map = {
                    COLOR_MAP["pass"]: "✓",
                    COLOR_MAP["fail"]: "✗",
                    COLOR_MAP["excluded"]: "—",
                    COLOR_MAP["not_run"]: "?",
                }
                ax.text(j + 0.5, len(oses) - 0.5 - i,
                        label_map[cell_colors[i][j]],
                        ha="center", va="center", fontsize=14)

        ax.set_xlim(0, len(pythons))
        ax.set_ylim(0, len(oses))
        ax.set_xticks([x + 0.5 for x in range(len(pythons))])
        ax.set_xticklabels([f"py{p}" for p in pythons])
        ax.set_yticks([y + 0.5 for y in range(len(oses))])
        ax.set_yticklabels([OS_SHORT.get(o, o) for o in reversed(oses)])
        ax.set_title(f"Backend: {backend.upper()}", fontsize=11, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=c, label=k.replace("_", " ").title())
        for k, c in COLOR_MAP.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("gprMax CI Matrix: Pass / Fail / Excluded", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Matrix heatmap saved to {output_path}")
    plt.close()


def plot_cache_hit_rate(
    decisions,
    output_path: Path = Path("outputs/cache_hit_rate.png"),
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pr_nums = [d.pr_number for d in decisions]
    colors = [
        "#4CAF50" if d.decision == "HIT"
        else "#FF9800" if d.decision == "PREFIX_HIT"
        else "#F44336"
        for d in decisions
    ]
    build_times = [
        0.5 if d.decision in ("HIT", "PREFIX_HIT") else 4.5
        for d in decisions
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.bar(pr_nums, [1] * len(pr_nums), color=colors, edgecolor="white")
    ax1.set_xticks(pr_nums)
    ax1.set_xticklabels([f"PR#{n}" for n in pr_nums], rotation=45, ha="right")
    ax1.set_yticks([])
    ax1.set_title("Cache Decision per PR", fontsize=11)
    patches = [
        mpatches.Patch(color="#4CAF50", label="HIT"),
        mpatches.Patch(color="#FF9800", label="PREFIX HIT"),
        mpatches.Patch(color="#F44336", label="MISS"),
    ]
    ax1.legend(handles=patches, loc="upper right", fontsize=9)

    ax2.bar(pr_nums, build_times,
            color=["#4CAF50" if t < 1 else "#F44336" for t in build_times],
            edgecolor="white")
    ax2.axhline(y=4.5, color="#F44336", linestyle="--", alpha=0.5, label="Cold build (4.5 min)")
    ax2.axhline(y=0.5, color="#4CAF50", linestyle="--", alpha=0.5, label="Cached build (0.5 min)")
    ax2.set_xticks(pr_nums)
    ax2.set_xticklabels([f"PR#{n}" for n in pr_nums], rotation=45, ha="right")
    ax2.set_ylabel("Build time (minutes)")
    ax2.set_title("Estimated Build Time per PR", fontsize=11)
    ax2.legend(fontsize=9)

    hit_rate = sum(1 for d in decisions if d.decision in ("HIT", "PREFIX_HIT")) / len(decisions)
    saved = sum(4.5 - 0.5 for d in decisions if d.decision in ("HIT", "PREFIX_HIT"))
    fig.suptitle(
        f"Cache Hit Rate: {hit_rate*100:.0f}%  |  Time Saved: {saved:.1f} min",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Cache hit rate chart saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    config_path = Path(__file__).parent.parent / "config" / "matrix.yml"
    parser = MatrixParser(config_path)
    active = parser.active_jobs()
    resolver = DependencyResolver(active)
    resolver.build_gprmax_dependencies()

    plot_dag(parser, resolver)
    plot_matrix_heatmap(parser)
    print("Done. Check outputs/ directory.")