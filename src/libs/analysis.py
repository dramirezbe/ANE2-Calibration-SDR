import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

class PSDHistogram:
    def __init__(self, datos_nodos, global_range=(-27, 6), alpha_node=0.72):
        self.datos_nodos = datos_nodos
        self.labels = list(datos_nodos.keys())
        self.n_nodes = len(self.labels)
        self.global_range = global_range
        self.alpha_node = alpha_node
        
        # Dynamically find pxx_len from the first available dataframe
        pxx_len = None
        for df in datos_nodos.values():
            if not df.empty:
                val = df['pxx'].iloc[0]
                pxx_len = len(ast.literal_eval(val) if isinstance(val, str) else val)
                break

        if pxx_len is None:
            raise ValueError("No valid 'pxx' data found in any node's dataframe.")
        
        # Calculate bins based on dynamic length
        self.n_bins = int(np.ceil((np.log2(pxx_len) + np.sqrt(pxx_len)) / 2))
        self.bin_edges = np.linspace(global_range[0], global_range[1], self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        
        # State containers
        self.global_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.node_counts = {}
        self.node_stats = {}
        self.total_values = 0

    def parse_data(self):
        print(f"Parsing pxx across {self.n_nodes} nodes...")
        clipped = 0

        for label in tqdm(self.labels, total=self.n_nodes):
            df = self.datos_nodos[label]
            arrs = []

            for val in df['pxx']:
                if isinstance(val, str):
                    arr = np.array(ast.literal_eval(val), dtype=np.float32)
                else:
                    arr = np.asarray(val, dtype=np.float32)
                arrs.append(arr)

            if not arrs:
                continue

            flat = np.concatenate(arrs)
            self.total_values += flat.size
            clipped += np.sum((flat < self.global_range[0]) | (flat > self.global_range[1]))

            counts, _ = np.histogram(flat, bins=self.bin_edges)
            self.node_counts[label] = counts
            self.global_counts += counts
            self.node_stats[label] = {'mean': float(np.mean(flat)), 'std': float(np.std(flat))}

        print(f"\nParsed {self.total_values:,} values total")
        if clipped:
            print(f"  {clipped:,} values ({(100 * clipped / self.total_values):.2f}%) fall outside {self.global_range}")

        bin_width = self.bin_edges[1] - self.bin_edges[0]
        self.global_density = self.global_counts / (self.global_counts.sum() * bin_width)
        self.node_densities = {lbl: c / (c.sum() * bin_width) for lbl, c in self.node_counts.items()}

    def plot_distributions(self):
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i / max(self.n_nodes - 1, 1)) for i in range(self.n_nodes)]

        fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.35}, layout='constrained')

        # Panel A: Overlay
        ax = axes[0]
        ax.fill_between(self.bin_centers, self.global_density, alpha=0.15, color='steelblue', label='_nolegend_')
        ax.plot(self.bin_centers, self.global_density, color='steelblue', lw=2.2, label=f'All nodes  (N={self.total_values:,})')

        for (lbl, density), color in zip(self.node_densities.items(), colors):
            mu = self.node_stats[lbl]['mean']
            ax.plot(self.bin_centers, density, lw=1.0, alpha=self.alpha_node, color=color, label=f"{lbl}  μ={mu:.1f} dB")

        ax.set_xlim(self.global_range)
        ax.set_xlabel('Power Spectral Density (dB)', fontsize=11)
        ax.set_ylabel('Probability Density', fontsize=11)
        ax.set_title('PSD Distribution — All Nodes (per-node overlay)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2, loc='upper right')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)

        # Panel B: Mean ± 1σ
        ax2 = axes[1]
        means = np.array([self.node_stats[l]['mean'] for l in self.labels])
        stds = np.array([self.node_stats[l]['std'] for l in self.labels])
        xs = np.arange(len(self.labels))

        ax2.bar(xs, stds, bottom=means - stds, alpha=0.4, color=colors, width=0.6, label='μ ± 1σ')
        ax2.scatter(xs, means, color=colors, zorder=5, s=60, edgecolors='black', lw=0.7)
        ax2.axhline(np.mean(means), color='steelblue', lw=1.5, ls='--', alpha=0.8, label=f'Global mean = {np.mean(means):.1f} dB')

        ax2.set_xticks(xs)
        ax2.set_xticklabels(self.labels, rotation=25, ha='right', fontsize=9)
        ax2.set_ylabel('PSD (dB)', fontsize=10)
        ax2.set_title('Per-Node Mean ± 1σ', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
        ax2.grid(True, axis='y', alpha=0.3)

        plt.show()

    def print_statistics(self):
        means = np.array([self.node_stats[l]['mean'] for l in self.labels])
        
        print(f"\n{'Node':<20} {'Mean (dB)':>10} {'Std (dB)':>10} {'Rows':>6} {'Values':>12}")
        print('-' * 62)
        for lbl in self.labels:
            n_rows = len(self.datos_nodos[lbl]) 
            n_vals = self.node_counts[lbl].sum()
            print(f"{lbl:<20} {self.node_stats[lbl]['mean']:>10.2f} {self.node_stats[lbl]['std']:>10.2f} {n_rows:>6} {n_vals:>12,}")
        print('-' * 62)
        print(f"{'GLOBAL':<20} {np.mean(means):>10.2f} {float(np.std(means)):>10.2f} {'':>6} {self.total_values:>12,}")

        # Similarity Ranking
        density_matrix = np.stack([self.node_densities[lbl] for lbl in self.labels])
        corr_matrix = np.corrcoef(density_matrix)
        abs_cross_means = (np.abs(corr_matrix).sum(axis=1) - 1.0) / (self.n_nodes - 1)
        ranked = sorted(zip(self.labels, abs_cross_means), key=lambda x: x[1], reverse=True)

        print("\n📊 Ranked by absolute mean cross-node similarity:")
        mean_val, std_val = abs_cross_means.mean(), abs_cross_means.std()
        for rank, (label, cm) in enumerate(ranked, 1):
            flag = "  ⚠️  outlier" if cm < mean_val - std_val else ""
            print(f"  {rank}. {label:<22} | abs mean r = {cm:.4f}{flag}")

    def run(self):
        self.parse_data()
        self.plot_distributions()
        self.print_statistics()