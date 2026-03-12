import ast
import numpy as np
import pandas as pd
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
        
        pxx_len = 1024 # Fallback
        for df in datos_nodos.values():
            if not df.empty:
                val = df['pxx'].iloc[0]
                pxx_len = len(ast.literal_eval(val) if isinstance(val, str) else val)
                break

        self.n_bins = int(np.ceil((np.log2(pxx_len) + np.sqrt(pxx_len)) / 2))
        self.bin_edges = np.linspace(global_range[0], global_range[1], self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        self.bin_width = self.bin_edges[1] - self.bin_edges[0]
        
        self.global_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.node_counts = {}
        self.node_densities = {}
        self.means = np.zeros(self.n_nodes)
        self.stds = np.zeros(self.n_nodes)
        self.total_values = 0

    @staticmethod
    def parse_pxx_cell(pxx_raw):
        if isinstance(pxx_raw, np.ndarray): return pxx_raw.astype(float).ravel()
        if isinstance(pxx_raw, (list, tuple)): return np.array(pxx_raw, dtype=float).ravel()
        if isinstance(pxx_raw, pd.Series): return pxx_raw.to_numpy(dtype=float).ravel()
        
        s = str(pxx_raw).strip()
        if s.startswith("[") and s.endswith("]"):
            return np.fromiter(map(float, s[1:-1].replace(",", " ").split()), dtype=float)
        return np.asarray(ast.literal_eval(s), dtype=float).ravel()

    def parse_data(self):
        print(f"Parsing pxx across {self.n_nodes} nodes for Histograms...")
        for i, label in enumerate(tqdm(self.labels)):
            df = self.datos_nodos[label]
            flat = np.concatenate([self.parse_pxx_cell(val) for val in df['pxx']])
            self.total_values += flat.size
            
            counts, _ = np.histogram(flat, bins=self.bin_edges, range=self.global_range)
            self.node_counts[label] = counts
            self.global_counts += counts
            
            self.means[i] = np.mean(flat)
            self.stds[i] = np.std(flat)

        c_sum_global = self.global_counts.sum()
        self.global_density = self.global_counts / (c_sum_global * self.bin_width) if c_sum_global else self.global_counts
        
        for lbl, c in self.node_counts.items():
            c_sum = c.sum()
            self.node_densities[lbl] = c / (c_sum * self.bin_width) if c_sum > 0 else np.zeros_like(c, dtype=float)

    def get_correlation_scores(self):
        if not self.node_densities:
            self.parse_data()
            
        density_matrix = np.array([self.node_densities[lbl] for lbl in self.labels])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(density_matrix)
            
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 0.0) 
        abs_cross_means = np.sum(np.abs(corr_matrix), axis=1) / (self.n_nodes - 1)
        return pd.Series(abs_cross_means, index=self.labels)

    def get_single_row_vectors(self):
        """Extracts a single representative row per node for MI alignment."""
        vectors = {}
        for lbl, df in self.datos_nodos.items():
            if 'pxx' not in df.columns:
                continue
            
            # Check up to row 104 to find the first valid parse, mimicking the ipynb logic
            for row in range(min(105, len(df))):
                try:
                    candidate = self.parse_pxx_cell(df["pxx"].iloc[row])
                    if candidate.size > 0:
                        vectors[lbl] = candidate
                        break
                except Exception:
                    continue
        return vectors

    @staticmethod
    def mutual_information_hist(x, y, bins=64, edges=None, base=2.0, normalized=False):
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        
        H, _, _ = np.histogram2d(x, y, bins=[edges, edges] if edges is not None else bins)
        total_H = np.sum(H)
        if total_H == 0:
            return 0.0
            
        Pxy = H / total_H
        Px, Py = np.sum(Pxy, axis=1), np.sum(Pxy, axis=0)
        
        Px_Py = Px[:, None] * Py[None, :]
        nz = (Pxy > 0) & (Px_Py > 0)
        mi = np.sum(Pxy[nz] * np.log(Pxy[nz] / Px_Py[nz])) / np.log(base)
        
        if normalized:
            Hx = -np.sum(Px[Px > 0] * np.log(Px[Px > 0])) / np.log(base)
            Hy = -np.sum(Py[Py > 0] * np.log(Py[Py > 0])) / np.log(base)
            mi = mi / np.sqrt(Hx * Hy) if Hx * Hy > 0 else 0.0
            
        return mi

    def MI_matrix(self, bins=64, qrange=(1.0, 99.0), base=2.0, normalized=False):
        print("\nComputing Mutual Information Matrix (Single Row Alignment)...")
        vectors = self.get_single_row_vectors()
        
        if not vectors:
            return pd.Series(0, index=self.labels), pd.DataFrame(), (0, 0)

        all_vals = np.concatenate(list(vectors.values()))
        lo, hi = np.percentile(all_vals, qrange)
        edges = np.linspace(lo, hi, bins + 1)

        valid_labels = list(vectors.keys())
        M = np.zeros((len(valid_labels), len(valid_labels)))
        for i, ni in enumerate(tqdm(valid_labels)):
            for j, nj in enumerate(valid_labels):
                if j < i: continue
                if i == j:
                    M[i, j] = 0.0
                else:
                    M[i, j] = M[j, i] = self.mutual_information_hist(
                        vectors[ni], vectors[nj], edges=edges, base=base, normalized=normalized
                    )

        mi_df = pd.DataFrame(M, index=valid_labels, columns=valid_labels)
        
        scores = pd.Series(M.sum(axis=1), index=valid_labels)
        if scores.max() > 0:
            scores = scores / scores.max()
            
        # Reindex to ensure it matches self.labels (filling missing nodes with 0)
        scores = scores.reindex(self.labels, fill_value=0.0)
        
        return scores, mi_df, (lo, hi)

    def print_and_plot_results(self, corr_scores, mi_scores, alpha=0.5):
        # Format labels to "NodeX" to match the notebook visually
        display_labels = [f"Node{l}" if str(l).isdigit() else str(l) for l in self.labels]
        corr_scores.index = display_labels
        mi_scores.index = display_labels

        df_product = pd.DataFrame({
            'correlation': corr_scores,
            'MI': mi_scores
        }).fillna(0)
        
        # Calculate Weighted Product Score
        df_product['product'] = alpha * df_product['correlation'].abs() + (1 - alpha) * df_product['MI']
        
        # 1. Print Table Sorted by Product (Descending)
        df_print = df_product.sort_values(by='product', ascending=False).reset_index().rename(columns={'index': 'label'})

        print("\n Nodes ranked by Product Score:")
        print("=" * 70)
        print(f"{'Rank':<6} {'Label':<20} {'Corr':<12} {'MI':<12} {'Product':<12}")
        print("-" * 70)
        
        for i, row in df_print.head(10).iterrows():
            print(f"{i+1:<6} {row['label']:<20} "
                  f"{row['correlation']:>11.4f} {row['MI']:>11.4f} {row['product']:>11.4f}")

        # 2. Plot Graph Sorted by MI Score (Descending) - Identical to IPYNB
        df_plot = df_product.sort_values(by='MI', ascending=False).reset_index().rename(columns={'index': 'label'})

        fig, ax1 = plt.subplots(figsize=(15, 6))
        
        plot_labels = df_plot['label'].tolist()
        x_pos = np.arange(len(plot_labels))

        # Plot MI Score on Left Axis
        line1 = ax1.semilogy(x_pos, df_plot['MI'].values, 'b-o', label='MI Score', linewidth=2, markersize=6)
        ax1.set_xlabel('Node Labels', fontsize=12)
        ax1.set_ylabel('Cumulative MI ', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')

        # Plot Product Score on Right Axis
        ax2 = ax1.twinx()
        line2 = ax2.semilogy(x_pos, df_plot['product'].values, 'r-s', label='Product', linewidth=2, markersize=6)
        ax2.set_ylabel(' cum Corr× cum MI', color='r', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='r')

        plt.xticks(x_pos, plot_labels, rotation=45, ha="right")
        
        # Restore Exact IPYNB Limits
        ax1.set_ylim(0.2, 1)
        ax2.set_ylim(0.2, 1)
        
        # Build Combined Legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower left')

        plt.title('Node Ranking: MI vs Product Score', fontsize=14)
        plt.grid(True, alpha=0.54)
        plt.tight_layout()
        plt.show()

    def plot_distributions(self):
        cmap = plt.get_cmap('tab10')
        colors = [cmap(0.5)] if self.n_nodes == 1 else cmap(np.linspace(0, 1, self.n_nodes))
        width = max(13, min(20, 8 + self.n_nodes * 0.3))
        
        fig, axes = plt.subplots(2, 1, figsize=(width, 10), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.35}, layout='constrained')
        ax1, ax2 = axes
        
        ax1.fill_between(self.bin_centers, self.global_density, alpha=0.15, color='steelblue', label='_nolegend_')
        ax1.plot(self.bin_centers, self.global_density, color='steelblue', lw=2.2, label=f'All nodes (N={self.total_values:,})', zorder=100)
        
        for i, lbl in enumerate(self.labels):
            ax1.plot(self.bin_centers, self.node_densities.get(lbl, np.zeros_like(self.bin_centers)), lw=1.0, alpha=self.alpha_node, color=colors[i], label=f"{lbl} μ={self.means[i]:.1f} dB")
        
        ax1.set_xlim(self.global_range)
        ax1.set_xlabel('Power Spectral Density (dB)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('PSD Distribution — All Nodes')
        ax1.legend(ncol=2 if self.n_nodes <= 10 else 3, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        xs = np.arange(self.n_nodes)
        bar_width = min(0.6, 0.8 * (1 / max(1, self.n_nodes / 10)))
        
        ax2.bar(xs, 2 * self.stds, bottom=self.means - self.stds, alpha=0.4, color=colors, width=bar_width, label='μ ± 1σ')
        ax2.scatter(xs, self.means, color=colors, zorder=5, s=60, edgecolors='black')
        ax2.axhline(np.mean(self.means), color='steelblue', lw=1.5, ls='--', alpha=0.8, label=f'Global mean = {np.mean(self.means):.1f} dB')
        
        ax2.set_xticks(xs)
        ax2.set_xticklabels(self.labels, rotation=45 if self.n_nodes > 10 else 25, ha='right')
        ax2.set_ylabel('PSD (dB)')
        ax2.legend(loc='best')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.show()

    def execute_histogram(self):
        self.parse_data()
        self.plot_distributions()

    def exec_cumm_ranking(self, normalized=False, alpha=0.5):
        if not self.node_densities:
            self.parse_data()
        corr_scores = self.get_correlation_scores()
        mi_scores, _, _ = self.MI_matrix(bins=self.n_bins, normalized=normalized)
        self.print_and_plot_results(corr_scores, mi_scores, alpha=alpha)