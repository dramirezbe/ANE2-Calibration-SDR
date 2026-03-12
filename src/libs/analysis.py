import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

class PSDHistogram:
    def __init__(self, datos_nodos, global_range=(-27, 6), alpha_node=0.72):
        self.datos_nodos = datos_nodos
        self.labels = list(datos_nodos.keys())
        self.n_nodes = len(self.labels)
        self.global_range = global_range
        self.alpha_node = alpha_node
        
        pxx_len = 1024
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
        for label in self.labels:
            df = self.datos_nodos[label]
            flat = np.concatenate([self.parse_pxx_cell(val) for val in df['pxx']])
            
            counts, _ = np.histogram(flat, bins=self.bin_edges, range=self.global_range)
            self.node_counts[label] = counts
            self.global_counts += counts

        c_sum_global = self.global_counts.sum()
        self.global_density = self.global_counts / (c_sum_global * self.bin_width) if c_sum_global else self.global_counts
        
        for lbl, c in self.node_counts.items():
            c_sum = c.sum()
            self.node_densities[lbl] = c / (c_sum * self.bin_width) if c_sum > 0 else np.zeros_like(c, dtype=float)

    def get_correlation_scores(self):
        if not self.node_densities: self.parse_data()
        density_matrix = np.array([self.node_densities[lbl] for lbl in self.labels])
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_matrix = np.corrcoef(density_matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 0.0) 
        return pd.Series(np.sum(np.abs(corr_matrix), axis=1) / (self.n_nodes - 1), index=self.labels)

    def get_single_row_vectors(self):
        vectors = {}
        for lbl, df in self.datos_nodos.items():
            if 'pxx' not in df.columns: continue
            for row in range(min(105, len(df))):
                try:
                    candidate = self.parse_pxx_cell(df["pxx"].iloc[row])
                    if candidate.size > 0:
                        vectors[lbl] = candidate
                        break
                except Exception: continue
        return vectors

    @staticmethod
    def mutual_information_hist(x, y, bins=64, edges=None):
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        H, _, _ = np.histogram2d(x, y, bins=[edges, edges] if edges is not None else bins)
        total_H = np.sum(H)
        if total_H == 0: return 0.0
            
        Pxy = H / total_H
        Px_Py = np.sum(Pxy, axis=1)[:, None] * np.sum(Pxy, axis=0)[None, :]
        nz = (Pxy > 0) & (Px_Py > 0)
        return np.sum(Pxy[nz] * np.log(Pxy[nz] / Px_Py[nz])) / np.log(2.0)

    def MI_matrix(self, bins=64):
        vectors = self.get_single_row_vectors()
        if not vectors: return pd.Series(0, index=self.labels)

        all_vals = np.concatenate(list(vectors.values()))
        lo, hi = np.percentile(all_vals, (1.0, 99.0))
        edges = np.linspace(lo, hi, bins + 1)

        valid_labels = list(vectors.keys())
        M = np.zeros((len(valid_labels), len(valid_labels)))
        for i, ni in enumerate(valid_labels):
            for j, nj in enumerate(valid_labels):
                if j < i: continue
                M[i, j] = M[j, i] = 0.0 if i == j else self.mutual_information_hist(vectors[ni], vectors[nj], edges=edges)

        scores = pd.Series(M.sum(axis=1), index=valid_labels)
        if scores.max() > 0: scores = scores / scores.max()
        return scores.reindex(self.labels, fill_value=0.0)

    def execute(self, alpha=0.5):
        if not self.node_densities: self.parse_data()
        
        corr_scores = self.get_correlation_scores()
        mi_scores = self.MI_matrix(bins=self.n_bins)
        
        self.df_product = pd.DataFrame({'correlation': corr_scores, 'MI': mi_scores}).fillna(0)
        self.df_product['product'] = alpha * self.df_product['correlation'].abs() + (1 - alpha) * self.df_product['MI']
        self.df_product.index = [f"Node{l}" if str(l).isdigit() else str(l) for l in self.labels]

    # --- Plot Helpers ---
    def plot_hist(self, **kwargs):
        plt.plot(self.bin_centers, self.global_density, **kwargs)

    def plot_mi(self, **kwargs):
        plt.semilogy(self.df_product.index, self.df_product['MI'], marker='o', **kwargs)