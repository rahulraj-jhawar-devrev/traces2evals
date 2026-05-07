import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fcluster, linkage

from open_standard_evaluation.config import ClusteringConfig
from open_standard_evaluation.models.facet import Facet


def run_clustering(
    facets: list[Facet],
    embeddings: np.ndarray,
    config: ClusteringConfig,
) -> tuple[np.ndarray, int, list[dict[int, int]]]:
    """
    Full COMPASS clustering pipeline.
    Returns (labels, n_clusters, hierarchy_levels).
    """
    if config.auto_tune:
        params = _auto_tune(embeddings, config.auto_tune_trials)
        config = ClusteringConfig(**{**config.model_dump(), **params})

    labels, n_clusters = _cluster_pipeline(embeddings, config)
    labels = _prune_by_sessions(facets, labels, config.min_sessions_per_cluster)

    # Relabel after pruning
    unique = sorted(set(labels) - {-1})
    remap = {old: new for new, old in enumerate(unique)}
    remap[-1] = -1
    labels = np.array([remap[l] for l in labels])
    n_clusters = len(unique)

    hierarchy = _build_hierarchy(n_clusters, embeddings, labels, config.max_group_size)

    # Quality metrics
    metrics = _compute_quality(embeddings, labels)
    print(f"  Clusters: {n_clusters} | Noise: {(labels == -1).sum()} | Silhouette: {metrics.get('silhouette_score', 'N/A')}")

    return labels, n_clusters, hierarchy


def _cluster_pipeline(embeddings: np.ndarray, config: ClusteringConfig) -> tuple[np.ndarray, int]:
    """UMAP → HDBSCAN → Merge → Prune small → Reassign noise."""
    import hdbscan
    import umap

    # UMAP reduction
    clustering_emb = embeddings
    if config.umap_n_components > 0 and embeddings.shape[1] > config.umap_n_components:
        print(f"  UMAP: {embeddings.shape[1]}d → {config.umap_n_components}d")
        reducer = umap.UMAP(
            n_components=config.umap_n_components,
            metric="cosine",
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            random_state=42,
        )
        clustering_emb = reducer.fit_transform(embeddings)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        cluster_selection_method=config.cluster_selection_method,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(clustering_emb)
    print(f"  HDBSCAN: {len(set(labels) - {-1})} clusters, {(labels == -1).sum()} noise")

    # Merge similar
    centroids = _compute_centroids(embeddings, labels)
    labels = _merge_similar(labels, centroids, config.merge_similarity_threshold, embeddings)

    # Prune small clusters
    n_total = len(labels)
    min_size = max(2, int(n_total * config.min_cluster_pct / 100))
    counts = pd.Series(labels).value_counts()
    for cluster_id in counts[(counts < min_size) & (counts.index != -1)].index:
        labels[labels == cluster_id] = -1

    # Reassign noise
    centroids = _compute_centroids(embeddings, labels)
    if centroids:
        noise_mask = labels == -1
        if noise_mask.any():
            centroid_ids = sorted(centroids.keys())
            centroid_matrix = np.array([centroids[c] for c in centroid_ids])
            sims = cosine_similarity(embeddings[noise_mask], centroid_matrix)
            nearest = sims.argmax(axis=1)
            for i, idx in enumerate(np.where(noise_mask)[0]):
                labels[idx] = centroid_ids[nearest[i]]

    # Relabel consecutively
    unique = sorted(set(labels) - {-1})
    remap = {old: new for new, old in enumerate(unique)}
    remap[-1] = -1
    labels = np.array([remap[l] for l in labels])
    return labels, len(unique)


def _compute_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        if mask.any():
            centroids[cid] = embeddings[mask].mean(axis=0)
    return centroids


def _merge_similar(
    labels: np.ndarray, centroids: dict, threshold: float, embeddings: np.ndarray
) -> np.ndarray:
    if len(centroids) <= 1:
        return labels
    centroid_matrix = np.array([centroids[k] for k in sorted(centroids.keys())])
    sims = cosine_similarity(centroid_matrix)
    label_list = sorted(centroids.keys())
    merge_map = {l: l for l in label_list}

    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            if sims[i, j] > threshold:
                old = label_list[j]
                new = merge_map[label_list[i]]
                for k, v in merge_map.items():
                    if v == old:
                        merge_map[k] = new

    new_labels = np.array([merge_map.get(l, l) if l != -1 else -1 for l in labels])
    return new_labels


def _prune_by_sessions(facets: list[Facet], labels: np.ndarray, min_sessions: int) -> np.ndarray:
    """Prune clusters with fewer than min_sessions unique sessions."""
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        session_ids = {facets[i].session_id for i in range(len(facets)) if mask[i]}
        if len(session_ids) < min_sessions:
            labels[mask] = -1
    return labels


def _build_hierarchy(
    n_clusters: int, embeddings: np.ndarray, labels: np.ndarray, max_group_size: int
) -> list[dict[int, int]]:
    """Recursively group clusters via Ward linkage."""
    if n_clusters <= max_group_size:
        return []

    centroids = _compute_centroids(embeddings, labels)
    levels = []
    current_ids = list(range(n_clusters))
    current_centroids = centroids

    while len(current_ids) > max_group_size:
        centroid_matrix = np.array([current_centroids[cid] for cid in current_ids])
        Z = linkage(centroid_matrix, method="ward")
        n_groups = max(2, int(np.ceil(len(current_ids) ** 0.5)))
        group_arr = fcluster(Z, t=n_groups, criterion="maxclust")

        raw_groups = sorted(set(group_arr))
        group_remap = {old: new for new, old in enumerate(raw_groups)}
        level_map = {cid: group_remap[group_arr[i]] for i, cid in enumerate(current_ids)}
        levels.append(level_map)

        new_ids = sorted(set(level_map.values()))
        new_centroids = {}
        for gid in new_ids:
            members = [cid for cid, g in level_map.items() if g == gid]
            new_centroids[gid] = np.array([current_centroids[c] for c in members]).mean(axis=0)
        current_ids = new_ids
        current_centroids = new_centroids

    return levels


def _compute_quality(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    valid_mask = labels != -1
    if valid_mask.sum() < 10:
        return {}
    valid_emb = embeddings[valid_mask]
    valid_labels = labels[valid_mask]
    try:
        return {
            "silhouette_score": round(float(silhouette_score(valid_emb, valid_labels, metric="cosine", sample_size=min(5000, len(valid_labels)))), 4),
            "davies_bouldin_score": round(float(davies_bouldin_score(valid_emb, valid_labels)), 4),
            "calinski_harabasz_score": round(float(calinski_harabasz_score(valid_emb, valid_labels)), 2),
        }
    except Exception:
        return {}


def _auto_tune(embeddings: np.ndarray, n_trials: int) -> dict:
    """Optuna hyperparameter optimization for clustering."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        import hdbscan
        import umap

        umap_n = trial.suggest_int("umap_n_components", 8, 25)
        reducer = umap.UMAP(n_components=umap_n, metric="cosine", n_neighbors=trial.suggest_int("umap_n_neighbors", 5, 50), min_dist=trial.suggest_float("umap_min_dist", 0.0, 0.3), random_state=42)
        reduced = reducer.fit_transform(embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=trial.suggest_int("min_cluster_size", 3, 20), min_samples=trial.suggest_int("min_samples", 1, 5), cluster_selection_method=trial.suggest_categorical("cluster_selection_method", ["leaf", "eom"]), metric="euclidean")
        labels = clusterer.fit_predict(reduced)
        valid = labels != -1
        if valid.sum() < 10:
            return -999
        try:
            sil = silhouette_score(embeddings[valid], labels[valid], metric="cosine", sample_size=min(5000, valid.sum()))
        except Exception:
            return -999
        n_clusters = len(set(labels) - {-1})
        penalty = max(0, (15 - n_clusters)) * 0.01 + max(0, (n_clusters - 70)) * 0.01
        return sil - penalty

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Auto-tune best: {study.best_params}")
    return study.best_params
