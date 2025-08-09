import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.sparse import csr_matrix, csc_matrix

from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn_extra.cluster import KMedoids

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count


__all__ = [
    'calc_haar_mags',
    'get_otu_abundances',
    'preprocess',
    'proximity_matrix',
    'mds_full_randomized',
    'convert_least_squares',
    'matching_pursuit_lazy_parallel',
    'diag_impo',
    'adaptive',
    'reconstruct_coord',
    'reconstruct',
    'new_biplot3d',
    'new_biplot3dnormalized',
    '_get_correlation',
    'rfgram_plot',
    'make_plots',
    'boxplot_plotter'
]


def calc_haar_mags(haar_basis, abund_vec):
    # Convert to sparse form
    abund_vec = csr_matrix(abund_vec)
    # Normalize abundances
    abund_vec = abund_vec / abund_vec.sum(axis=0)
    # Apply Haar transformation
    mags = haar_basis @ abund_vec
    mags = csr_matrix(mags)
    return mags


def get_otu_abundances(table, tree):
    """ Maps the given OTU abundances onto a reference tree.
        table: pd.DataFrame
        tree: skbio.TreeNode (Newick tree structure)

        Returns:
        - abundvec: np.ndarray (OTU abundances mapped onto the tree)
    """

    return table.reindex(columns=[leaf.name for leaf in tree.tips()], fill_value=0).T.values


def preprocess(label, biom_table, haar_basis, metadata, tree):
    """ For adaptive hld. Converts to necessary types for
        Evan's HLD code to work.

        label: column name in metadata,
        biomtab: biom.Table,
        shl: result from sparsification,
        meta: metadata with `label` not as index,
        returns:  X, Y, mags, dic
        """

    labels = list(metadata[label].unique())
    dic = dict(zip(labels, list(range(1, len(labels)+1))))

    mapped_labels = metadata[label].map(dic).rename("labels")
    X = biom_table.to_dataframe().T.join(
        mapped_labels, how="inner").sort_values("labels")
    Y = X["labels"]
    X = X.drop(columns="labels")

    abundance_vector = get_otu_abundances(X, tree)
    mags = calc_haar_mags(haar_basis, abundance_vector)

    # Save outputs for comparison
    print("X shape:", X.shape)
    print(type(X))
    print(X)
    print("Y shape:", Y.shape)
    print(type(Y))
    print("abundance_vector shape:", abundance_vector.shape)
    print("mags shape:", mags.shape)
    print("haar_basis shape:", haar_basis.shape)

    # Normalize X
    sums = X.to_numpy().sum(axis=1)
    X = X.div(sums, axis=0)
    X = np.array(X)

    return X, Y, mags, dic


def proximity_matrix(clf, X):
    """ Generate random forest affinity matrix
    clf: a classifier (RandomForestClassifier)
    X : data matrix with dimensions n by m 
    lgbm: Model to use (true=LGBM/False)
    """

    print('proximity matrix:')
    terminals = clf.apply(X)
    print('tree leaves preds. shape', terminals.shape)

    nsamples, nTrees = terminals.shape
    prox = np.zeros((nsamples, nsamples))
    for i in range(nTrees):
        a = terminals[:, i]
        prox += 1*np.equal.outer(a, a)
    prox = 1 - (prox / nTrees)
    return prox


def mds_full_randomized(D, dim=50, n_oversamples=10, random_state=None):
    """
    Memory-efficient full MDS using randomized SVD.

    Parameters
    ----------
    D : ndarray of shape (n_samples, n_samples)
        Full distance matrix (symmetric, zero diagonal).
    dim : int
        Target embedding dimension.
    n_oversamples : int
        Extra dimensions to improve accuracy in randomized SVD.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, dim)
        Low-dimensional embedding.
    """
    # Squared distances
    D2 = D ** 2

    # Double centering
    row_mean = np.mean(D2, axis=1, keepdims=True)
    col_mean = np.mean(D2, axis=0, keepdims=True)
    total_mean = np.mean(D2)
    B = -0.5 * (D2 - row_mean - col_mean + total_mean)

    # Randomized truncated SVD
    U, S, _ = randomized_svd(
        B,
        n_components=dim,
        n_oversamples=n_oversamples,
        random_state=random_state
    )

    # Convert eigenvalues to coordinates
    pos_mask = S > 1e-12
    U = U[:, pos_mask]
    S = S[pos_mask]
    X = U * np.sqrt(S)

    return X


def select_hybrid_balanced_medoids(affinity, Y, target_total, random_state=0, verbose=True):
    """
    Hybrid medoid selection:
    - Includes all points from small labels (≤ target_per_label)
    - Uses KMedoids on larger labels to match target_total exactly
    - Maintains original row order
    """
    labels = np.array(Y)
    unique_labels = np.unique(labels)
    print('Y', labels)
    print('unique', unique_labels)
    label_to_indices = {label: np.where(labels == label)[
        0] for label in unique_labels}
    n_labels = len(unique_labels)
    target_per_label = target_total // n_labels

    selected_indices = []
    large_labels = []
    actual_total = 0

    # Step 1: Handle small labels
    for label in unique_labels:
        idxs = label_to_indices[label]
        if len(idxs) <= target_per_label:
            selected_indices.extend(idxs.tolist())
            actual_total += len(idxs)
        else:
            large_labels.append((label, idxs))

    # Step 2: Distribute remaining medoids across larger labels
    remaining_slots = target_total - actual_total
    if verbose:
        print(f"Total from small labels: {actual_total}")
        print(
            f"Remaining slots: {remaining_slots} for {len(large_labels)} labels")

    if large_labels and remaining_slots > 0:
        base_per_label = remaining_slots // len(large_labels)
        extras = remaining_slots % len(large_labels)

        for i, (label, idxs) in enumerate(large_labels):
            n_clusters = base_per_label + (1 if i < extras else 0)
            n_clusters = min(n_clusters, len(idxs))  # safety check

            X_label = affinity[idxs]
            if hasattr(X_label, "toarray"):
                X_label = X_label.toarray()

            km = KMedoids(n_clusters=n_clusters,
                          random_state=random_state).fit(X_label)
            medoids = [idxs[j] for j in km.medoid_indices_]
            selected_indices.extend(medoids)

    if len(selected_indices) != target_total:
        raise ValueError(
            f"Expected {target_total} medoids but got {len(selected_indices)}.")

    # Preserve order
    ordered_indices = sorted(selected_indices)

    if len(ordered_indices) != target_total:
        raise ValueError(
            "Mismatch after ordering — likely duplicate trimming removed too much.")

    return np.array(ordered_indices)


def convert_least_squares(affinity, Y, clstr, num_clstr, size_embedding=50):
    print('convert least squares:')
    st = time.time()

    print('doing randomized MDS')
    embedding = mds_full_randomized(affinity, size_embedding)
    affinity_transformed = csr_matrix(embedding)
    print(f'MDS time: {time.time() - st}')

    if clstr and affinity_transformed.shape[0] > num_clstr:
        print('clustering samples based on tree-leaf-predictions representation')
        medoid_inds = select_hybrid_balanced_medoids(affinity, Y, num_clstr)
        affinity_transformed = affinity_transformed[medoid_inds]
        print('medoid inds shape:', medoid_inds.shape)
        print('affinity shape:', end=' ')
        print(affinity_transformed.shape)
    else:
        medoid_inds = np.arange(affinity_transformed.shape[0])

    affinity_transformed = csr_matrix(affinity_transformed)
    sparsegram = affinity_transformed @ affinity_transformed.T
    print('sparsegram shape:', end=' ')
    print(sparsegram.shape)

    signal = csr_matrix(sparsegram.reshape(
        (sparsegram.shape[0]**2, 1), order='F'))
    signal = csc_matrix(signal)

    print('signal created.')
    print('signal shape:', signal.shape)

    return signal, sparsegram, medoid_inds


def _score_chunk_return_arrays(chunk_indices, sub_chunk, M, row_sq_norms):
    """
    Compute scores for a chunk and return arrays aligned with chunk_indices.
    Returns (chunk_indices, nums_array, denoms_array, scores_array)
    All arrays use float64 and preserve the same index order as chunk_indices.
    """
    n_local = len(chunk_indices)
    nums = np.zeros(n_local, dtype=np.float64)
    denoms = np.zeros(n_local, dtype=np.float64)
    scores = np.full(n_local, -np.inf, dtype=np.float64)

    for local_i, global_i in enumerate(chunk_indices):
        u = sub_chunk.getrow(local_i).T  # column (n_samples, 1), sparse
        if u.nnz == 0:
            # leave score = -inf
            continue

        # exact same matvec order as serial: tmp = M @ u; then num = u.T @ tmp
        tmp = M.dot(u)

        # get scalar robustly from 1x1 sparse result
        tmpmat = u.T.dot(tmp)
        if getattr(tmpmat, "nnz", 0):
            num = float(tmpmat.data[0])
        else:
            num = 0.0

        denom = float(row_sq_norms[global_i])  # ensure float64

        if denom == 0.0:
            # avoid division by zero
            scores[local_i] = -np.inf
        else:
            scores[local_i] = num / denom

        nums[local_i] = num
        denoms[local_i] = denom

    return chunk_indices, nums, denoms, scores


def matching_pursuit_lazy_parallel(signal, sub_mags, s, n_jobs=None):
    """
    Parallel matching pursuit that is exact-equivalent to the serial implementation.
    - signal: sparse vector (n_samples**2, 1), Fortran-order vec(M)
    - sub_mags: csr_matrix, shape (n_bases, n_samples)  <-- each row is u_i
    - s: number of atoms to select
    - n_jobs: number of threads (defaults to min(cpu_count(), n_bases))

    Returns: indices, coefs (same semantics as serial matching_pursuit_lazy).
    """
    if not isinstance(sub_mags, csr_matrix):
        sub_mags = csr_matrix(sub_mags)

    # ensure float64 to avoid dtype differences
    if sub_mags.dtype != np.float64:
        sub_mags = sub_mags.astype(np.float64)

    n_bases, n_samples = sub_mags.shape

    # reshape signal -> residual matrix M (CSR)
    R_vec = signal.toarray().ravel()
    M = csr_matrix(R_vec.reshape(
        (n_samples, n_samples), order='F'), dtype=np.float64)

    # precompute denom (||u||^2) exactly as serial
    row_sq_norms = np.asarray(sub_mags.power(2).sum(
        axis=1)).reshape(-1).astype(np.float64)

    # chunking strategy (deterministic)
    if n_jobs is None:
        n_jobs = min(cpu_count(), n_bases)
    else:
        n_jobs = min(n_jobs, n_bases)
    all_indices = np.arange(n_bases)
    chunks = [c for c in np.array_split(all_indices, n_jobs) if len(c) > 0]
    sub_chunks = [sub_mags[chunk] for chunk in chunks]

    indices = []
    coefs = []

    # reuse thread pool across iterations for speed
    with ThreadPoolExecutor(max_workers=n_jobs) as exc:
        for _ in range(s):
            # submit scoring jobs for each chunk
            futures = [
                exc.submit(_score_chunk_return_arrays,
                           chunks[i], sub_chunks[i], M, row_sq_norms)
                for i in range(len(chunks))
            ]
            # collect chunk outputs and assemble full arrays
            # create arrays to hold full results in global order
            nums_full = np.zeros(n_bases, dtype=np.float64)
            denoms_full = np.zeros(n_bases, dtype=np.float64)
            scores_full = np.full(n_bases, -np.inf, dtype=np.float64)

            for f in futures:
                chunk_indices, nums, denoms, scores = f.result()
                # chunk_indices is a numpy array of global indices; we place results accordingly
                scores_full[chunk_indices] = scores
                nums_full[chunk_indices] = nums
                denoms_full[chunk_indices] = denoms

            # now pick global best (np.argmax picks first occurrence on ties)
            best_idx = int(np.argmax(scores_full))
            best_score = float(scores_full[best_idx])

            # if best_score is -inf then no candidate left
            if not np.isfinite(best_score) or best_score == -np.inf:
                break

            best_num = float(nums_full[best_idx])
            best_denom = float(denoms_full[best_idx])

            # reconstruct the chosen atom and update residual exactly as serial
            u_chosen = sub_mags.getrow(best_idx).toarray().ravel()
            atom_vec = np.kron(u_chosen, u_chosen)
            atom_norm = best_denom
            maxproj = best_score

            coef = maxproj / atom_norm
            coefs.append(coef)
            indices.append(best_idx)

            # Update residual vector and M for next iteration
            R_vec = R_vec - (maxproj / atom_norm) * atom_vec
            M = csr_matrix(R_vec.reshape(
                (n_samples, n_samples), order='F'), dtype=np.float64)

    return indices, coefs


def diag_impo(mags, coordinates, coefficients):
    """ Reconstruct the diagonal arrray
        from the adaptive haarlike model.

        Parameters:
        mags
        coordinates
        coefficients

        Returns:
        coefs
    """

    assigned = []
    coefs = np.zeros(mags.shape[0])

    for i in range(len(coordinates)):

        coord = coordinates[i]
        coef = coefficients[i]
        if abs(coef) > 1e10:
            print(
                f"Warning: suspiciously large coefficient {coef} at coord {coord}")

        if not coord in assigned:

            coefs[coord] = coef

    return coefs


def adaptive(haar_basis, biom_table, label, tree, meta, s, cluster_affinity, num_clstr):
    """ shl: sparse haar like coordinates
        biom_table_data: biom_table, 
        label: str, column in meta to use 
        biom_table: skbio.TreeNode
        meta: pd.dataframe
        s=5: Number of important nodes to find

        returns dic, rfgram, coordinates, coefs, Y, dic, new_diag
    """

    # Control Vars. Maybe later add to func args?
    print('running with cluster_affinity=', cluster_affinity, sep='')

    X, Y, mags, dic = preprocess(label, biom_table, haar_basis, meta, tree)
    print('preprocessing done.')

    clf = RandomForestClassifier(n_estimators=500,
                                 bootstrap=True,
                                 min_samples_leaf=1,
                                 n_jobs=-1)
    clf.fit(X, Y)
    print('RF training done.')

    rfaffinity = proximity_matrix(clf, X)
    print('affinity generated.')

    signal, rfgram, medoid_indices = convert_least_squares(
        rfaffinity,
        Y,
        cluster_affinity, num_clstr)
    Y = pd.Series([Y.iloc[i] for i in range(len(Y)) if i in medoid_indices])

    mags = mags[:, medoid_indices]

    st = time.time()
    coordinates, coefs = matching_pursuit_lazy_parallel(signal, mags, s)
    print(coordinates)
    print(coefs)
    print(f'signal estimated with Haar coefs in {time.time() - st} seconds.')

    new_diag = diag_impo(mags, coordinates, coefs)
    print('diag created.')

    return dic, rfgram, coordinates, coefs, Y, dic, new_diag, mags


def reconstruct_coord(coefs, coordinates, mags, s):
    """ This function is for reconstructing and obtaining a vector
        used in plotting the biplots. The resulting coord is of shape
        n_internal_nodes_found (default 5) x n_samples (?) """

    coord = np.sqrt(coefs[0]) * mags[coordinates[0], :].todense()
    for i in range(1, s):
        nodei = np.sqrt(coefs[i]) * mags[coordinates[i], :].todense()
        coord = np.vstack((coord, nodei))
    return coord


def reconstruct(coordinates, mags, s, coefs):
    """ This function reconstructs the matrix using weights of the
        top coordinates and the modmags matrix, which is the projection
        onto the shl matrix. 

        coordinates: 
        mags: csr_matrix
        s: int - number of coordinates to find (default 5)
        coefs:

        returns 
        outer: numpy.matrix of shape s x n_samples
    """
    d, n = mags.get_shape()  # d=n internal nodes; n=n samples
    outer = np.zeros((n, n))
    for i in range(s):
        temp = mags[coordinates[i], :].todense()
        out = np.outer(temp, temp)
        outer = outer + coefs[i]*out
    return outer


# PLOTTERS

def new_biplot3d(s, coefs, coordinates, mags, y,
                 labeltype, dic, k, n, save, path):

    Z = np.transpose(reconstruct_coord(coefs, coordinates, mags, s))
    pca = PCA()
    x_new = pca.fit_transform(np.asarray(Z))
    score = x_new[:, 0:3]
    coeff = np.transpose(pca.components_[0:3, :])
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    scalex = 1.0
    scaley = 1.0
    scalez = 1.0
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')

    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels = [inv_map[i] for i in y]

    if labeltype == 'classification':

        if len(y) == 2:
            colors = cm.tab10(y/(2*max(y)))
        else:
            colors = cm.tab10(y/max(y))

        for xs, ys, zs, c, label in zip(xs, ys, zs, colors, labels):
            ax.scatter(xs*scalex, ys*scaley, zs*scalez,
                       facecolors="None", edgecolors=c, label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 7})

    elif labeltype == 'regression':
        c = cm.viridis(y/max(y))
        p = ax.scatter(xs*scalex, ys*scaley, zs*scalez, c=y, cmap='viridis')
        plt.colorbar(p, pad=0.15)

    for i in range(n):
        ax.quiver(
            0, 0, 0,  # starting point of vector
            1.15*coeff[i, 0], 1.15*coeff[i, 1], 1.15 *
            coeff[i, 2],  # vector directi
            color='black', alpha=.7, lw=2
        )
        ax.text(
            coeff[i, 0] * 1.25, coeff[i, 1] * 1.25, coeff[i, 2] * 1.25,
            coordinates[i],
            color='black', ha='center', va='center'
        )

    rat0 = np.around(pca.explained_variance_ratio_[0] * 100, 2)
    rat1 = np.around(pca.explained_variance_ratio_[1] * 100, 2)
    rat2 = np.around(pca.explained_variance_ratio_[2] * 100, 2)
    xlab = "PC1" + ' ' + str(rat0) + '%'
    ylab = "PC2" + ' ' + str(rat1) + '%'
    zlab = "PC3" + ' ' + str(rat2) + '%'
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    ax.set_zlabel(zlab)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])

    plt.title('PCA Biplot')
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    ax.dist = 12
    fig.tight_layout(pad=2)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def new_biplot3dnormalized(s, coefs, coordinates, mags, y,
                           labeltype, dic, k, n, save, path):

    Z = np.transpose(reconstruct_coord(coefs, coordinates, mags, s))
    scaler = StandardScaler()
    scaler.fit(np.asarray(Z))
    Z = scaler.transform(np.asarray(Z))
    pca = PCA()
    x_new = pca.fit_transform(Z)
    score = x_new[:, 0:3]
    coeff = np.transpose(pca.components_[0:3, :])
    xs = score[:, 0]
    ys = score[:, 1]
    zs = score[:, 2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scalez = 1.0 / (zs.max() - ys.min())
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')

    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels = [inv_map[i] for i in y]
    if labeltype == 'classification':
        if len(y) == 2:
            colors = cm.tab10(y / (2*max(y)))
        else:
            colors = cm.tab10(y / max(y))
        for xs, ys, zs, c, label in zip(xs, ys, zs, colors, labels):

            ax.scatter(xs * scalex, ys * scaley, zs * scalez,
                       facecolors="None",
                       edgecolors=c, label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 7})

    elif labeltype == 'regression':
        c = plt.cm.viridis(y / max(y))
        p = ax.scatter(xs*scalex, ys*scaley, zs*scalez,
                       c=y, cmap='viridis')
        plt.colorbar(p, pad=0.15)

    for i in range(n):
        ax.quiver(
            0, 0, 0,  # starting point of vector
            1.15*coeff[i, 0], 1.15*coeff[i, 1], 1.15 *
            coeff[i, 2],  # vector directi
            color='black', alpha=.7, lw=2
        )
        ax.text(coeff[i, 0] * 1.25, coeff[i, 1] * 1.25, coeff[i, 2] * 1.25,
                coordinates[i], color='black', ha='center', va='center')

    rat0 = np.around(pca.explained_variance_ratio_[0]*100, 2)
    rat1 = np.around(pca.explained_variance_ratio_[1]*100, 2)
    rat2 = np.around(pca.explained_variance_ratio_[2]*100, 2)
    plt.xlabel("PC1" + ' ' + str(rat0) + '%')
    plt.ylabel("PC2" + ' ' + str(rat1) + '%')
    ax.set_zlabel("PC3" + ' ' + str(rat2) + '%')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 6, box.height])
    plt.title('PCA Biplot Normalized')
    ax.grid(False)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_facecolor('white')
    ax.dist = 12
    fig.tight_layout(pad=2)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def _get_correlation(sorted_rfgram, sorted_reconstructed):

    matrix1 = np.array(sorted_rfgram.todense())
    matrix2 = sorted_reconstructed
    corr = np.corrcoef(matrix1.flatten(), matrix2.flatten())[0, 1]
    return corr


def rfgram_plot(rfgram, coordinates, modmags, s, coefs, Y, dic,
                save, path):
    """ rfgram is the representation of the dataset using a trained rf 
        model, which works by training an RF on all data poins then forming
        an rf gram matrix. this matrix represents how similar two points are 
        to eachother by seeing how similarly they are grouped by the decision 
        trees in the random forest. the reconstructed plot takes our learned
        haar-like weight vector, which only uses 5 internal nodes to re-
        represent the data set. if the two plots look similar, this means
        that our learned weight vector w does a good job of recapitulating the
        original random forest in a far smaller feature space."""

    groups = list(Y)
    inv_dic = {v: k for k, v in dic.items()}

    reconstructed = reconstruct(coordinates, modmags, s, coefs)
    sorted_indices = np.argsort(groups)
    sorted_groups = np.array(groups)[sorted_indices]
    group_names = [inv_dic[x] for x in sorted_groups]  # Name for each group

    sorted_rfgram = rfgram[:, sorted_indices][sorted_indices, :]

    sorted_reconstructed = reconstructed[:, sorted_indices][sorted_indices, :]
    # reconstructed = reconstruct(coordinates, modmags, s, coefs)

    # Create figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Heatmap 1
    axes[0].imshow(sorted_rfgram.todense(), vmin=0, vmax=.3, cmap='binary')
    axes[0].xaxis.set_visible(False)

    # Heatmap 2
    axes[1].imshow(sorted_reconstructed, vmin=0, vmax=.3, cmap='binary')
    axes[1].xaxis.set_visible(False)

    # Create new axes below each heatmap for the group bars
    # Adjust the position and size to fit the bars
    group_bar_height = 0.05  # Height of the group bars
    label_padding = 0.1  # Space between the bar and the label
    ax_group_bar1 = fig.add_axes([axes[0].get_position().x0, axes[0].get_position(
    ).y0, axes[0].get_position().width, group_bar_height], frameon=False)
    ax_group_bar2 = fig.add_axes([axes[1].get_position().x0, axes[1].get_position(
    ).y0, axes[1].get_position().width, group_bar_height], frameon=False)

    # Plot the group bars
    ax_group_bar1.imshow([sorted_groups], aspect='auto', cmap='Set1')
    ax_group_bar2.imshow([sorted_groups], aspect='auto', cmap='Set1')

    # Hide y-axis and ticks for the group bars
    ax_group_bar1.set_yticks([])
    ax_group_bar1.set_xticks([])
    ax_group_bar2.set_yticks([])
    ax_group_bar2.set_xticks([])

    # Determine where each group starts and ends
    unique_groups, start_idx = np.unique(sorted_groups, return_index=True)
    end_idx = np.append(start_idx[1:], len(sorted_groups))

    # Add vertical group labels centered below each group segment
    for i, group in enumerate(unique_groups):

        # Center position for the group label
        center = (start_idx[i] + end_idx[i] - 1) / 2

        # Adjust the vertical position of the labels to avoid overlap
        label_y = 1
        ax_group_bar1.text(
            center, label_y, group_names[start_idx[i]], ha='center', va='top', rotation=90, fontsize=10)
        ax_group_bar2.text(
            center, label_y, group_names[start_idx[i]], ha='center', va='top', rotation=90, fontsize=10)

    # Adjust layout to fit the labels
    # Increase bottom margin to accommodate labels
    plt.subplots_adjust(bottom=0.2)

    corr_coeff = _get_correlation(sorted_rfgram, sorted_reconstructed)
    axes[1].text(1.1, 0.5, f'Pearson Correlation: {corr_coeff:.2f}',
                 fontsize=12, va='center', ha='left',
                 transform=axes[1].transAxes)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def boxplot_plotter(mags, y, indices, dic,
                    xlabels, save, path):

    fig, ax = plt.subplots(len(indices), sharex=True,
                           figsize=(14, 2.5 * len(indices)))
    boxlabels = list(dic.keys())
    inv_map = {v: k for k, v in dic.items()}
    datalabels = [inv_map[i] for i in y]
    for k in range(len(indices)):
        alldata = []
        for i in range(len(boxlabels)):
            dataindices = [
                j
                for j, x in enumerate(datalabels)
                if x == boxlabels[i]]
            alldata.append(
                np.asarray(mags[indices[k], dataindices].todense()).squeeze())

        temp = ax[k].boxplot(alldata,
                             vert=True,  # vertical box alignment
                             patch_artist=True,  # fill with color
                             labels=xlabels)  # will be used to label x-ticks
        ax[k].set_title('Haar-like Node' + ' ' + str(indices[k]))
        plt.setp(temp['whiskers'], color='black')
        plt.setp(temp['medians'], color='black')
        plt.setp(temp['fliers'], color='black', marker='')

        numerator = np.array(list(dic.values()))
        if len(y) == 2:
            denominator = (2 * max(np.array(list(dic.values()))))
            colors = cm.tab10(numerator / denominator)
        else:
            denominator = (max(np.array(list(dic.values()))))
            colors = cm.tab10(numerator / denominator)

        for patch, color in zip(temp['boxes'], colors):
            patch.set_facecolor(color)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def make_plots(adhld_results, modmags, path, s, k, n):

    # unpack reslts
    dic, rfgram, coordinates, coefs, Y, dic, _, _ = adhld_results

    # define paths to save all 4 plots
    path1 = os.path.join(path, 'rfgram.svg')
    path2 = os.path.join(path, 'boxplot.svg')
    path3 = os.path.join(path, 'biplot.svg')
    path4 = os.path.join(path, 'biplotn.svg')

    # RF GRAM MATRIX
    rfgram_plot(rfgram, coordinates, modmags, s, coefs, Y, dic,
                save=True, path=path1)

    # BOXPLOTS OF NODES
    boxplot_plotter(modmags, Y.values, coordinates[0:s], dic,
                    dic.keys(), save=True, path=path2)

    # BIPLOT
    new_biplot3d(s, coefs, coordinates, modmags, Y,
                 'classification', dic, k, n,
                 save=True, path=path3)

    new_biplot3dnormalized(s, coefs, coordinates, modmags, Y,
                           'classification', dic, k, n,
                           save=True, path=path4)
