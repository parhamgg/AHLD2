import numpy as np
import pandas as pd
import os
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from itertools import chain
from scipy.sparse import linalg, csr_matrix, csc_matrix, coo_matrix, vstack

from sklearn.manifold import MDS
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn_extra.cluster import KMedoids

from skbio.stats.ordination import pcoa


import lightgbm as lgb

__all__ = [
    'calc_haar_mags',
    'get_otu_abundances',
    'preprocess',
    'proximity_matrix',
    'spouter',
    'convert_least_squares',
    'matching_pursuit',
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
# ADAPTIVE FNS -> What is this comment?


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


def proximity_matrix(clf, X, lgbm):
    """ Generate random forest affinity matrix
    clf: a classifier (RandomForestClassifier)
    X : data matrix with dimensions n by m 
    lgbm: Model to use (true=LGBM/False)
    """

    # terminals = index of the leaf in each decision tree that a sample
    # (or samples) would land in. (nsamples x n estimators)
    print('proximity matrix:')
    if lgbm:
        terminals = clf.predict(X, pred_leaf=True, n_jobs=6)
    else:
        terminals = clf.apply(X)
    print('tree leaves preds. shape', terminals.shape)

    nsamples, nTrees = terminals.shape
    prox = np.zeros((nsamples, nsamples))
    for i in range(nTrees):
        a = terminals[:, i]
        prox += 1*np.equal.outer(a, a)
    prox = 1 - (prox / nTrees)
    return prox


def spouter_partitioned(A, B, num_partitions=5000):
    """ 
    Computes the sparse outer product of two sparse matrices A and B in partitions.
    Optimized to avoid memory overload using float32 and COO construction.
    """

    # Get matrix shapes
    N, L = A.shape
    _, K = B.shape

    # Partition rows to reduce memory pressure
    row_splits = np.array_split(np.arange(N), num_partitions)

    # To collect all row-wise sparse outer products
    sparse_partitions = []

    for row_indices in row_splits:
        # Get submatrices and ensure float32 for lower memory usage
        A_partition = A[row_indices].astype(np.float32)
        B_partition = B[row_indices].astype(np.float32)

        sparse_outer_products = []

        for i in range(A_partition.shape[0]):
            a = A_partition[i].data
            a_idx = A_partition[i].indices

            b = B_partition[i].data
            b_idx = B_partition[i].indices

            row_inds = []
            col_inds = []
            data = []

            # Only compute for non-zero pairs
            for ai, av in zip(a_idx, a):
                for bi, bv in zip(b_idx, b):
                    row_inds.append(0)  # single-row matrix
                    col_inds.append(ai * K + bi)
                    data.append(av * bv)

            if data:
                row_sparse = coo_matrix(
                    (data, (row_inds, col_inds)), shape=(1, L * K), dtype=np.float32
                ).tocsr()
            else:
                row_sparse = csr_matrix((1, L * K), dtype=np.float32)

            sparse_outer_products.append(row_sparse)

        # Stack all sparse rows in this partition
        partition_result = vstack(sparse_outer_products)
        sparse_partitions.append(partition_result)

    # Combine all partitions
    result = vstack(sparse_partitions)

    print(
        f'Type of result: {type(result)}, dtype: {result.dtype}, shape: {result.shape}')

    return result


def spouter(A, B):
    """ Quickly compute sparse outer product
    of two matrices. 

    SOURCE:   https://stackoverflow.com/
        questions/57099722/row-wise-outer-product-on-sparse-matrices """

    N, L = A.shape
    N, K = B.shape

    drows = zip(*(np.split(x.data, x.indptr[1:-1]) for x in (A, B)))
    data = [np.outer(a, b).ravel() for a, b in drows]
    irows = zip(*(np.split(x.indices, x.indptr[1:-1]) for x in (A, B)))
    indices = [
        np.ravel_multi_index(np.ix_(a, b), (L, K)).ravel()
        for a, b in irows
    ]
    indptr = np.fromiter(chain((0, ), map(len, indices)), int).cumsum()

    return csr_matrix((np.concatenate(data), np.concatenate(indices), indptr),
                      (N, L * K))


def landmark_MDS(D, lands, dim):
    """based on https://github.com/danilomotta/LMDSParameters:
        D (ndarray): Full distance matrix of shape (n_samples, n_samples)
        lands (list or array): Indices of landmark points
        dim (int): Number of dimensions to project onto

    Returns:
        ndarray: Low-dimensional embedding of shape (n_samples, dim)
    """
    Dl = D[:, lands]  # shape: (n_samples, n_landmarks)
    n, k = Dl.shape

    # Double centering for non-square distance matrix
    Dl2 = Dl ** 2
    row_mean = np.mean(Dl2, axis=1, keepdims=True)     # shape (n, 1)
    col_mean = np.mean(Dl2, axis=0, keepdims=True)     # shape (1, k)
    total_mean = np.mean(Dl2)                          # scalar

    B = -0.5 * (Dl2 - row_mean - col_mean + total_mean)  # shape (n, k)

    # SVD decomposition
    U, S, _ = np.linalg.svd(B, full_matrices=False)

    # Keep only significantly positive singular values
    pos = S > 1e-10  # Filter out numerical noise
    num_pos = np.count_nonzero(pos)

    if num_pos == 0:
        print("Error: No positive singular values found.")
        return np.empty((n, 0))  # Return empty array with correct shape

    # Adjust `dim` to available singular values
    dim = min(dim, num_pos)
    print(f"Using {dim} dimensions out of {num_pos} available.")

    # Use top-dim components
    U = U[:, pos][:, :dim]  # Filter U for valid singular values
    S = S[pos][:dim]        # Keep top singular values

    # Compute embedding
    X = U * np.sqrt(S)  # shape: (n, dim)

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


def convert_least_squares(
    affinity,
    mags,
    Y,
    lmds,
    num_lmds,
    clstr,
    num_clstr,
    num_sparse_partitions,
    size_embedding=50,
    filter_by_taxonomy=False,
    taxonomy_map=None,
):
    """ Parameters:
        affinity: rf affinity matrix
        mags: internal node matrix (n_nodes x n_samples)
        Y: optional labels for clustering
        lmds: boolean, whether to use landmark MDS
        num_lmds: number of landmark points for LMDS
        clstr: boolean, whether to cluster and reduce samples
        num_clstr: number of clusters (medoids)
        num_sparse_partitions: for basis dictionary construction
        size_embedding: size of projected embedding
        filter_by_taxonomy: whether to exclude poorly resolved taxonomies
        taxonomy_map: dict mapping internal node names to taxonomy strings

        Returns:
        signal: vectorized sparsegram (n_samples^2 x 1)
        A: basis dictionary as csc_matrix
        sparsegram: transformed affinity inner-product matrix
        medoid_inds: indices of selected medoids (if clustering)
    """
    print('convert least squares:')

    # Step 1: Embedding
    if not lmds:
        ordination_results = pcoa(
            affinity, method='fsvd', number_of_dimensions=size_embedding)
        affinity_transformed = csr_matrix(ordination_results.samples.values)
    else:
        print('doing landmark MDS with', end=' ')
        num_landmarks = min(affinity.shape[0], num_lmds)
        print('landmarks', end=' ')
        np.random.seed(0)
        lands = np.random.choice(
            affinity.shape[0], num_landmarks, replace=False)
        print(lands)
        embedding = landmark_MDS(affinity, lands, size_embedding)
        affinity_transformed = csr_matrix(embedding)

    # Step 2: Optional clustering
    if clstr and affinity_transformed.shape[0] > num_clstr:
        print('clustering samples based on tree-leaf-predictions representation')
        medoid_inds = select_hybrid_balanced_medoids(affinity, Y, num_clstr)
        affinity_transformed = affinity_transformed[medoid_inds]
        print('medoid inds shape:', medoid_inds.shape)
        print('affinity shape:', affinity_transformed.shape)
    else:
        medoid_inds = np.arange(affinity_transformed.shape[0])

    affinity_transformed = csr_matrix(affinity_transformed)
    sparsegram = affinity_transformed @ affinity_transformed.T
    print('sparsegram shape:', sparsegram.shape)

    signal = csr_matrix(sparsegram.reshape(
        (sparsegram.shape[0]**2, 1), order='F'))
    print('signal shape:', signal.shape)

    # Step 3: Prepare mags
    sub_mags = mags[:, medoid_inds]

    # Step 4: Filter based on taxonomy
    if filter_by_taxonomy:
        if taxonomy_map is None:
            raise ValueError(
                "taxonomy_map must be provided when filter_by_taxonomy=True")

        def has_sufficient_resolution(tax_str):
            if not tax_str or tax_str == 'not found':
                return False

            levels = [level.strip() for level in tax_str.split(';')]
            
            # 1. Reject if any level ABOVE class is empty (d__, p__, c__)
            required_ranks = ['d__', 'p__', 'c__']
            for level, required_prefix in zip(levels, required_ranks):
                if not level.startswith(required_prefix):
                    return False
                if level == required_prefix:
                    return False
            
            # 2. Require at least class-level has a value (c__Something)
            return any(level.startswith("c__") and level != "c__" for level in levels)

        keep_inds = [
            i for i, node_name in enumerate(taxonomy_map)
            if has_sufficient_resolution(taxonomy_map[node_name])
        ]
        print(
            f"Filtering {mags.shape[0]} internal nodes to {len(keep_inds)} based on taxonomy resolution")

        mags = mags[keep_inds, :]
        sub_mags = sub_mags[keep_inds, :]

    # Step 5: Sparse basis dictionary
    basis_dictionary_sparse = csc_matrix(
        spouter_partitioned(sub_mags, sub_mags, num_partitions=num_sparse_partitions).T)

    print('basis dictionary shape', basis_dictionary_sparse.shape,
          basis_dictionary_sparse[0].shape)

    return signal, basis_dictionary_sparse, sparsegram, medoid_inds


def matching_pursuit(signal, dictionary, s):

    dictionarynorm = normalize(dictionary, norm='l2', axis=0)
    coefs = []
    indices = []
    R = signal

    for i in range(s):
        innerprod = dictionarynorm.T@R

        index = np.argmax((innerprod))
        indices.append(index)

        maxproj = innerprod[index].todense().item()

        coefs.append(maxproj/linalg.norm(dictionary[:, index]))
        R = R - maxproj*dictionarynorm[:, index]

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

        if not coord in assigned:

            coefs[coord] = coef

    return coefs


def train_LGBM(X, Y):
    # Convert Y to 1D if it's a single column DataFrame
    Y_1d = Y.squeeze()
    unique_labels = sorted(Y_1d.unique())
    num_unique = len(unique_labels)

    # Shift labels so smallest is 0 (LightGBM multiclass requires 0-based)
    shift = unique_labels[0]
    Y_1d_zero_based = Y_1d - shift

    # Choose objective
    if num_unique == 2:
        objective = 'binary'
        metric = 'binary_logloss'
    else:
        objective = 'multiclass'
        metric = 'multi_logloss'

    # Prepare LightGBM Datasets
    train_data = lgb.Dataset(X, label=Y_1d_zero_based)

    # LightGBM parameters for a fast model
    params = {
        'objective': objective,
        'metric': metric,
        'num_class': num_unique,
        'learning_rate': 0.16819239188201524,
        'num_leaves': 6,
        'max_depth': 38,
        'min_data_in_leaf': 1,  # equivalent to min_samples_leaf
        'bagging_fraction': 0.8945334948696256,
        'bagging_freq': 1,
        'feature_fraction': 0.7271535699005337,
        'verbose': -1
    }

    num_boost_round = 247

    start_lgb = time.time()
    bst = lgb.train(
        params=params,
        train_set=train_data,
        num_boost_round=num_boost_round,
    )
    end_lgb = time.time()
    lgb_time = end_lgb - start_lgb
    print(f"LightGBM training time: {lgb_time:.2f} s")

    return bst


def adaptive(
    haar_basis, biom_table, label, tree, meta, s, lgbm,
    use_landmarkMDS, num_lmds, cluster_affinity, num_clstr,
    num_sparse_partitions, taxonomy=None, filter_by_taxonomy=False  # NEW ARGS
):
    print('running with lgbm=', lgbm, ' cluster_affinity=',
          cluster_affinity, ' lmds=', use_landmarkMDS, sep='')

    X, Y, mags, dic = preprocess(label, biom_table, haar_basis, meta, tree)
    print('preprocessing done.')

    if lgbm:
        clf = train_LGBM(X, Y)
    else:
        clf = RandomForestClassifier(
            n_estimators=500, bootstrap=True, min_samples_leaf=1, n_jobs=-1)
        clf.fit(X, Y)
    print('training done.')

    rfaffinity = proximity_matrix(clf, X, lgbm)
    print('affinity generated.')

    # Internal node names for alignment
    nontips = [n for n in tree.postorder() if not n.is_tip()]
    node_names = [str(n.name) for n in nontips]

    taxonomy_map = None
    if filter_by_taxonomy:
        if taxonomy is None:
            raise ValueError(
                "Taxonomy must be provided if filter_by_taxonomy=True")
        tree, taxonomy_map = annotate_tree(tree, taxonomy)
        mags, node_names, taxonomy_map = align_taxonomy_to_mags(
            mags, node_names, taxonomy_map)

    signal, dictionary, rfgram, medoid_indices = convert_least_squares(
        rfaffinity, mags, Y, use_landmarkMDS, num_lmds, cluster_affinity,
        num_clstr, num_sparse_partitions,
        filter_by_taxonomy=filter_by_taxonomy,
        taxonomy_map=taxonomy_map
    )

    Y = pd.Series([Y.iloc[i] for i in range(len(Y)) if i in medoid_indices])
    mags = mags[:, medoid_indices]

    signal = csc_matrix(signal)
    print('signal created.')

    coordinates, coefs = matching_pursuit(signal, dictionary, s)
    print('signal estimated with Haar coefs.')
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
