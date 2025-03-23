import numpy as np
import os
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from itertools import chain
from scipy.sparse import linalg, csr_matrix, csc_matrix

from sklearn.manifold import MDS
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

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
    # row_sum = X.to_numpy().sum(axis=1)
    # print("\nRow sums (sum along axis=1):")
    # print(row_sum)
    # print("\nCheck for NaN or infinite values in row sums:")
    # print(np.count_nonzero(~np.isnan(row_sum)))  # Count NaN values
    # print((row_sum == float('inf')).sum())  # Count infinite values
    print("Y shape:", Y.shape)
    print(type(Y))
    print("abundance_vector shape:", abundance_vector.shape)
    print("mags shape:", mags.shape)
    print("haar_basis shape:", haar_basis.shape)
    # import json
    # import scipy.sparse as sp
    # X.to_csv("X_check.csv")
    # Y.to_csv("Y_check.csv")
    # np.save('abundance_vector_check.npy', abundance_vector)
    # sp.save_npz("mags_check.npz", mags)
    # sp.save_npz('shl_check.npz', csr_matrix(haar_basis))
    # with open("dic_check.txt", "w") as f:
    #     json.dump(dic, f)

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
    print('proximity matrix')
    if lgbm:
        terminals = clf.predict(X, pred_leaf=True, n_jobs=6)
    else:
        terminals = clf.apply(X)

    print('tree leaves preds. shape', terminals.shape)

    nsamples, nTrees = terminals.shape

    # 1 - orig
    prox = np.zeros((nsamples, nsamples))

    for i in range(nTrees):
        a = terminals[:, i]
        prox += 1*np.equal.outer(a, a)

    prox = prox / nTrees
    print(prox)
    print(prox.shape)



    # 2
    # prox = np.sum(np.equal.outer(terminals, terminals), axis=2) / nTrees


    # 3
    # from joblib import Parallel, delayed
    # def compute_prox(i):
    #     """Computes contribution of tree i to proximity matrix."""
    #     a = terminals[:, i]
    #     return np.equal.outer(a, a).astype(int)
    # # Run in parallel across multiple threads
    # prox = sum(Parallel(n_jobs=-1)(delayed(compute_prox)(i) for i in range(nTrees))) / nTrees
    # print(prox)
    # print()

    # # 4
    # from joblib import Parallel, delayed
    # def compute_sparse_row(i, terminals):
    #     """Computes a single row of the sparse dissimilarity matrix."""
    #     diff = np.sum(terminals[i] == terminals, axis=1) / terminals.shape[1]
    #     return diff  # Return only the computed row
    # # Get total number of samples
    # n_samples = terminals.shape[0]
    # # Compute dissimilarity matrix in parallel (row-wise)
    # results = Parallel(n_jobs=-1)(
    #     delayed(compute_sparse_row)(i, terminals) for i in range(n_samples)
    # )

    # print(np.array(np.vstack(results)))
    # print()
    # # Convert results into a sparse CSR matrix
    # sparse_dissimilarity = csr_matrix(np.vstack(results))

    # from sklearn.decomposition import TruncatedSVD

    # # Suppose X is large and sparse (e.g., shape [n_samples, n_features])
    # svd = TruncatedSVD(n_components=600, random_state=42)

    # sparse_dissimilarity_reduced = svd.fit_transform(sparse_dissimilarity)

    # print(sparse_dissimilarity_reduced.shape)
    # return sparse_dissimilarity_reduced
    return prox


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


def convert_least_squares(affinity, mags, size_embedding=50):
    """ Parameters:
        affinity: rf affinity matrix
        mags: same mags from before

        Returns:
        signal: a n_otus x n_samples**2 
                 vector of all the sample mapping to the forest
        A: a csc_matrix which is something??? 
        sparsegram : transformed signals? some sort of mapping
    """
    print('convert least squares')
    embedding = MDS(n_components=size_embedding,
                    dissimilarity='precomputed', n_jobs=-1)
    # embedding = MDS(n_components=size_embedding, dissimilarity='euclidean', n_jobs=-1)
    affinity_transformed = csr_matrix(embedding.fit_transform(affinity))
    print(affinity_transformed.shape)
    print(affinity_transformed)


    # from umap import UMAP
    # umap_model = UMAP(n_components=size_embedding)
    # affinity_transformed = csr_matrix(umap_model.fit_transform(affinity))


    sparsegram = affinity_transformed @ affinity_transformed.T
    print(sparsegram.shape)

    nsamples = mags.shape[1]
    signal = csr_matrix.reshape(sparsegram,
                                ((nsamples**2, 1)),
                                order='F')
    basis_dictionary = spouter(mags, mags).T
    basis_dictionary_sparse = csc_matrix(basis_dictionary)

    return signal, basis_dictionary_sparse, sparsegram


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

    # LightGBM parameters for a FAST model
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


def adaptive(haar_basis, biom_table, label, tree, meta, s, lgbm=False):
    """ shl: sparse haar like coordinates
        biom_table_data: biom_table, 
        label: str, column in meta to use 
        biom_table: skbio.TreeNode
        meta: pd.dataframe
        s=5: Number of important nodes to find

        returns dic, rfgram, coordinates, coefs, Y, dic, new_diag
    """

    X, Y, mags, dic = preprocess(label, biom_table, haar_basis, meta, tree)
    print('preprocessing done.')

    # DEBUG
    if lgbm:
        clf = train_LGBM(X, Y)
    else:
        clf = RandomForestClassifier(n_estimators=500,
                                     bootstrap=True,
                                     min_samples_leaf=1,
                                     n_jobs=-1)
        clf.fit(X, Y)
    print('training done.')

    rfaffinity = proximity_matrix(clf, X, lgbm)
    print('affinity generated.')

    signal, dictionary, rfgram = convert_least_squares(rfaffinity, mags)
    signal = csc_matrix(signal)
    print('signal created.')

    coordinates, coefs = matching_pursuit(signal, dictionary, s)
    print('signal estimated with Haar coefs.')
    new_diag = diag_impo(mags, coordinates, coefs)

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
