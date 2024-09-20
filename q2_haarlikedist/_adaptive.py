import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from itertools import chain
from scipy.sparse import linalg, csr_matrix, csc_matrix

from sklearn.manifold import MDS
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

__all__ = [
    'preprocess',
    'proximity_matrix',
    'spouter',
    'convert_least_squares',
    'matching_pursuit',
    'diag_impo',
    '_dimensions',
    '_validate',
    'adaptive',
    'reconstruct_coord',
    'reconstruct',
    'new_biplot3d',
    'new_biplot3dnormalized',
    'rfgram_plot',
    'boxplot_plotter',
    'make_plots',
    '_get_correlation'
]

# ADAPTIVE FNS

def preprocess(label, biomtab, shl, meta):
    """ For adaptive hld. Converts to necesseary types for 
        Evan's HLD code to work.
    
        label: column name in metadata,
        biomtab: biom.Table,
        shl: result from sparsification,
        meta: metadata with `label` not as index,
        returns:  X, Y, mags, dic
        """

    dic = {x:i for i, x in enumerate(list(set(meta[label])), 1)}
    data = [dic[x] for x in meta[label]]
    Y = pd.Series(name='label', index=meta.index, data=data)
    X = biomtab.to_dataframe().T
    table = csr_matrix(biomtab.matrix_data)
    mags = shl@table 
    X = X.div(X.sum(axis=1), axis=0) # I don't think this is necessary
    X = np.array(X)
    return X, Y, mags, dic


def proximity_matrix(clf, X):     
    """ Generate random forest affinity matrix
    clf: a classifier (RandomForestClassifier)
    X : data matrix with dimensions n by m """
    
    # terminals = index of the leaf in each decision tree that a sample 
    # (or samples) would land in. (nsamples x n estimators)

    terminals = clf.apply(X)
    nsamples, nTrees = terminals.shape
    prox = np.zeros((nsamples, nsamples))

    for i in range(nTrees):
        a = terminals[:, i]
        prox += 1*np.equal.outer(a, a)

    prox = prox / nTrees

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


def convert_least_squares(affinity, mags):

    """ Parameters:
        affinity: rf affinity matrix
        mags: same mags from before
        
        Returns:
        signal: a n_otus x n_samples**2 
                 vector of all the sample mapping to the forest
        A: a csc_matrix which is something??? 
        sparsegram : transformed signals? some sort of mapping
    """

    embedding = MDS(n_components=50,dissimilarity='precomputed')
    nsamples = mags.shape[1]

    # not storing 1-affinity as a matrix for size reasons 
    X_transformed = csr_matrix(embedding.fit_transform(1-affinity))
    sparsegram = X_transformed @ X_transformed.T

    signal = csr_matrix.reshape(sparsegram,
                                ((nsamples**2,1)),
                                order='F')
    C = spouter(mags, mags)
    A = csc_matrix(C.T)

    return signal, A, sparsegram


def matching_pursuit(signal, dictionary, s):

    dictionarynorm = normalize(dictionary, norm='l2', axis=0)
    coefs = []
    indices = []
    importances = []
    R = signal
    
    for i in range(s):
        print(i)
        innerprod = dictionarynorm.T@R

        index = np.argmax((innerprod))
        indices.append(index)

        maxproj = innerprod[index].todense().item()
        importances.append(maxproj)

        coefs.append(maxproj/linalg.norm(dictionary[:,index]))
        R = R - maxproj*dictionarynorm[:,index]

    return indices, coefs, importances, R


def diag_impo(mags, coordinates, importances, coefficients):
    """ Reconstruct the diagonal arrray
        from the adaptive haarlike model.
         
        Parameters:
        mags
        coordinates
        importances
        coefficients

        Returns:
        coefs:
        impos:
    """
    
    assigned = []
    coefs = np.zeros(mags.shape[0])
    impos = np.zeros(mags.shape[0])

    for i in range(len(coordinates)):

        coord = coordinates[i]
        impo = importances[i]
        coef = coefficients[i]

        if not coord in assigned:

            coefs[coord] = coef
            impos[coord] = impo

    return coefs, impos


def _dimensions(shl, table):
    
    if shl.shape[0] != table.shape[0]:
        m = f'shl ({shl.shape}) and table ({table.shape})' \
            f'shape mismatch in adaptive'
        raise ValueError(m)


def _validate(metadata, label, biomtab):  # metadata is of type Metadata

    meta = metadata.to_dataframe()

    if label not in metadata.columns:
        m = f'Label column id: {label} not found in metadata columns.'
        raise ValueError(m)

    ids = list(biomtab.ids(axis='sample'))
    if not all(z in meta.index for z in ids):
        m = 'Table indeces are not a subset of metadata.'
        raise KeyError(m)

    meta = meta.loc[ids]

    return meta


def adaptive(shl, table, label, biom_table, meta, s=5):
    """ shl
        table: biom_table's data, 
        label: str, column in meta to use 
        biom_table: biom.Table
        meta: pd.dataframe
        s=5

        returns signal, dictionary, rfrgam, mpresults, coordinates,
        coefs, importances, R, new_diag, new_impo, X, Y, dic
    """

    _dimensions(shl, table)

    mags = shl@table

    X, Y, mags, dic = preprocess(label, biom_table, shl, meta)
    clf = RandomForestClassifier(n_estimators=500,
                                 bootstrap=True,
                                 min_samples_leaf=1)
    clf.fit(X, Y)

    # s = 4 # number of important nodes to find
    rfaffinity = proximity_matrix(clf, X)

    signal, dictionary, rfrgam = convert_least_squares(rfaffinity, mags)
    signal = csc_matrix(signal)

    mpresults = matching_pursuit(signal, dictionary, s)
    coordinates, coefs, importances, R = mpresults
    new_diag, new_impo = diag_impo(mags, coordinates, importances, coefs)

    return signal, dictionary, rfrgam, mpresults, coordinates, \
        coefs, importances, R, new_diag, new_impo, X, Y, dic


# RECONSTRUCTING

def reconstruct_coord(coefs, coordinates, mags, s):
    """ This function is for reconstructing and obtaining a vector
        used in plotting the biplots. The resulting coord is of shape
        n_internal_nodes_found (default 5) x n_samples (?) """
    
    d, n = mags.get_shape()
    print("Reconstructing")
    coord = np.sqrt(coefs[0]) * mags[coordinates[0],:].todense()
    for i in range(1,s):
        nodei = np.sqrt(coefs[i]) * mags[coordinates[i],:].todense()
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
    d, n = mags.get_shape() # d=n internal nodes; n=n samples
    print(d, n)
    outer = np.zeros((n,n))
    print("Reconstructing")
    for i in range(s):
        print(i)
        temp = mags[coordinates[i],:].todense()
        out = np.outer(temp, temp)
        outer = outer + coefs[i]*out
    return outer


# PLOTTERS

def new_biplot3d(s, coefs, coordinates, mags, y, \
                 labeltype,dic,k,n,save,path):
    Z=np.transpose(reconstruct_coord(coefs, coordinates, mags, s))
    pca = PCA()
    x_new = pca.fit_transform(np.asarray(Z))
    score = x_new[:,0:3]
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

        for xs , ys, zs, c, label in zip(xs, ys, zs, colors, labels):
            ax.scatter(xs*scalex, ys*scaley, zs*scalez,
                        facecolors="None", edgecolors=c,label=label)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 7})

    elif labeltype == 'regression': 
        c = cm.viridis(y/max(y))
        p = ax.scatter(xs*scalex ,ys*scaley, zs*scalez,c=y, cmap='viridis')
        plt.colorbar(p, pad = 0.15)
        
    for i in range(n):
        ax.quiver(
            0, 0, 0, # starting point of vector
            1.15*coeff[i,0], 1.15*coeff[i,1], 1.15*coeff[i,2], # vector directi
            color = 'black', alpha = .7, lw = 2
        )
        ax.text(
            coeff[i,0]* 1.25, coeff[i,1] * 1.25, coeff[i,2] * 1.25,
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
    ax.dist=12
    fig.tight_layout(pad=2)

    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def new_biplot3dnormalized(s, coefs, coordinates, mags, y,\
                           labeltype, dic, k, n, save, path):
    Z = np.transpose(reconstruct_coord(coefs, coordinates, mags, s))
    scaler = StandardScaler()
    scaler.fit(np.asarray(Z))
    Z = scaler.transform(np.asarray(Z))  
    pca = PCA()
    x_new = pca.fit_transform(Z)
    score = x_new[: ,0:3]
    coeff = np.transpose(pca.components_[0:3, :])
    xs = score[:,0]
    ys = score[:,1]
    zs = score[:,2]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    scalez = 1.0 / (zs.max() - ys.min())
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(projection='3d')

    if not dic is None:
        inv_map = {v: k for k, v in dic.items()}
        labels=[inv_map[i] for i in y]
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
        p=ax.scatter(xs*scalex, ys*scaley, zs*scalez,
                     c=y, cmap='viridis')
        plt.colorbar(p, pad=0.15)
    
  
    for i in range(n):
        ax.quiver(
            0, 0, 0, # starting point of vector
            1.15*coeff[i,0], 1.15*coeff[i,1], 1.15*coeff[i,2], # vector directi
            color = 'black', alpha = .7, lw = 2
        )
        ax.text(coeff[i,0]* 1.25, coeff[i,1] * 1.25,coeff[i,2] * 1.25,
                coordinates[i], color = 'black', ha = 'center', va = 'center') 
        
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
    ax.dist=12
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

    reconstructed = reconstruct(coordinates, modmags, s, coefs)
    groups = list(Y) 
    inv_dic = {v:k for k, v in dic.items()}

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
    x_0 = axes[0].get_position().x0
    y_0 = axes[0].get_position().y0
    ax_group_bar1 = fig.add_axes([axes[0].get_position().x0,
                                  axes[0].get_position().y0,
                                  axes[0].get_position().width,
                                  group_bar_height], frameon=False)
    ax_group_bar2 = fig.add_axes([axes[1].get_position().x0,
                                  axes[1].get_position().y0,
                                  axes[1].get_position().width,
                                  group_bar_height], frameon=False)

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
        center = (start_idx[i] + end_idx[i] -1) / 2
        print(center, start_idx[i], end_idx[i])

        # Adjust the vertical position of the labels to avoid overlap
        label_y = 1
        ax_group_bar1.text(center, label_y, group_names[start_idx[i]], ha='center', va='top', rotation=90, fontsize=10)
        ax_group_bar2.text(center, label_y, group_names[start_idx[i]], ha='center', va='top', rotation=90, fontsize=10)

    corr_coeff = _get_correlation(sorted_rfgram, sorted_reconstructed)
    text = f'Pearson Correlation: {corr_coeff:.2f}'
    axes[1].text(1.1, 0.5, text, fontsize=12, va='center', ha='left', transform=axes[1].transAxes)
    # Adjust layout to fit the labels
    plt.subplots_adjust(bottom=0.2)  # Increase bottom margin to accommodate labels


    if save == True:
        plt.savefig(path, dpi=400, bbox_inches='tight')


def boxplot_plotter(mags, y, indices, dic,
                    xlabels, save, path):

    fig, ax = plt.subplots(len(indices), sharex=True,
                           figsize=(4, 2.5 * len(indices)))
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


def make_plots(adhld_results, modmags, path, s=5):

    # unpack reslts
    signal, dic, rfgram, mpresults, coordinates, coefs, \
        importances, R, diagonal, new_impo, X, Y, dic = adhld_results
    
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
    new_biplot3d(s, coefs, coordinates, modmags, Y, \
                'classification', dic, k=3, n=3, \
                save=True, path=path3)

    new_biplot3dnormalized(s, coefs, coordinates, modmags, Y, \
                        'classification', dic, k=3, n=3,
                        save=True, path=path4)