import numpy as np
from tinybenchmarks_utils import *
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances

from tinybenchmarks_irt import *

random_state = 42

def tinybenchmarks_wrapper_all_num_medoids(Y, source_models, target_models, all_num_medoids, run_idx):
    N = Y.shape[1]
    n_sub = 1
    balance_weights = np.ones(Y.shape[1])

    Y_train = Y[source_models, :]
    Y_test = Y[target_models, :]

    # print(Y_test.mean(axis=1))

    # make sure Y is binarized if we need to. don't need to for any GLUE tasks
    Y_bin_train = Y_train
    Y_bin_test = Y_test

    device = 'cpu' # Either 'cuda' or 'cpu' 
    epochs = 2000  # Number of epochs for IRT model training (py-irt default is 2000)
    lr = .1  # Learning rate for IRT model training (py-irt default is .1)
    D = 10 # Dimensions

    dataset_no_saving = create_irt_dataset_no_saving(Y_bin_train)

    best_parameters = train_irt_model_no_saving(dataset=dataset_no_saving, 
                    model_name=f'data/irt_model_{run_idx}', 
                    D=D, lr=lr, epochs=epochs, device=device)   

    # now select 'anchor points' with trained IRT model
    clustering = 'irt' # 'correct.' or 'irt'
    anchor_points = []
    anchor_weights = []

    if clustering=='correct.':
        X = Y_train.T
    elif clustering=='irt':
        A, B, _ = load_irt_parameters_no_saving(best_parameters)
        X = np.vstack((A.squeeze(), B.squeeze().reshape((1,-1)))).T
    else:
        raise NotImplementedError 
            
    norm_balance_weights = balance_weights
    norm_balance_weights /= norm_balance_weights.sum()

    # Fitting the KMeans model
    all_Y_hat = []
    all_Y_anchor = []
    all_anchor_weights = []
    for num_medoids in all_num_medoids:
        run_num = int(run_idx.split('_')[0])
        print(f'tinybenchmarks, {num_medoids}, {run_num}')
        kmeans = KMeans(n_clusters=num_medoids, n_init="auto", random_state=random_state)
        kmeans.fit(X, sample_weight=norm_balance_weights)

        # Calculating anchor points
        anchor_points = pairwise_distances(kmeans.cluster_centers_, X, metric='euclidean').argmin(axis=1)

        # Calculating anchor weights
        anchor_weights = np.array([np.sum(norm_balance_weights[kmeans.labels_==c]) for c in range(num_medoids)])

        Y_anchor = Y_test[:,anchor_points]
        Y_hat = (Y_anchor*anchor_weights).sum(axis=1)
        Y_true = (balance_weights*Y_test).sum(axis=1)

        print(f'Y_hat, {num_medoids} medoids, run {run_num}: {Y_hat}')
        print(f'Y_true {num_medoids} medoids, run {run_num}: {Y_true}')
        print(f"avg. error {num_medoids} medoids, run {run_num}: {np.abs(Y_hat-Y_true).mean():.3f}")

        all_Y_hat.append(Y_hat)
        all_Y_anchor.append(Y_anchor)
        all_anchor_weights.append(anchor_weights)

    return all_Y_hat, all_Y_anchor, all_anchor_weights