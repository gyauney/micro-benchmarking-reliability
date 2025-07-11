import numpy as np
from numpy.linalg import lstsq as LS
from scipy.special import expit, logit
import kmedoids

# for DPP anchor selection
from sklearn.decomposition import PCA
from dpp_src.samplers import draw_discrete_OPE


class AnchorPointPredictor:
    def __init__(self, dataset, test_data, train_data, num_anchors, use_logit=True):
        _, num_points, num_classes = test_data.shape

        self.use_logit = use_logit

        if self.use_logit:
            original_train_data = train_data
            self.train_data = logit(train_data)
            self.test_data = logit(test_data)
        else:
            self.train_data = train_data
            self.test_data = test_data

        self.dataset_name = dataset
        self.num_anchors = num_anchors
        self.num_floaters = num_points - num_anchors
        self.num_points = num_points
        self.use_logit = use_logit
        self.num_classes = num_classes

        # slopes (y = mx + b)
        self.M = np.zeros((self.num_floaters, num_anchors, num_classes))

        # biases
        self.B = np.zeros((self.num_floaters, num_anchors, num_classes))

        # residuals (error for each anchor on each point)
        self.resid = np.zeros((self.num_floaters, num_anchors, num_classes))

        # indicator of which anchor point is nearest to any given point
        self.nearest = np.zeros((self.num_floaters, num_anchors, num_classes))

    def set_corr_matrix(self):
        self.corrs = np.zeros((self.num_points, self.num_points))
        for i in range(self.num_classes):
            train_data = self.train_data[:, :, i]
            # handle underflow for classes with lots of dummy answers
            corrs = np.corrcoef(train_data, rowvar=False)
            corrs[np.where(np.isnan(corrs))] = 0
            np.fill_diagonal(corrs, 1)
            self.corrs += corrs

    def select_anchors(self, technique="kmedoids", num_clusters=30, num_from_each_cluster=1):
        if technique == "kmedoids":
            idxs = kmedoids.fasterpam(
                1 - np.abs(self.corrs), self.num_anchors, init="random"
            ).medoids
            self.anchors = idxs

            floaters = list(
                set(list(range(self.num_points))).difference(set(list(self.anchors)))
            )[: self.num_floaters]
            assert len(floaters) == self.num_floaters
            self.floaters = np.array(list(floaters))
        elif technique == "dpp":

            # get just the confidence in the correct class
            # for some reason the vectorized version doesn't yield the right shape...
            # TODO does it matter that this is now examples x examples instead of examples x models
            X = np.transpose(self.corrs)

            X = PCA(n_components=10).fit_transform(X)
            thrs = 0.01
            low, high = np.quantile(X, thrs, 0),  np.quantile(X, 1-thrs, 0)
            sli_tail = np.all(low <= X, 1) & np.all(X <= high, 1)
            final_idxs = np.arange(0, X.shape[0], 1)[sli_tail]
            X = X[sli_tail]
            X = 0.99 * (2 * (X - X.min(0)) / (X.max(0) - X.min(0)) - 1)

            
            n = len(X)
            # Init OPE sampler by pre-computing KDE on data
            # kde = KernelDensity(kernel="epanechnikov", bandwidth="scott").fit(X)
            # kde_distr = np.exp(kde.score_samples(X))
            # ab_coeff = generate_Jacobi_parameters(X)
            # Just draw from OPE to test and show in scatter plot
            samples, _ = draw_discrete_OPE(X, self.num_anchors, 1)
            self.anchors = np.squeeze(samples)

            floaters = list(
                set(list(range(self.num_points))).difference(set(list(self.anchors)))
            )[: self.num_floaters]
            assert len(floaters) == self.num_floaters
            self.floaters = np.array(list(floaters))

        elif technique == "random":
            idxs = np.random.choice(
                list(range(self.num_points)), size=self.num_anchors, replace=False
            )
            self.anchors = idxs

            floaters = list(
                set(list(range(self.num_points))).difference(set(list(self.anchors)))
            )[: self.num_floaters]
            assert len(floaters) == self.num_floaters
            self.floaters = np.array(list(floaters))
        # choose randomly from within each cluster!
        elif technique == "stratified":
            cluster_assignments = kmedoids.fasterpam(
                1 - np.abs(self.corrs), num_clusters, init="random"
            ).labels
            
            # now sample 1 or more points from each cluster
            self.anchors = []
            for i in range(num_clusters):
                cluster_idxs = np.nonzero(cluster_assignments == i)[0]
                self.anchors.extend(np.random.choice(
                    cluster_idxs, size=num_from_each_cluster, replace=True # TODO should be false!!!!
                ))
            
            floaters = list(
                set(list(range(self.num_points))).difference(set(list(self.anchors)))
            )[: self.num_floaters]
            assert len(floaters) == self.num_floaters
            self.floaters = np.array(list(floaters))

    def set_holdout(self, held_out_idx):
        self.held_out = self.test_data[held_out_idx, :]

    def fit_anchors(self):
        # TODO: vectorize this

        for class_label in range(self.num_classes):
            class_data = self.train_data[:, :, class_label]

            for i, floater in enumerate(self.floaters):
                for j, anchor in enumerate(self.anchors):
                    apoints = class_data[:, anchor]
                    fpoints = class_data[:, floater]

                    # add one for bias term
                    A = np.vstack([apoints, np.ones(len(apoints))]).T

                    # get y = mx + b terms and residual
                    # breakpoint()
                    theta, residual = LS(A, fpoints, rcond=None)[:2]

                    # models are often linearly dependent for dummy answer numbers
                    # so there won't be any residual
                    if residual.shape[0] == 0:
                        residual = 0
                    
                    # set the params
                    self.M[i, j, class_label] = theta[0]
                    self.B[i, j, class_label] = theta[1]
                    self.resid[i, j, class_label] = residual

        self.find_nearest_anchors()

    def find_nearest_anchors(self, technique="nearest"):
        # TODO: clean up

        if technique == "nearest":
            for class_label in range(self.num_classes):
                mins = np.argmin(self.resid[:, :, class_label], axis=1)

                for i, min in enumerate(mins):
                    # weigh the nearest neighbor with smallest residuals 1, all others are zero
                    self.nearest[i, min, class_label] = 1

        else:
            raise NotImplementedError(f"Technique {technique} not implemented")

    def predict(self, anchors):
        # anchors are size |A| = |num_points|

        all_preds = np.zeros((self.num_floaters, self.num_classes))
        for class_label in range(self.num_classes):
            # breakpoint()
            anchor_preds = (
                self.M[:, :, class_label] * anchors[np.newaxis, :, class_label]
                + self.B[:, :, class_label]
            )
            weighted_anchor_preds = anchor_preds * self.nearest[:, :, class_label]

            preds = np.sum(weighted_anchor_preds, axis=1)

            all_preds[:, class_label] = preds

        if self.use_logit:
            all_preds = expit(all_preds)

        return all_preds
