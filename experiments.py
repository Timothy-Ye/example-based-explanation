import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import TSNE

from influential_points.utils.k_selector import select_top_sum_redun_k
from influential_points.utils.k_selector import select_top_diverse_k

hls = sns.color_palette("hls", 10)

class ExperimentRunner(object):
    """
    Class for repeating experiments across different models and image datasets.
    """

    def __init__(
        self,
        inf, # Respective scores.
        rif,
        gc,
        rp,
        ds,
        train_x, # Dataset used.
        train_y,
        test_x,
        test_y,
        num_train_points,
        num_test_points,
        classes, # Readable and comparable labels.
        train_true,
        test_true,
        train_preds,
        test_preds,
    ):
        self.inf = inf
        self.rif = rif
        self.gc = gc
        self.rp = rp
        self.ds = ds
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.num_train_points = num_train_points
        self.num_test_points = num_test_points
        self.classes = classes
        self.train_true = train_true
        self.test_true = test_true
        self.train_preds = train_preds
        self.test_preds = test_preds

        self.tsne = None

    def disp_global(self):
        """
        Visualise global examples.
        """

        idxs = [
            np.argsort(np.sum(self.inf, axis=1)),
            np.argsort(np.sum(self.rif, axis=1)),
            np.argsort(np.sum(self.gc, axis=1)),
            np.argsort(np.sum(self.rp, axis=1)),
            np.argsort(self.ds)
        ]

        ylabels = [
            "IF",
            "RIF",
            "GC",
            "RP",
            "DS"
        ]

        fig, ax = plt.subplots(
            5,
            7,
            subplot_kw={'xticks':[], 'yticks':[]},
            figsize=(7, 6)
        )

        ax[0, 1].set_title("Positive Examples", fontdict={"fontsize": 14})
        ax[0, 5].set_title("Negative Examples", fontdict={"fontsize": 14})

        for i in range(5):
            ax[i, 0].set_ylabel(ylabels[i], fontdict={"fontsize": 14})
            ax[i, 3].set_visible(False)

            for j in range(3):
                ax[i, j].imshow(self.train_x[idxs[i][-j-1]], cmap=plt.cm.binary)
                ax[i, j].set_xlabel("{}".format(self.train_true[idxs[i][-j-1]]))

                ax[i, j+4].imshow(self.train_x[idxs[i][j]], cmap=plt.cm.binary)
                ax[i, j+4].set_xlabel("{}".format(self.train_true[idxs[i][j]]))

    def disp_local(self, test_idx):
        """
        Visualise local examples.
        """

        idxs = [
            np.argsort(self.inf[:, test_idx]),
            np.argsort(self.rif[:, test_idx]),
            np.argsort(self.gc[:, test_idx]),
            np.argsort(self.rp[:, test_idx])
        ]

        ylabels = [
            "IF",
            "RIF",
            "GC",
            "RP"
        ]

        fig, ax = plt.subplots(
            4,
            9,
            subplot_kw={'xticks':[], 'yticks':[]},
            figsize=(9, 5)
        )

        ax[0, 3].set_title("Positive Examples", fontdict={"fontsize": 14})
        ax[0, 7].set_title("Negative Examples", fontdict={"fontsize": 14})

        ax[0, 0].set_title("Test Point", fontdict={"fontsize": 14})
        ax[0, 0].imshow(self.test_x[test_idx], cmap=plt.cm.binary)
        ax[0, 0].set_xlabel("{}\nPredicted: {}".format(self.test_true[test_idx], self.test_preds[test_idx]))

        for i in range(4):
            ax[i, 2].set_ylabel(ylabels[i], fontdict={"fontsize": 14})
            ax[i, 1].set_visible(False)
            ax[i, 5].set_visible(False)

            for j in range(3):
                ax[i, j+2].imshow(self.train_x[idxs[i][-j-1]], cmap=plt.cm.binary)
                ax[i, j+2].set_xlabel("{}".format(self.train_true[idxs[i][-j-1]]))

                ax[i, j+6].imshow(self.train_x[idxs[i][j]], cmap=plt.cm.binary)
                ax[i, j+6].set_xlabel("{}".format(self.train_true[idxs[i][j]]))

        for i in range(1, 4):
            ax[i, 0].set_visible(False)

    def apd(self):
        """
        Calculates and plots average pairwise distance between local examples for each method.
        """

        ks = [3, 5, 10]

        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))
        rng = np.random.default_rng(0)
        N = 1000

        labels = [
            "RDM (All)",
            "RDM (Same)",
            "IF",
            "RIF",
            "GC",
            "RP"
        ]

        apds = []
        errs = []

        for k in ks:
            # Baseline using random points from any class.
            all_class_idxs = [rng.choice(self.num_train_points, size=k) for _ in range(N)]

            # Baseline using random points from the same class (averaged over all classes).
            same_class_idxs = []
            for i in self.classes:
                same_class_idxs += [rng.choice(np.arange(self.num_train_points)[self.train_true == i], size=k) for _ in range(N)]

            # Average over all local explanations for each method.
            inf_idxs = np.argsort(self.inf, axis=0)[-k:].T
            rif_idxs = np.argsort(self.rif, axis=0)[-k:].T
            gc_idxs = np.argsort(self.gc, axis=0)[-k:].T
            rp_idxs = np.argsort(self.rp, axis=0)[-k:].T

            idxs = [
                all_class_idxs,
                same_class_idxs,
                inf_idxs,
                rif_idxs,
                gc_idxs,
                rp_idxs
            ]

            sub_apds = []
            sub_errs = []

            # Calculate APD and standard deviation.
            for i in idxs:
                pds = [np.sum(pairwise_distances(reshaped_train_x[j])) / (k*(k-1)) for j in i]

                apd = np.mean(pds)
                err = np.std(pds)

                sub_apds += [apd]
                sub_errs += [err]

            apds += [sub_apds]
            errs += [sub_errs]

        # Print results.
        for i in range(len(ks)):
            print("k = {}:".format(ks[i]))

            for j in range(len(labels)):
                print("{: >10}: {:.3f} ± {:.3f}".format(labels[j], apds[i][j], errs[i][j]))

        # Plot results.
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(labels))
        width = 0.7 / len(ks)

        for i in range(len(ks)):
            ax.bar(x+i*width, apds[i], width, label=r"$k$ = {}".format(ks[i]))
            ax.errorbar(x+i*width, apds[i], yerr=errs[i], fmt="none", ecolor="black", capsize=5)

        ax.set_ylabel("Average Pairwise Distance", fontdict={"fontsize": 14})
        ax.set_xlabel("Method", fontdict={"fontsize": 14})
        ax.set_xticks(x+0.35-width/2)
        ax.set_xticklabels(labels)
        ax.legend(title=r"$k$-most influential", loc="upper right")

    def gamma_tradeoff(self):
        """
        Plots tradeoff curve of global explanation metric vs. APD when varying gamma used in DIVINE.
        """

        k = 5
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        scores = [
            np.sum(self.inf, axis=1),
            np.sum(self.rif, axis=1),
            np.sum(self.gc, axis=1),
            np.sum(self.rp, axis=1),
            self.ds
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP",
            "DS"
        ]

        gammas = np.append([0], np.logspace(-3, 9, num=50))

        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(5):
            idxs = [select_top_diverse_k(scores[i], k, reshaped_train_x, gamma=g) for g in gammas]

            xs = [np.sum(pairwise_distances(reshaped_train_x[j])) / (k*(k-1)) for j in idxs]
            ys = [np.sum(scores[i][j]) for j in idxs]
            ys = ys / ys[0] # Normalise

            ax.plot(xs, ys, label=labels[i], zorder=1)
            # ax.scatter(xs, ys, s=16, marker="x")

            # Maximum metric.
            ax.scatter(xs[0], ys[0], s=64, marker="D", color=hls[0], edgecolors="black", zorder=2)

            # Maximum product.
            prod = np.argmax(xs * ys)
            ax.scatter(xs[prod], ys[prod], s=64, marker="D", color=hls[3], edgecolors="black", zorder=2)
            print("Optimal Gamma for {} = {:3e}".format(labels[i], gammas[prod]))

            # Maximum APD.
            max_apd = np.argmax(xs)
            ax.scatter(xs[max_apd], ys[max_apd], s=64, marker="D", color=hls[6], edgecolors="black", zorder=2)

        ax.set_ylabel("Normalised Metric", fontdict={"fontsize": 14})
        ax.set_xlabel("Average Pairwise Distance", fontdict={"fontsize": 14})
        ax.legend(title="Method", loc="lower left")

    def gamma_tradeoff_sr(self):
        """
        Plots tradeoff curve of global explanation metric vs. SR when varying gamma used in DIVINE.
        """

        k = 5
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))
        kappa = np.sum(rbf_kernel(reshaped_train_x))

        scores = [
            np.sum(self.inf, axis=1),
            np.sum(self.rif, axis=1),
            np.sum(self.gc, axis=1),
            np.sum(self.rp, axis=1),
            self.ds
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP",
            "DS"
        ]

        gammas = np.append([0], np.logspace(-3, 9, num=10))

        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(5):
            idxs = [select_top_sum_redun_k(scores[i], k, reshaped_train_x, gamma=g) for g in gammas]

            xs = [kappa - np.sum(rbf_kernel(reshaped_train_x[j])) for j in idxs]
            ys = [np.sum(scores[i][j]) for j in idxs]
            ys = ys / ys[0] # Normalise~

            ax.plot(xs, ys, label=labels[i], zorder=1)
            # ax.scatter(xs, ys, s=16, marker="x")

            # Maximum metric.
            ax.scatter(xs[0], ys[0], s=64, marker="D", color=hls[0], edgecolors="black", zorder=2)

            # Maximum product.
            prod = np.argmax(xs * ys)
            ax.scatter(xs[prod], ys[prod], s=64, marker="D", color=hls[3], edgecolors="black", zorder=2)

            # Maximum APD.
            max_apd = np.argmax([np.sum(pairwise_distances(reshaped_train_x[j])) for j in idxs])
            ax.scatter(xs[max_apd], ys[max_apd], s=64, marker="D", color=hls[6], edgecolors="black", zorder=2)


        ax.set_ylabel("Normalised Metric", fontdict={"fontsize": 14})
        ax.set_xlabel("Sum Redundancy", fontdict={"fontsize": 14})
        ax.legend(title="Method", loc="lower left")

    def gamma_tradeoff_local(self, test_idx):
        """
        Plots tradeoff curve of local explanation metric vs. APD when varying gamma used in DIVINE.
        """

        k = 5
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        scores = [
            self.inf[:, test_idx],
            self.rif[:, test_idx],
            self.gc[:, test_idx],
            self.rp[:, test_idx]
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP"
        ]

        gammas = np.append([0], np.logspace(-6, 6, num=50))

        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(4):
            idxs = [select_top_diverse_k(scores[i], k, reshaped_train_x, gamma=g) for g in gammas]

            xs = [np.sum(pairwise_distances(reshaped_train_x[j])) / (k*(k-1)) for j in idxs]
            ys = [np.sum(scores[i][j]) for j in idxs]
            ys = ys / ys[0] # Normalise~

            ax.plot(xs, ys, label=labels[i], zorder=1)
            # ax.scatter(xs, ys, s=16, marker="x")

            # Maximum metric.
            ax.scatter(xs[0], ys[0], s=64, marker="D", color=hls[0], edgecolors="black", zorder=2)

            # Maximum product.
            prod = np.argmax(xs * ys)
            ax.scatter(xs[prod], ys[prod], s=64, marker="D", color=hls[3], edgecolors="black", zorder=2)
            print("Optimal Gamma for {} = {:3e}".format(labels[i], gammas[prod]))

            # Maximum APD.
            max_apd = np.argmax(xs)
            ax.scatter(xs[max_apd], ys[max_apd], s=64, marker="D", color=hls[6], edgecolors="black", zorder=2)


        ax.set_ylabel("Normalised Metric", fontdict={"fontsize": 14})
        ax.set_xlabel("Average Pairwise Distance", fontdict={"fontsize": 14})
        ax.legend(title="Method", loc="lower left")

    def divine_global(self, gammas):
        """
        Visualises global examples with and without DIVINE.
        """

        k = 3
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        normal_idxs = [
            np.argsort(np.sum(self.inf, axis=1)),
            np.argsort(np.sum(self.rif, axis=1)),
            np.argsort(np.sum(self.gc, axis=1)),
            np.argsort(np.sum(self.rp, axis=1)),
            np.argsort(self.ds)
        ]

        divine_idxs = [
            select_top_diverse_k(np.sum(self.inf, axis=1), k, reshaped_train_x, gamma=gammas["inf"]),
            select_top_diverse_k(np.sum(self.rif, axis=1), k, reshaped_train_x, gamma=gammas["rif"]),
            select_top_diverse_k(np.sum(self.gc, axis=1), k, reshaped_train_x, gamma=gammas["gc"]),
            select_top_diverse_k(np.sum(self.rp, axis=1), k, reshaped_train_x, gamma=gammas["rp"]),
            select_top_diverse_k(self.ds, k, reshaped_train_x, gamma=gammas["ds"]),
        ]

        ylabels = [
            "IF",
            "RIF",
            "GC",
            "RP",
            "DS"
        ]

        fig, ax = plt.subplots(
            5,
            7,
            subplot_kw={'xticks':[], 'yticks':[]},
            figsize=(7, 6)
        )

        ax[0, 1].set_title("Base", fontdict={"fontsize": 14})
        ax[0, 5].set_title("DIVINE", fontdict={"fontsize": 14})

        for i in range(5):
            ax[i, 0].set_ylabel(ylabels[i], fontdict={"fontsize": 14})
            ax[i, 3].set_visible(False)

            for j in range(3):
                ax[i, j].imshow(self.train_x[normal_idxs[i][-j-1]], cmap=plt.cm.binary)
                ax[i, j].set_xlabel("{}".format(self.train_true[normal_idxs[i][-j-1]]))

                ax[i, j+4].imshow(self.train_x[divine_idxs[i][j]], cmap=plt.cm.binary)
                ax[i, j+4].set_xlabel("{}".format(self.train_true[divine_idxs[i][j]]))

    def divine_local(self, test_idx, gammas):
        """
        Visualises local examples with and without DIVINE.
        """

        k = 3
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        normal_idxs = [
            np.argsort(self.inf[:, test_idx]),
            np.argsort(self.rif[:, test_idx]),
            np.argsort(self.gc[:, test_idx]),
            np.argsort(self.rp[:, test_idx])
        ]

        divine_idxs = [
            select_top_diverse_k(self.inf[:, test_idx], k, reshaped_train_x, gamma=gammas["inf"]),
            select_top_diverse_k(self.rif[:, test_idx], k, reshaped_train_x, gamma=gammas["rif"]),
            select_top_diverse_k(self.gc[:, test_idx], k, reshaped_train_x, gamma=gammas["gc"]),
            select_top_diverse_k(self.rp[:, test_idx], k, reshaped_train_x, gamma=gammas["rp"]),
        ]

        ylabels = [
            "IF",
            "RIF",
            "GC",
            "RP",
        ]

        fig, ax = plt.subplots(
            4,
            9,
            subplot_kw={'xticks':[], 'yticks':[]},
            figsize=(9, 5)
        )

        ax[0, 3].set_title("Base", fontdict={"fontsize": 14})
        ax[0, 7].set_title("DIVINE", fontdict={"fontsize": 14})

        ax[0, 0].set_title("Test Point", fontdict={"fontsize": 14})
        ax[0, 0].imshow(self.test_x[test_idx], cmap=plt.cm.binary)
        ax[0, 0].set_xlabel("{}\nPredicted: {}".format(self.test_true[test_idx], self.test_preds[test_idx]))

        for i in range(4):
            ax[i, 2].set_ylabel(ylabels[i], fontdict={"fontsize": 14})
            ax[i, 1].set_visible(False)
            ax[i, 5].set_visible(False)

            for j in range(3):
                ax[i, j+2].imshow(self.train_x[normal_idxs[i][-j-1]], cmap=plt.cm.binary)
                ax[i, j+2].set_xlabel("{}".format(self.train_true[normal_idxs[i][-j-1]]))

                ax[i, j+6].imshow(self.train_x[divine_idxs[i][j]], cmap=plt.cm.binary)
                ax[i, j+6].set_xlabel("{}".format(self.train_true[divine_idxs[i][j]]))

        for i in range(1, 4):
            ax[i, 0].set_visible(False)

    def unique_examples(self):
        """
        Plots the number of unique examples across all local examples for each method.
        """

        ks = [1, 3, 5, 10]

        idxs = [
            np.argsort(self.inf, axis=0),
            np.argsort(self.rif, axis=0),
            np.argsort(self.gc, axis=0),
            np.argsort(self.rp, axis=0),
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP",
        ]

        fig, ax = plt.subplots(figsize=(12, 8))

        xs = np.arange(len(ks))
        width = 0.7 / len(ks)

        for i in range(len(ks)):
            num_unique = [len(np.unique(j[-ks[i]:, :])) for j in idxs]

            rects = ax.bar(xs+i*width, num_unique, width, label=r"$k$ = {}".format(ks[i]))

            for j in range(len(idxs)):
                ax.annotate(
                    num_unique[j],
                    xy=(xs[j]+i*width, num_unique[j]),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom")

        ax.set_ylabel("Number of Unique Examples", fontdict={"fontsize": 14})
        ax.set_xlabel("Method", fontdict={"fontsize": 14})
        ax.set_xticks(xs+0.35-width/2)
        ax.set_xticklabels(labels)
        ax.legend(title=r"$k$-most influential", loc="upper right")

    def id_class_test(self):
        """
        Performs and plots results of identical class test.
        """

        ks = [1, 3, 5, 10]

        idxs = [
            np.argsort(self.inf, axis=0),
            np.argsort(self.rif, axis=0),
            np.argsort(self.gc, axis=0),
            np.argsort(self.rp, axis=0),
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP",
        ]

        probs = []
        errs = []

        for k in ks:
            sub_probs = []
            sub_errs = []

            # Calculate mean and standard deviation.
            for i in idxs:
                same_class = [(self.train_true[i[-k:, j]] == self.test_preds[j]) for j in range(self.num_test_points)]

                prob = np.mean(same_class)
                err = np.std(same_class)

                sub_probs += [prob]
                sub_errs += [err]

            probs += [sub_probs]
            errs += [sub_errs]

        # Print results.
        for i in range(len(ks)):
            print("k = {}:".format(ks[i]))

            for j in range(len(labels)):
                print("{: >3}: {:.3f} ± {:.3f}".format(labels[j], probs[i][j], errs[i][j]))

        # Plot results.
        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(labels))
        width = 0.7 / len(ks)

        for i in range(len(ks)):
            ax.bar(x+i*width, probs[i], width, label=r"$k$ = {}".format(ks[i]))
            ax.errorbar(x+i*width, probs[i], yerr=errs[i], fmt="none", ecolor="black", capsize=5)

        ax.set_ylabel("Identical Class Probability", fontdict={"fontsize": 14})
        ax.set_xlabel("Method", fontdict={"fontsize": 14})
        ax.set_xticks(x+0.35-width/2)
        ax.set_xticklabels(labels)
        ax.legend(title=r"$k$-most influential", loc="upper right")

    def fit_tsne(self, perplexity=30.0, init="pca"):
        """
        Projects all the training points onto a 2D space using TSNE, for visualisation.
        """

        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        self.tsne = TSNE(perplexity=perplexity, init=init).fit_transform(reshaped_train_x)

    def tsne_global(self, divine=False, gammas=None):
        """
        Visualises global examples in 2D space created by TSNE.
        """

        k = 3
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        idxs = [
            np.argsort(np.sum(self.inf, axis=1))[-k:],
            np.argsort(np.sum(self.rif, axis=1))[-k:],
            np.argsort(np.sum(self.gc, axis=1))[-k:],
            np.argsort(np.sum(self.rp, axis=1))[-k:],
            np.argsort(self.ds)[-k:],
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP",
            "DS"
        ]

        colours = range(5)

        if divine:
            idxs += [
                select_top_diverse_k(np.sum(self.inf, axis=1), k, reshaped_train_x, gamma=gammas["inf"]),
                select_top_diverse_k(np.sum(self.rif, axis=1), k, reshaped_train_x, gamma=gammas["rif"]),
                select_top_diverse_k(np.sum(self.gc, axis=1), k, reshaped_train_x, gamma=gammas["gc"]),
                select_top_diverse_k(np.sum(self.rp, axis=1), k, reshaped_train_x, gamma=gammas["rp"]),
                select_top_diverse_k(self.ds, k, reshaped_train_x, gamma=gammas["ds"]),
            ]

            labels += [
                "IF+DIVINE",
                "RIF+DIVINE",
                "GC+DIVINE",
                "RP+DIVINE",
                "DS+DIVINE"
            ]

            colours = range(10)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.scatter(self.tsne[:, 0], self.tsne[:, 1], color="C7")

        for i in range(len(idxs)):
            ax.scatter(self.tsne[idxs[i], 0], self.tsne[idxs[i], 1], color=hls[colours[i]], s=64, marker="D", edgecolors="black", label=labels[i])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(title="Method", loc="upper right")


    def tsne_local(self, test_idx, divine=False, gammas=None):
        """
        Visualises global examples in 2D space created by TSNE.
        """

        k = 3
        reshaped_train_x = self.train_x.reshape((self.num_train_points, -1))

        idxs = [
            np.argsort(self.inf[:, test_idx])[-k:],
            np.argsort(self.rif[:, test_idx])[-k:],
            np.argsort(self.gc[:, test_idx])[-k:],
            np.argsort(self.rp[:, test_idx])[-k:],
        ]

        labels = [
            "IF",
            "RIF",
            "GC",
            "RP"
        ]

        colours = range(4)

        if divine:
            idxs += [
                select_top_diverse_k(self.inf[:, test_idx], k, reshaped_train_x, gamma=gammas["inf"]),
                select_top_diverse_k(self.rif[:, test_idx], k, reshaped_train_x, gamma=gammas["rif"]),
                select_top_diverse_k(self.gc[:, test_idx], k, reshaped_train_x, gamma=gammas["gc"]),
                select_top_diverse_k(self.rp[:, test_idx], k, reshaped_train_x, gamma=gammas["rp"]),
            ]

            labels += [
                "IF+DIVINE",
                "RIF+DIVINE",
                "GC+DIVINE",
                "RP+DIVINE"
            ]

            colours = range(8)

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.scatter(self.tsne[:, 0], self.tsne[:, 1], color="C7")

        for i in range(len(idxs)):
            ax.scatter(self.tsne[idxs[i], 0], self.tsne[idxs[i], 1], color=hls[colours[i]], s=64, marker="D", edgecolors="black", label=labels[i])

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.legend(title="Method", loc="upper right")
