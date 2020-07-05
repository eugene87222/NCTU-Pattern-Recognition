import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from numpy.linalg import inv, det
from abc import ABC, abstractmethod

title_size = 13
text_size = 13
annot_size = 13


class Classifier(ABC):
    @abstractmethod
    def __init__(self, clf=None):
        pass

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def test(self, X):
        pass


class BayesianClassifier(Classifier):
    def __init__(self, clf=None):
        if clf is None:
            self.class_num = 0
            self.x_dim = 0
            self.prior = None
            self.mu = None
            self.cov = None
            self.cov_inv = None
            self.cov_det = None
        else:
            self.class_num = clf.class_num
            self.x_dim = clf.x_dim
            self.prior = np.copy(clf.prior)
            self.mu = np.copy(clf.mu)
            self.cov = np.copy(clf.cov)
            self.cov_inv = np.copy(clf.cov_inv)
            self.cov_det = np.copy(clf.cov_det)

    def train(self, X, y):
        _, cnt = np.unique(y, return_counts=True)

        self.class_num = cnt.shape[0]
        self.x_dim = X.shape[1]
        self.prior = np.zeros(self.class_num, dtype=np.float64)
        self.mu = np.zeros((self.class_num, self.x_dim), dtype=np.float64)
        self.cov = np.zeros(
            (self.class_num, self.x_dim, self.x_dim),
            dtype=np.float64)
        self.cov_inv = np.zeros(
            (self.class_num, self.x_dim, self.x_dim),
            dtype=np.float64)
        self.cov_det = np.zeros(self.class_num, dtype=np.float64)

        for i, c in enumerate(cnt):
            x = X[y==i]
            self.prior[i] = c / X.shape[0]
            self.mu[i] = np.mean(x, axis=0)
            self.cov[i] = np.cov(x.T) + np.eye(self.x_dim) * 1e-2
            self.cov_inv[i] = inv(self.cov[i])
            self.cov_det[i] = det(self.cov[i])

    def test(self, X):
        g = np.zeros((X.shape[0], self.class_num), dtype=np.float64)
        pred = np.zeros(X.shape[0], dtype=np.int32)

        for i in range(self.class_num):
            likelihood = -0.5 * self.x_dim * np.log(2*np.pi)
            likelihood += -0.5 * np.log(self.cov_det[i])
            delta = X - self.mu[i]
            t = delta @ self.cov_inv[i] @ delta.T
            likelihood += -0.5 * np.diagonal(t)
            g[:, i] = likelihood + np.log(self.prior[i])

        g = (g.T / np.abs(np.sum(g, axis=1))).T
        pred = np.argmax(g, axis=1)

        return g, pred


class NaiveBayesClasssifier(Classifier):
    def __init__(self, clf=None):
        if clf is None:
            self.class_num = 0
            self.x_dim = 0
            self.prior = None
            self.mu = None
            self.var = None
        else:
            self.class_num = clf.class_num
            self.x_dim = clf.x_dim
            self.prior = np.copy(clf.prior)
            self.mu = np.copy(clf.mu)
            self.var = np.copy(clf.var)

    def train(self, X, y):
        _, cnt = np.unique(y, return_counts=True)

        self.class_num = cnt.shape[0]
        self.x_dim = X.shape[1]
        self.prior = np.zeros(self.class_num, dtype=np.float64)
        self.mu = np.zeros((self.class_num, self.x_dim), dtype=np.float64)
        self.var = np.zeros((self.class_num, self.x_dim), dtype=np.float64)

        for i, c in enumerate(cnt):
            x = X[y==i]
            self.prior[i] = c / X.shape[0]
            self.mu[i] = np.mean(x, axis=0)
            self.var[i] = np.var(x, axis=0) + 1e-2

    def test(self, X):
        g = np.zeros((X.shape[0], self.class_num), dtype=np.float64)
        pred = np.zeros(X.shape[0], dtype=np.int32)

        for i in range(self.class_num):
            t = -0.5 * np.log(2*np.pi*self.var[i])
            likelihood = np.sum(t)
            t = -0.5 * (X-self.mu[i])**2 / self.var[i]
            likelihood += np.sum(t, axis=1)
            g[:, i] = likelihood + np.log(self.prior[i])

        g = (g.T / np.abs(np.sum(g, axis=1))).T
        pred = np.argmax(g, axis=1)

        return g, pred


def sign(num):
    return 1 if num >= 0 else -1


def error_rate(w, X, y):
    error = 0
    for i in range(X.shape[0]):
        if sign(X[i].dot(w)) != y[i]:
            error += 1
    return error / X.shape[0]


class PLA(Classifier):
    def __init__(self, clf=None):
        if clf is None:
            self.class_num = 2
            self.x_dim = 0
            self.w = None
        else:
            self.class_num = clf.class_num
            self.x_dim = clf.x_dim
            self.w = np.copy(clf.w)

    def train(self, X, y):
        self.x_dim = X.shape[1]
        pocket = np.zeros(self.x_dim, dtype=np.float64)
        w = np.zeros(self.x_dim, dtype=np.float64)
        idx, iteration = 0, 0
        while 1:
            if sign(X[idx].dot(w)) != y[idx]:
                yx = y[idx] * X[idx]
                w += 0.2 * yx
                if error_rate(w, X, y) < error_rate(pocket, X, y):
                    pocket = np.copy(w)
            idx = (idx+1) % X.shape[0]
            iteration += 1
            if iteration>=5000 or error_rate(pocket, X, y)<0.1:
                break
        self.w = np.copy(pocket)

    def test(self, X):
        g = np.zeros((X.shape[0], 1), dtype=np.float64)
        pred = np.zeros(X.shape[0], dtype=np.int32)

        for i, x in enumerate(X):
            g[i, 0] = x.dot(self.w)
            pred[i] = sign(g[i])

        return g, pred


def load_data(path, two_class):
    start_idx = {True: -1, False: 0}
    step = {True: 2, False: 1}

    label = {}
    encode_y = start_idx[two_class]
    X, y = [], []

    with open(path, 'r') as file:
        for line in file:
            if not line.strip():
                break
            t = line.strip().split(',')
            if label.get(t[-1]) is None:
                label[t[-1]] = encode_y
                encode_y += step[two_class]

            X.append(np.asarray(t[:-1]).astype(np.float64))
            y.append(label[t[-1]])

    X = np.asarray(X)
    if two_class:
        X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = np.asarray(y)

    return X, y, label


def train_test_split(X, y, train_ratio):
    train_size = int(np.ceil(X.shape[0] * train_ratio))
    idx = np.arange(X.shape[0])
    train_idx = np.random.choice(idx, train_size, replace=False)
    train_idx = np.sort(train_idx)

    mask = np.ma.array(idx, mask=False)
    mask.mask[train_idx] = True
    test_idx = mask.compressed()

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def get_cm(y, pred, class_num):
    cm = np.zeros((class_num, class_num), dtype=np.int32)
    for i, j in zip(y, pred):
        cm[i, j] += 1
    return cm


def get_roc_cm(y, g, thres, is_PLA):
    cm = np.zeros((2, 2), dtype=np.int32)
    truth = y==0
    for t, score in zip(truth, g):
        # Confusion Matrix: cm
        # ---------------------------------------
        # |   cm[0, 0]: TP   |   cm[0, 1]: FN   |
        # |                  |                  |
        # |   predict true,  |   predict false, |
        # |   actually true  |   actually true  |
        # ---------------------------------------
        # |   cm[1, 0]: FP   |   cm[1, 1]: TN   |
        # |                  |                  |
        # |   predict true,  |   predict false, |
        # |   actually false |   actually false |
        # ---------------------------------------
        if is_PLA:
            cm[int(~t)][int(~(score<thres))] += 1
        else:
            cm[int(~t)][int(~(score>thres))] += 1
    cm_dict = {
        'tp': cm[0, 0], 'fp': cm[1, 0],
        'tn': cm[1, 1], 'fn': cm[0, 1]
    }
    return cm_dict


def get_fpr(cm):
    return cm['fp'] / (cm['fp']+cm['tn']) if (cm['fp']+cm['tn']) else 0


def get_tpr(cm):
    return cm['tp'] / (cm['tp']+cm['fn']) if (cm['tp']+cm['fn']) else 0


def get_auc(fpr, tpr):
    x = np.asarray(fpr+[1])
    y = np.asarray(tpr+[0])
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1))-np.dot(x, np.roll(y, 1)))


def get_roc(y, g, is_PLA):
    low, high = np.min(g), np.max(g)
    step = (abs(low) + abs(high)) / 1000
    thres = np.arange(low-2*step, high+2*step, step)
    if is_PLA:
        thres = thres[::-1]

    cms = []
    for t in thres:
        cms.append(get_roc_cm(y, g, t, is_PLA))

    fpr = list(map(get_fpr, cms))
    tpr = list(map(get_tpr, cms))

    return fpr, tpr, thres


def evaluation(y, pred, g):
    new_y = np.copy(y)
    new_pred = np.copy(pred)
    if g.shape[1] == 1:
        np.place(new_y, new_y==-1, 0)
        np.place(new_pred, new_pred==-1, 0)
    class_num = max(g.shape[1], 2)
    cm = get_cm(new_y, new_pred, class_num)
    if g.shape[1] <= 2:
        roc = get_roc(new_y, g[:, 0], g.shape[1]==1)
    else:
        roc = None

    return cm, roc


def k_fold(clf, X_train, y_train, X_test, y_test, label, k):
    print(label)

    data_idx = np.arange(X_train.shape[0])
    np.random.shuffle(data_idx)
    fold = np.asarray(np.array_split(data_idx, k))
    k_idx = np.arange(k)

    class_num = len(label.keys())
    cms = np.zeros((class_num, class_num), dtype=np.int32)
    rocs = []
    color = ['r', 'darkorange', 'gold', 'limegreen', 'royalblue', 'blueviolet']

    for i, f in enumerate(fold):
        train_idx = np.concatenate(fold[k_idx[k_idx!=i]])
        test_idx = np.concatenate(fold[k_idx[k_idx==i]])
        X_train_sub, y_train_sub = X_train[train_idx], y_train[train_idx]
        X_test_sub, y_test_sub = X_train[test_idx], y_train[test_idx]
        clf.train(X_train_sub, y_train_sub)
        g, pred = clf.test(X_test_sub)

        acc = (y_test_sub==pred).astype(np.int32)
        acc = np.sum(acc) / y_test_sub.shape[0]
        print(f'accuracy fold {i}: {acc:>.3f}')

        cm, roc = evaluation(y_test_sub, pred, g)
        cms += cm
        rocs.append(roc)

    clf.train(X_train, y_train)
    g, pred = clf.test(X_test)

    test_acc = (y_test==pred).astype(np.int32)
    test_acc = np.sum(test_acc) / y_test.shape[0]
    print(f'accuracy test: {test_acc:>.3f}')

    ax = sn.heatmap(
        cms, annot=True, annot_kws={'size': annot_size}, cmap='GnBu',
        square=True, fmt='g', cbar_kws={'format': '%d'})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=annot_size)
    plt.gca().set_xticklabels(label.keys(), fontsize=text_size)
    plt.gca().set_yticklabels(
        label.keys(), va='center', rotation=0, fontsize=text_size)

    plt.title(f'acc: {test_acc:.3f}\n', fontsize=title_size)
    plt.savefig('cm.png', dpi=300, transparent=True)
    plt.clf()

    if class_num == 2:
        mean_fpr = np.linspace(0.0, 1.0, 1000)
        tprs = []
        avg_auc = 0
        for i, roc in enumerate(rocs):
            fpr, tpr, thres = roc
            tprs.append(np.interp(mean_fpr, fpr[::-1], tpr[::-1]))
            auc = get_auc(fpr, tpr)
            avg_auc += auc
            plt.plot(
                fpr, tpr, c=color[i%len(color)], lw=2, alpha=0.3,
                label=f'ROC fold {i} (AUC={auc:>.3f})')

        mean_tpr = np.mean(tprs, axis=0)
        plt.plot(mean_fpr, mean_tpr, c='b', lw=3, alpha=1.0, label=f'ROC mean (AUC={avg_auc/k:>.3f})')

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr+std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr-std_tpr, 0)
        plt.fill_between(
            mean_fpr, tprs_lower, tprs_upper, color='grey',
            alpha=0.2, label=r'$\pm$ 1 std. dev.')

        plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        plt.xlim([-0.01, 1])
        plt.ylim([0, 1.01])
        plt.xlabel('FP / (TN+FP)', fontsize=text_size)
        plt.ylabel('TP / (TP+FN)', fontsize=text_size)
        plt.legend(loc='lower right')
        plt.gca().set_aspect('equal')
        plt.savefig('roc.png', dpi=300, transparent=True)
        plt.clf()


if __name__ == '__main__':
    dataset = 'ionosphere'

    X, y, label = load_data(f'./{dataset}/data', False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.7)

    clf = BayesianClassifier()
    k_fold(clf, X_train, y_train, X_test, y_test, label, 5)
