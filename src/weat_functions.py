import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from scipy.stats import norm
from sklearn import svm

def SC_WEAT_Projection(A, B, permutations):
    joint_associations = np.concatenate((A,B),axis=0)
    test_statistic = np.mean(A) - np.mean(B)
    effect_size = test_statistic / np.std(joint_associations,ddof=1)

    midpoint = A.shape[0]
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:,:midpoint],axis=1) - np.mean(sample_distribution[:,midpoint:],axis=1)
    p_value = min(1 - norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1)),norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1)))

    return effect_size, p_value

#Only used in ValNorm, permutation test commented out to improve speed
def SC_WEAT(w, A, B):#, permutations):
    w_normed = w / np.linalg.norm(w)
    A_normed = A / np.linalg.norm(A,axis=-1,keepdims=True)
    B_normed = B / np.linalg.norm(B,axis=-1,keepdims=True)

    A_associations = w_normed @ A_normed.T
    B_associations = w_normed @ B_normed.T
    joint_associations = np.concatenate((A_associations,B_associations),axis=-1)

    test_statistic = np.mean(A_associations) - np.mean(B_associations)
    effect_size = test_statistic / np.std(joint_associations,ddof=1)

    """
    midpoint = len(A)
    sample_distribution = np.array([np.random.permutation(joint_associations) for _ in range(permutations)])
    sample_associations = np.mean(sample_distribution[:,:midpoint],axis=1) - np.mean(sample_distribution[:,midpoint:],axis=1)
    p_value = 1 - norm.cdf(test_statistic,np.mean(sample_associations),np.std(sample_associations,ddof=1))
    """

    return effect_size#, p_value


def my_svc(pairs, embedding):

    X = np.zeros((len(pairs)*2, len(embedding[0])))
    for i in range(len(pairs*2)):
        X[i] = embedding[i]

    X=normalize(X,axis=0)
    y = [1] * len(pairs) + [0] * len(pairs)
    clf = LinearSVC(C = 5)
    clf.fit(X, y)
    coef = clf.coef_
    direction = np.reshape(coef/np.linalg.norm(coef), (len(embedding[0]),))
    clf = svm.SVC(kernel='linear', C=1, random_state=42)

    return direction, clf