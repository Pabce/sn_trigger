'''
classifier.py

Contains the functions that train and apply the BDT.
'''


from locale import normalize
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn import tree
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import zero_one_loss, confusion_matrix
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform, randint, expon



def build_decision_tree(features, targets):

    train_features, test_features, train_targets, test_targets = train_test_split(features, targets,
                                                                test_size=0.1, random_state=212)
    
    decision_tree = tree.DecisionTreeClassifier(max_depth=2, random_state=988).fit(train_features, train_targets)

    # Bias calculation with 0-1 loss
    predictions = decision_tree.predict(train_features)
    predictions_test = decision_tree.predict(test_features)
    print("0-1 loss: " + str(zero_one_loss(predictions, train_targets)))
    print("0-1 loss TEST: " + str(zero_one_loss(predictions_test, test_targets)))

    return decision_tree


def tree_filter(tree, clusters, features, threshold=0.5):
    predictions_prob = tree.predict_proba(features)[:, 1]

    # print(features.shape, "features shape")
    # print(len(clusters), "clusters")
    
    new_clusters = []
    filter = []

    #print("Clusters to filter", len(clusters))
    for i, cluster in enumerate(clusters):
        if predictions_prob[i] >= threshold:
            new_clusters.append(cluster)
            filter.append(i)
    
    new_features = features[filter, :]

    # Also return the hit multp. for format reasons
    return new_clusters, new_features[:, -1].astype('int'), predictions_prob


def gradient_boosted_tree(features, targets, n_estimators=200, max_depth=2, threshold_scan=False):
    #n_estimators = 200

    train_features, test_features, train_targets, test_targets =\
                                    train_test_split(features, targets, test_size=0.01, shuffle=True)
    
    # Shuffle the features and targets
    # p = np.random.permutation(len(features))
    # split = int(len(features) * 0.9)
    # train_features = features[p, :][:split]
    # train_targets = targets[p][:split]
    
    # print("Train features shape", train_features.shape)
    # print(train_features[:10, -1])
    # print(train_targets[:10])

    boosted_tree = GradientBoostingClassifier(max_depth=max_depth, n_estimators=n_estimators,
                    loss='deviance', learning_rate=0.1, subsample=1).fit(train_features, train_targets)

    #svm = SVC(gamma='scale', C=0.1, kernel='sigmoid').fit(train_features, train_targets)
    
    if threshold_scan:
        # Plot the efficiency and purity
        predictions_test_prob = boosted_tree.predict_proba(test_features)
        threshold = np.linspace(0, 1, 100)
        efficiencies = []
        bg_efficiencies = []

        for x in threshold:
            new_predictions = np.zeros((predictions_test_prob.shape[0]))
            new_predictions[predictions_test_prob[:,1] > x] = 1 
            conf = confusion_matrix(test_targets, new_predictions)

            efficiencies.append(conf[1, 1] / (conf[1, 1] + conf[1, 0]))
            bg_efficiencies.append(conf[0, 0] / (conf[0, 0] + conf[0, 1]))
        
        efficiencies = np.array(efficiencies)
        bg_efficiencies = np.array(bg_efficiencies)

        plt.figure()
        plt.plot(threshold, efficiencies, label="SN efficiency", color="red")
        plt.plot(threshold, bg_efficiencies, label="BG efficiency", color="blue")
        plt.xlabel("Classifier cutoff")
        plt.legend()

        plt.show()
    

    return boosted_tree, test_features, test_targets, train_features, train_targets


def hist_gradient_boosted_tree(features, targets, n_estimators=200, max_depth=2, optimize_hyperparameters=False):
    train_features, test_features, train_targets, test_targets =\
                                    train_test_split(features, targets, test_size=0.01, shuffle=True)
    

    # Optimize hyperparameters
    if optimize_hyperparameters:
        base_estimator = HistGradientBoostingClassifier()
        param_grid = {'learning_rate': loguniform(1e-4, 2e-1),
                    'max_iter': randint(50, 2000),
                    'max_leaf_nodes': randint(16, 64),
                    'min_samples_leaf': randint(5, 25),
                    'max_bins': randint(16, 255),
                    'l2_regularization': loguniform(1e-10, 1e-1),
                    'class_weight': ['balanced', None],
                    }
        
        search = HalvingRandomSearchCV(base_estimator, param_grid, cv=3, verbose=1, n_jobs=-1).fit(train_features, train_targets)
        print(search.best_params_)
                  
        hist_boosted_tree = search.best_estimator_
    else:
        hist_boosted_tree = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=n_estimators, class_weight='balanced').fit(train_features, train_targets)

    return hist_boosted_tree, test_features, test_targets, train_features, train_targets
    




if __name__ == "__main__":
    features, targets = pickle.load(open("../saved_pickles/classifier/points", "rb"))
    # features = features[0:10000, :]
    # targets = targets[0:10000]
    print(features[targets==0].shape, features[targets==1].shape)

    # Normalize dataset (not needed for BDT but in case you use others)
    # for i in range(features.shape[1]):
    #     features[:, i] = (features[:, i] - np.mean(features[:, i]))/np.std(features[:, i])

    plt.scatter(features[targets == 0, 2], features[targets == 0, -11], alpha=1, s=2,
                        color='b', marker="o", label="Background")
    plt.scatter(features[targets == 1, 2], features[targets == 1, -11], alpha=1, s=3, 
                        color='r', marker="v", label="Supernova")
    plt.xlabel(r'Time duration ($\mu$s)', fontsize=17)
    plt.ylabel("z coordinate extension (cm)", fontsize=17)
    plt.legend()

    # plt.figure()
    # plt.hist(features[:,-4])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(features[targets == 1, -6], features[targets == 1, -1], features[targets == 1, -5])
    ax.scatter(features[targets == 0, -6], features[targets == 0, -1], features[targets == 0, -5])
    plt.legend()
    plt.show()

    #build_decision_tree(features, targets)


    gradient_boosted_tree(features[:,:], targets)

