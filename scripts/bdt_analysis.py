import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn import inspection

import config as cf
import saver as sv
import classifier as cl
import gui
from gui import console
import shap

class BDTAnalysis:
    def __init__(self, config, trees, features, targets, tree_names=None, feature_names_list=None):
        # If trees is a single tree, convert it to a list
        if not isinstance(trees, list):
            trees = [trees]
            features = [features]
            targets = [targets]
            tree_names = [tree_names]
            feature_names_list = [feature_names_list]
        
        for i, name in enumerate(tree_names):
            if name is None:
                tree_names[i] = f"Tree {i}"

        # TODO: figure out if you actually need the config in the class
        self.config = config
        self.trees = trees
        self.features = features
        self.targets = targets
        self.feature_names_list = feature_names_list
        self.tree_names = tree_names



    # @classmethod
    # def data_from_files(cls, config, tree_path, features_and_targets_path):
    #     # Get the tree from the pickled file
    #     with open(tree_path, "rb") as file:
    #         tree = pickle.load(file)
        
    #     # Get the features and targets from the pickled file
    #     with open(features_and_targets_path, "rb") as file:
    #         features, targets = pickle.load(file)
        
    #     return cls(config, tree, features, targets)
    

    def print_tree_info(self):
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            console.log(f"Tree: {tree_name}")
            console.log(f"Feature names: {feature_names}")
            console.log(f"Features shape: {features.shape}")
            console.log(f"Targets shape: {targets.shape}")
        
            # Print the tree parameters
            console.log("[bold green]Tree parameters:")
            console.log(f"{tree.get_params()}")
            # Print a new line
            console.log("")


    def plot_roc_curve(self):
        # False positive rate, true positive rate, thresholds
        plt.figure()

        for tree, features, targets in zip(self.trees, self.features, self.targets):
            fpr, tpr, thresholds = roc_curve(targets, tree.predict_proba(features)[:,1])
            # Area under the curve
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc))
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
    
    def plot_efficiency_vs_threshold(self):
        plot_styles = ["-", "--", "-.", ":"]
        
        plt.figure()

        for i, (tree, features, targets, tree_name) in enumerate(zip(self.trees, self.features, self.targets, self.tree_names)):
            # False positive rate, true positive rate, thresholds
            fpr, tpr, thresholds = roc_curve(targets, tree.predict_proba(features)[:,1])
            # SN efficiency is the true positive rate
            sn_efficiency = tpr
            # BG efficiency is the true negative rate
            bg_efficiency = 1 - fpr
            # Threshold is the threshold value
            threshold = thresholds

            #console.log(f"Threshold: {threshold}")
            plt.plot(threshold, sn_efficiency, label=f"SN efficiency, {tree_name}", color='red', linestyle=plot_styles[i])
            plt.plot(threshold, bg_efficiency, label=f"BG efficiency, {tree_name}", color='blue', linestyle=plot_styles[i])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Threshold")
        plt.ylabel("Efficiency")
        plt.legend()
        plt.show()
    
    def plot_score_histograms(self):

        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            predictions_prob = tree.predict_proba(features)[:,1]

            bg_predictions_prob = predictions_prob[targets == 0]
            sn_predictions_prob = predictions_prob[targets == 1]

            bg_hit_multiplicities = features[:, 0][targets == 0].astype('int')
            sn_hit_multiplicities = features[:, 0][targets == 1].astype('int')
            
            # Plot the BDT score histograms
            plt.figure()
            plt.hist(bg_predictions_prob, bins=np.arange(0, 1, 0.04), alpha=0.8, density=True, label="BG")
            plt.hist(sn_predictions_prob, bins=np.arange(0, 1, 0.04), alpha=0.5, density=True, label="Signal")
            plt.xlabel("Score")
            plt.ylabel("Density")
            plt.yscale("log")
            plt.title(f"BDT score histogram for {tree_name}")
            plt.legend()

            # Plot the hit multiplicity histograms
            if 'hit_multiplicity' in feature_names:
                # Get the max and min hit multiplicities
                max_hit_multiplicity = np.max(np.concatenate((bg_hit_multiplicities, sn_hit_multiplicities)))
                min_hit_multiplicity = np.min(np.concatenate((bg_hit_multiplicities, sn_hit_multiplicities)))

                plt.figure()
                plt.hist(bg_hit_multiplicities, bins=np.arange(min_hit_multiplicity, max_hit_multiplicity + 1, 1), alpha=0.8, density=True, label="BG")
                plt.hist(sn_hit_multiplicities, bins=np.arange(min_hit_multiplicity, max_hit_multiplicity + 1, 1), alpha=0.5, density=True, label="Signal")
                plt.xlabel("Hit Multiplicity")
                plt.ylabel("Density")
                plt.title(f"Hit multiplicity for {tree_name}")
                plt.legend()
        
        plt.show()
    
    def plot_permutation_importance(self):
        # Plot the permutation importance
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            perm_importances = inspection.permutation_importance(tree, features[:], targets[:], n_repeats=20)

            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(perm_importances.importances_mean))]

            sort_labels = np.array(feature_names)[np.argsort(perm_importances.importances_mean)]
            sort_importance = np.sort(perm_importances.importances_mean)
            
            plt.figure()
            plt.barh(sort_labels, sort_importance)
            plt.xlabel(f"Permutation importance for {tree_name}")
            plt.xticks(rotation=90)
            plt.tight_layout()

        plt.show()
    
    def plot_shapley_summary(self):
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            # Shuffle the features and targets just in case
            p = np.random.permutation(len(features))
            features = features[p]
            targets = targets[p]

            explainer = shap.TreeExplainer(tree, feature_names=feature_names)
            shap_values = explainer(features)

            # Plot the SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, features)
            plt.tight_layout()

            # Plot the absolute importances of each feature
            plt.figure()
            shap.plots.bar(shap_values)
            plt.title(f"SHAP feature importance for {tree_name}")
            plt.tight_layout()

        plt.show()
    
    def plot_feature_correlation(self):
        # Plot the normalised feature correlation matrix, with colorbar between -1 and 1
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            plt.figure()
            plt.imshow(np.corrcoef(features.T), aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
        
        plt.show()


if __name__ == "__main__":
    # Read the save directory from the config
    config_path = '../configs/default_config_vd.yaml'
    config = cf.Configurator(config_path=config_path, 
                            default_config_path=config_path)
    save_dir = config.get("IO", "pickle_save_dir")

    # Load the BDT and the features/targets
    tree = sv.DataWithConfig.load_data_from_file(save_dir + "hist_boosted_tree.pkl").data
    features, targets = sv.DataWithConfig.load_data_from_file(
                            save_dir + "test_features_targets.pkl"
                        ).data

    # Create a BDT analyzer
    bdt_ana = BDTAnalysis(config, tree, features, targets)

    bdt_ana.plot_roc_curve()
    bdt_ana.plot_efficiency_vs_threshold()
    bdt_ana.plot_score_histograms()
    bdt_ana.plot_permutation_importance()
    bdt_ana.plot_shapley_summary()