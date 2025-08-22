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
        fig, ax = plt.subplots(figsize=(8, 5))

        for tree, features, targets in zip(self.trees, self.features, self.targets):
            fpr, tpr, thresholds = roc_curve(targets, tree.predict_proba(features)[:,1])
            # Area under the curve
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, lw=2, label='ROC curve (area = {:.3f})'.format(roc_auc))
            
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc="lower right")
        fig.tight_layout()
        plt.show()

        return fig, ax
    
    def plot_efficiency_vs_threshold(self):
        plot_styles = ["-", "--", "-.", ":"]
        
        fig, ax = plt.subplots(figsize=(8, 5))

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
            ax.plot(threshold, sn_efficiency, label=f"SN efficiency, {tree_name}", color='red', linestyle=plot_styles[i])
            ax.plot(threshold, bg_efficiency, label=f"BG efficiency, {tree_name}", color='blue', linestyle=plot_styles[i])

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Efficiency")
        ax.legend()
        fig.tight_layout()
        plt.show()
        return fig, ax
    
    def plot_score_histograms(self, colors=None):
        figs_axes = []
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            predictions_prob = tree.predict_proba(features)[:,1]

            bg_predictions_prob = predictions_prob[targets == 0]
            sn_predictions_prob = predictions_prob[targets == 1]

            bg_hit_multiplicities = features[:, 0][targets == 0].astype('int')
            sn_hit_multiplicities = features[:, 0][targets == 1].astype('int')
            
            # Plot the BDT score histograms
            fig10, ax10 = plt.subplots()
            ax10.hist(bg_predictions_prob, bins=np.arange(0, 1, 0.04), alpha=0.8, density=True, label="BG")
            ax10.hist(sn_predictions_prob, bins=np.arange(0, 1, 0.04), alpha=0.5, density=True, label="Signal")
            ax10.set_xlabel("Score")
            ax10.set_ylabel("Density")
            ax10.set_yscale("log")
            #ax1.set_title(f"BDT score histogram for {tree_name}")
            ax10.legend()
            fig10.tight_layout()
            figs_axes.append((fig10, ax10))

            # Now plot the bg and sn predictions separately
            if colors is None:
                colors = ["red", "blue"]
            fig11, ax11 = plt.subplots()
            ax11.hist(bg_predictions_prob, bins=np.arange(0, 1, 0.04), alpha=1.0, 
                      density=True, color=colors[0])
            ax11.set_xlabel("Score")
            ax11.set_ylabel("Density")
            ax11.set_yscale("log")
            fig11.tight_layout()
            figs_axes.append((fig11, ax11))

            fig12, ax12 = plt.subplots()
            ax12.hist(sn_predictions_prob, bins=np.arange(0, 1, 0.04), alpha=1.0, 
                      density=True, color=colors[1])
            ax12.set_xlabel("Score")
            ax12.set_ylabel("Density")
            ax12.set_yscale("log")
            fig12.tight_layout()
            figs_axes.append((fig12, ax12))

            # Plot the hit multiplicity histograms
            if 'hit_multiplicity' in feature_names:
                # Get the max and min hit multiplicities
                max_hit_multiplicity = np.max(np.concatenate((bg_hit_multiplicities, sn_hit_multiplicities)))
                min_hit_multiplicity = np.min(np.concatenate((bg_hit_multiplicities, sn_hit_multiplicities)))

                fig2, ax2 = plt.subplots()
                ax2.hist(bg_hit_multiplicities, bins=np.arange(min_hit_multiplicity, max_hit_multiplicity + 1, 1), alpha=0.8, density=True, label="BG")
                ax2.hist(sn_hit_multiplicities, bins=np.arange(min_hit_multiplicity, max_hit_multiplicity + 1, 1), alpha=0.5, density=True, label="Signal")
                ax2.set_xlabel("Hit Multiplicity")
                ax2.set_ylabel("Density")
                ax2.set_title(f"Hit multiplicity for {tree_name}")
                ax2.legend()
                fig2.tight_layout()
                figs_axes.append((fig2, ax2))
        
        plt.show()
        return figs_axes

    def plot_permutation_importance(self):
        figs_axes = []
        # Plot the permutation importance
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            perm_importances = inspection.permutation_importance(tree, features[:], targets[:], n_repeats=20)

            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(perm_importances.importances_mean))]

            sort_labels = np.array(feature_names)[np.argsort(perm_importances.importances_mean)]
            sort_importance = np.sort(perm_importances.importances_mean)
            
            fig, ax = plt.subplots()
            ax.barh(sort_labels, sort_importance)
            ax.set_xlabel(f"Permutation importance for {tree_name}")
            ax.set_title(f"Permutation importance for {tree_name}")
            plt.xticks(rotation=90)
            fig.tight_layout()
            figs_axes.append((fig, ax))

        plt.show()
        return figs_axes

    def plot_shapley_summary(self, approx=False, n_features=10, tick_font_size=12):
        figs_axes = []
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            # Shuffle the features and targets just in case
            p = np.random.permutation(len(features))
            features = features[p]
            targets = targets[p]
 
            explainer = shap.TreeExplainer(tree, approximate=approx, feature_names=feature_names)
            shap_values = explainer(features)

            # Plot the SHAP summary plot
            shap.summary_plot(
                shap_values,
                features,
                show=False,
                max_display=n_features,
                feature_names=feature_names,
                plot_size=(10, 6)
            )

            # The figure/axes are now the current ones in matplotlib.
            fig1 = plt.gcf()

            # The summary plot axis is the one that owns the x-label produced by
            # SHAP ("SHAP value (impact on model output)").  Locate it to be
            # robust against the additional color-bar axis.
            summary_axes = [ax for ax in fig1.axes if ax.get_xlabel() != ""]

            if len(summary_axes):
                summary_ax = summary_axes[0]
            else:
                # Fallback: assume first axis.
                summary_ax = fig1.axes[0]

            # Iterate over *all* axes (main axis + colour-bar) to guarantee
            # uniform font sizing.
            for ax in fig1.axes:
                ax.tick_params(axis="both", which="both", labelsize=tick_font_size)

                for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                    lbl.set_fontsize(tick_font_size)

                # Update axis labels if present
                if ax.get_xlabel():
                    ax.set_xlabel(ax.get_xlabel(), fontsize=tick_font_size)
                if ax.get_ylabel():
                    ax.set_ylabel(ax.get_ylabel(), fontsize=tick_font_size)

            fig1.tight_layout()
            figs_axes.append((fig1, summary_ax))

            # Plot the absolute importances of each feature
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            shap.plots.bar(
                shap_values,
                max_display=n_features+1,
            )
            #ax2.set_title(f"SHAP feature importance for {tree_name}")
            ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)
            ax2.set_xlabel("mean(|SHAP value|)", fontsize=tick_font_size)
            fig2.tight_layout()
            figs_axes.append((fig2, ax2))

        plt.show()
        return figs_axes

    def plot_feature_correlation(self):
        figs_axes = []
        # Plot the normalised feature correlation matrix, with colorbar between -1 and 1
        for tree, features, targets, tree_name, feature_names in zip(self.trees, self.features, self.targets, self.tree_names, self.feature_names_list):
            fig, ax = plt.subplots()
            cax = ax.imshow(np.corrcoef(features.T), aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Feature correlation for {tree_name}")
            figs_axes.append((fig, ax))
        
        plt.show()
        return figs_axes


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