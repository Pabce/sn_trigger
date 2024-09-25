import logging
import numpy as np
import matplotlib.pyplot as plt

from gui import console


class HitPlotter:
    def __init__(self, config, logging_level=logging.INFO):
        self.config = config
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)


    def plot_3d_clusters(self, cluster, space_clusters, plot_complement_cluster=False):
        # Config reads ----
        detector_type = self.config.get("Detector", "type")
        if detector_type != "VD":
            raise ValueError("VD detector type required for 3D plotting")   
        # Get detector ARAPUCA coordinates
        x_coords = self.config.loaded["x_coords"]
        y_coords = self.config.loaded["y_coords"]
        z_coords = self.config.loaded["z_coords"]
        # -----------------

        self.log.info("Plotting 3D clusters...")
        self.log.info(f"\tCluster shape: {cluster.shape}")
        
        if not isinstance(space_clusters, list):
            space_clusters = [space_clusters]
        
        # If we want to plot the "unclustered" hits, we need to build the complement cluster
        if plot_complement_cluster:
            space_cluster_union = np.concatenate(space_clusters, axis=0)
            self.log.info(f"Space cluster union shape: {space_cluster_union.shape}")
            complement_cluster = []
            for i in range(cluster.shape[0]):
                hit_1 = cluster[i, :]
                for j in range(space_cluster_union.shape[0]):
                    hit_2 = space_cluster_union[j, :]
                    if np.array_equal(hit_1, hit_2):
                        break
                else: # Neat trick: else is executed only if the for loop wasn't broken
                    complement_cluster.append(cluster[i])
            
            complement_cluster = np.array(complement_cluster)
            self.log.info(f"Complement cluster shape: {complement_cluster.shape}")
            #self.log.info(cluster, complement_cluster)

        fig = plt.figure(figsize=(9, 6.5), dpi=100)
        ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
        x_range = [min(x_coords), max(x_coords)]
        y_range = [min(y_coords), max(y_coords)]
        z_range = [min(z_coords), max(z_coords)]

        # Set the limits and aspect ratio for the plot
        ax.set_xlim(y_range)
        ax.set_ylim(z_range)
        ax.set_zlim(x_range)
        ax.set_box_aspect([1, (z_range[1]-z_range[0])/(y_range[1]-y_range[0]), (x_range[1]-x_range[0])/(y_range[1]-y_range[0])])

        ax.set_xlabel("Y (cm)")
        ax.set_ylabel("Z (cm)")
        ax.set_zlabel("X (cm)")

        # Scatter the arapuca coordinates
        ax.scatter(y_coords, z_coords, x_coords, edgecolors='gray', facecolors='none', marker='o', alpha=0.15, s=15)

        # If there are several space clusters, we need a mono color scheme for each of them
        colors = ['blue', 'red', 'orange', 'black',
                'hotpink', 'gray', 'cyan', 'magenta', 'yellow',
                'brown', 'violet', 'purple', 'white', 'pink']

        for i in range(len(space_clusters)):
            space_cluster = space_clusters[i]
            
            self.log.info(f"\t\tSpace cluster shape: {space_cluster.shape}")

            ax.set_title(f"Cluster size: {space_cluster.shape[0]} / {cluster.shape[0]}")

            # Keep track of how many hits each opchannel has
            cluster_opchannels = cluster[:, 2].astype(int)
            opchannel_number = self.config.get("Detector", "optical_channel_number")
            opchannel_hits = np.zeros(opchannel_number)
            for opchannel in cluster_opchannels:
                opchannel_hits[opchannel] += 1
            
            # Plot all ARAPUCA coordinates as empty circles
            ax.scatter(y_coords, z_coords, x_coords, edgecolors='gray', facecolors='none', marker='o', alpha=0.3, s=15, linewidth=0.5)

            # Define a custom colormap of 4 colors to color by hit number in each opchannel
            # hit_number_colors = ['orange', 'red', 'purple', 'black']
            # cmap = plt.cm.colors.ListedColormap(hit_number_colors)
            # scatter = ax.scatter(sy, sz, sx, c=opchannel_hits[space_cluster[:, 2].astype(int)], 
            #                         cmap=cmap, alpha=1, norm=plt.Normalize(vmin=1, vmax=4))
            # cbar = plt.colorbar(scatter, label='Number of hits', ticks=range(1, 5))
            # cbar.set_ticklabels(range(1, 5))
            
            # size of the points should be proportional to the number of hits in each opchannel
            size = 20 * opchannel_hits[space_cluster[:, 2].astype(int)]
            sx, sy, sz = space_cluster[:, 4], space_cluster[:, 5], space_cluster[:, 6]
            ax.scatter(sy, sz, sx, c=colors[i], alpha=1, s=size, zorder=10)

        if plot_complement_cluster:
            if complement_cluster.shape[0] > 0:
                cx, cy, cz = complement_cluster[:, 4], complement_cluster[:, 5], complement_cluster[:, 6]
                ax.scatter(cy, cz, cx, c='green', alpha=1, s=25, zorder=10, marker='^')

        fig.tight_layout()
        plt.show()
