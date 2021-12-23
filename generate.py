# required modules:
import numpy as np
from random import randrange

"""
2021.12.22
"""


def _euclidean_normalize(datablock: np.ndarray):
    """Normalize vectors in a data block using Euclidean distances. Each row is a point in space with <number of columns> dimensions."""
    return datablock / (np.sqrt(np.sum(np.square(datablock), axis=1, keepdims=1)))



def clusters(number_of_centers: int = 2, number_of_datapoints: (int or tuple) = 50, limit: (int or tuple) = None, **options):
    """generates clusters, outputs data points and their cluster number. the generation will be centered at 0"""
    
    # number_of_centers             INT                 number of cluster centers
    # number_of_datapoints          TUPLE/INT           number of data points for each cluster
    # limit                         TUPLE/INT           limit of data points. i.e. data points will not go beyond limit. if None, no limit.
    #                                                   INT -> max value for all dimensions. min value is negative of max value.
    #                                                   TUPLE -> max value for the dimensions. min value is ngative of max value for each dimension.
    # **options: see belowrandomize
    shape=options.get("shape", "point")                     # shape of centers: point, curve/line/linear/nonlinear
    if shape not in ['point', 'curve', 'linear', 'nonlinear', 'line']:
        raise ValueError('Unknown cluster center shape. Options: "point", "curve". If using "curve", provide a parametric function using "parametric" keyword.')
    dimensions=int(options.get("dimensions", 2))            # dimensions of data points.
    radii=options.get("radii", 1)                           # radii of clusters, i.e. how far a data points deviate from the cluster center. Note that cluster centers can be lines/curves.
    shuffle=options.get("shuffle", True)                    # shuffle data points.
    # OPTION FOR POINT CENTERS:
    distance=options.get("distance", 1)                     # mean distance of cluster centers from origin.
    # OPTION FOR 2-CLASS POINT CLUSTERS
    opposite=options.get("opposite", False)                  # opposite: whether the clusters are opposite of each other relative to the origin.
    parametric=options.get("parametric")

    if shape in ['linear', 'line', 'curve', 'nonlinear']:
        assert parametric!=None, 'Cluster centers shape "%s" requires a parametric function.' % (shape)
    

    ###-------------------------------------------------
    # more input error checking, and format conversions
    assert dimensions>0 and number_of_centers>0 and np.all(number_of_datapoints)>0, "Dimensions, number of cluster centers and number of data points must be greater than 0."

    # convert radii and number_of_datapoints to a list of radii that can be used to numpy-multiply with randomly generated values.
    if type(number_of_datapoints)==int:
        assert number_of_datapoints>0, "Number of data points must be positive for each cluster center."
        number_of_datapoints=(number_of_datapoints,)*number_of_centers
    elif type(number_of_datapoints)==tuple:
        assert np.all(number_of_datapoints)>0, "Number of data points must be positive for each cluster center."
        assert len(number_of_datapoints)==number_of_centers, "Number of data points specified must be one or the same as the number of dimensions."
        if len(number_of_datapoints)==1 and number_of_centers!=1:
            number_of_datapoints=(number_of_datapoints,)*number_of_centers

    if type(radii)==tuple:
        assert len(radii)==1 or len(radii)==number_of_centers, "Number of cluster radii specified must be one or the same as the number of dimensions."
        if len(radii)==1:
            radii=(radii,)*number_of_centers
    elif type(radii)==int or type(radii)==float:
        radii=(radii,)*number_of_centers
    radii=np.asarray(radii)

    if opposite==None or opposite==False:
        # if opposite option (for 2-class point clusters) is not used, generate a normal distribution of distances.
        # The distances are between point cluster centers and the origin.
        # The mean is the specified distance, the standard deviation increases with distance to ensure a good range.
        sign=np.sign(np.random.random(size=number_of_centers)-0.5)
        distances=sign*np.random.normal(loc=distance, scale=(distance/3)-0.2*distance, size=number_of_centers)
    else:
        # if opposite option is true: use "distance" instead of "distances".
        distances=(distance, -distance) # for prints only
        pass


    ###--------------------------------------------------

    clusters_of_datapoints=[]       # list of cluster clouds, to be concatenated into a numpy array
    cluster_indices=[]              # list of cluster indices that the rows in the list above correspond to

    for cluster in range(number_of_centers):

        cluster_index=np.ones((number_of_datapoints[cluster], 1))*cluster # gets the data's cluster. This is used to build the reference categorization (supervised learning).
        cloud=_euclidean_normalize(np.tan(1.5708*np.random.random((number_of_datapoints[cluster], dimensions))-0.5))*(np.random.random((number_of_datapoints[cluster], dimensions))*2-1) # starts to generate the actual data.
        #    radius modifier | ^^ normalizes vector relative to center |^^ generates the random data. These decides the outer limit | ^^ adjusts location so the data points fill the area.

        if shape=="point":
            cloud=cloud*2*radii[cluster]-radii[cluster]
            if opposite:
                if cluster==0:
                    center_to=np.multiply(_euclidean_normalize(np.random.random((1, dimensions))), np.square(distance))
                elif cluster==1:
                    center_to=np.multiply(_euclidean_normalize(np.random.random((1, dimensions))), -np.square(distance))
            else:
                center_to=np.multiply(_euclidean_normalize(np.tan(1.5708*np.random.random((1, dimensions))-0.5)), np.sign(distances[cluster])*np.square(distances[cluster]))
            
        elif shape in ['linear', 'line', 'curve', 'nonlinear']:
            
            cloud=radii[cluster]*cloud
            func=parametric[cluster]
            dims=func(np.random.random((number_of_datapoints[cluster], 1)))
            center_to=np.concatenate(dims, axis=1)

            
        else: pass # unknown shape option has been dealt with in input checking.

        cloud=cloud+center_to
        clusters_of_datapoints.append(cloud)
        cluster_indices.append(cluster_index)
        

    generated_data=np.concatenate(clusters_of_datapoints, axis=0)
    cluster_indices=np.concatenate(cluster_indices, axis=0)
    #print(generated_data, cluster_indices)


    summary=np.concatenate((generated_data, cluster_indices), axis=1)
    if shuffle==True:
        np.random.shuffle(summary)
    
    generated_data=summary[:, 0:-1]
    cluster_indices=summary[:, -1:]
    
    return (generated_data, cluster_indices)

