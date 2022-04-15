import torch
from sklearn.manifold import Isomap


list_of_dist = []


m = torch.cdist(a, a, p=2.0)
q = m.cpu().detach().numpy()
embed3 = Isomap(
    n_neighbors=5,  # default=5, algorithm finds local structures based on the nearest neighbors
    n_components=3,  # number of dimensions
    eigen_solver='auto',  # {‘auto’, ‘arpack’, ‘dense’}, default=’auto’
    tol=0,  # default=0, Convergence tolerance passed to arpack or lobpcg. not used if eigen_solver == ‘dense’.
    max_iter=None,
    # default=None, Maximum number of iterations for the arpack solver. not used if eigen_solver == ‘dense’.
    path_method='auto',  # {‘auto’, ‘FW’, ‘D’}, default=’auto’, Method to use in finding shortest path.
    neighbors_algorithm='auto',  # neighbors_algorithm{‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, default=’auto’
    n_jobs=-1,  # n_jobsint or None, default=None, The number of parallel jobs to run. -1 means using all processors
    metric='minkowski',  # string, or callable, default=”minkowski”
    p=2,
    # default=2, Parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
    metric_params=None  # default=None, Additional keyword arguments for the metric function.
)
s = embed3.fit_transform(q)
s1 = torch.from_numpy(s)
dist = torch.cdist(s1, s1, p=2.0)

