import matplotlib
matplotlib.use("TKagg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

#manifold learning
#unsupervised estimators that descibe datasets as low dim manifolds
#embeded in high dim space


#Manifold learning "HELLO"

#generate some data
def make_hello(N=1000, rseed=42):
    """ Make a plot with "HELLO" text; save as PNG"""
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold', size=85)
    fig.savefig('hello.png')
    plt.close(fig)
    
    # Open this PNG and draw random points from it
    from matplotlib.image import imread
    data = imread('hello.png')[::-1, :, 0].T
    rng = np.random.RandomState(rseed)
    X = rng.rand(4 * N, 2)
    i, j = (X * data.shape).astype(int).T
    mask = (data[i, j] < 1)
    X = X[mask]
    X[:, 0] *= (data.shape[0] / data.shape[1])
    X = X[:N]
    return X[np.argsort(X[:, 0])]

def show():
    plt.show()
    plt.clf()

#try it out
X = make_hello(1000)
colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
plt.scatter(X[:, 0], X[:, 1], **colorize)
plt.axis('equal');
show()


#Multidimensional Scaling (MDS)

def rotate(X, angle):
    """Rotate a point"""
    theta = np.deg2rad(angle)
    R = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]]
    return np.dot(X, R)

#rotate our data and shift
X2 = rotate(X, 20) + 5
plt.scatter(X2[:, 0], X2[:, 1], **colorize)
plt.axis('equal');
show()
#the x,y cords dont matter, the distance between others relative is
#we can use pairwise_distances to look at this data
from sklearn.metrics import pairwise_distances
D = pairwise_distances(X) #(1000,1000)

#lets plot this
plt.imshow(D, zorder=2, cmap='Blues', interpolation='nearest')
plt.colorbar();
show()

#we can do the same with out rotated matrix and find there are the same
D2 = pairwise_distances(X2)
print(np.allclose(D, D2)) #true

#finding the distances is easy but going back to points is not

from sklearn.manifold import MDS
model = MDS(n_components=2, dissimilarity='precomputed', random_state=1)
out = model.fit_transform(D)
plt.scatter(out[:,0], out[:, 1], **colorize)
plt.axis('equal')
show()

#we recovered the x, y chords somewhat

#MDS as Manifold learning
#we cand find distances between any dimension

#lets build some data
def random_projection(X, dimension=3, rseed=42):
    assert dimension >= X.shape[1]
    rng = np.random.RandomState(rseed)
    C = rng.randn(dimension, dimension)
    e, V = np.linalg.eigh(np.dot(C, C.T))
    return np.dot(X, V[:X.shape[1]])
    
X3 = random_projection(X, 3)
X3.shape

#visualize it
from mpl_toolkits import mplot3d
ax = plt.axes(projection='3d')
ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2],
             **colorize)
ax.view_init(azim=70, elev=50)
show()

#we can feeed this to an MDS and find the optimal 2dim embedding

model = MDS(n_components=2, random_state=1)
out3 = model.fit_transform(X3)
plt.scatter(out3[:, 0], out3[:, 1], **colorize)
plt.axis('equal');
show()

#Nonlinear Embeddings: where MDS fails

#transform our data nonlinearly
def make_hello_s_curve(X):
    t = (X[:, 0] - 2) * 0.75 * np.pi
    x = np.sin(t)
    y = X[:, 1]
    z = np.sign(t) * (np.cos(t) - 1)
    return np.vstack((x, y, z)).T

XS = make_hello_s_curve(X)

ax = plt.axes(projection='3d')
ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2],
             **colorize);
show()
#simple MDS will fail to unwrap this

model = MDS(n_components=2, random_state=2)
outS = model.fit_transform(XS)
plt.scatter(outS[:, 0], outS[:, 1], **colorize)
plt.axis('equal');
show()

#we lost the y axis instead of unwrapping

#Nonlinear Manifolds: locally Linear Embedding
#preserve only the distances of nearbny
#we can use this (LLE) to unwrap out data

from sklearn.manifold import LocallyLinearEmbedding as LLE
model = LLE(n_neighbors=100, n_components=2,
            eigen_solver='dense')
out = model.fit_transform(XS)

fig, ax = plt.subplots()
ax.scatter(out[:, 0], out[:, 1], **colorize)
ax.set_ylim(0.15, -0.15);
show() #pretty close to the original

print("isomaps")
#example: isomaps on faces

#get the data
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=30)
#(2370, 2914) shape

#visualize the data
fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='gray')

#wwe can use a pca to find out how many linear features we need

from sklearn.decomposition import PCA as RandomizedPCA
model = RandomizedPCA(100).fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cumulative variance')
show()
#we need 100 components to preserve 90% of the variance
#LLE/ isomaps can be useful here

from sklearn.manifold import Isomap
odel = Isomap(n_components=2)
proj = model.fit_transform(faces.data) #(2370, 2) shape

#lets plot the thumbs along these two dimensions
from matplotlib import offsetbox

def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)

            
fig, ax = plt.subplots(figsize=(10, 10))
plot_components(faces.data,
                model=Isomap(n_components=2),
                images=faces.images[:, ::2, ::2])
show()


print("#Example: Visualizing Structure in digits")
#wacked
