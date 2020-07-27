import matplotlib
matplotlib.use("TKagg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def show():
    plt.show()
    plt.clf()

#unsupervised estimator, principle component analysis (PCA)

#Intro to PAC

#lets generate some data
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
plt.scatter(X[:, 0], X[:, 1])
plt.axis('equal');
show()

#we want to learn the relationship between x and y, not predict values

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_)
print(pca.explained_variance_)

#lets visualize these numbers as vectors to better understand their meaning
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');
show()

#PCA as dimensionality reduction

pca= PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("original shape:   ", X.shape)    #(200,2)
print("transformed shape:", X_pca.shape)#(200,1)

#lets inverse transform and then plot with the orginial data
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal');
show()


#PCA for Visualization: hand-written digits
from sklearn.datasets import load_digits
digits = load_digits() #(1797,64)

#lets reduce this down to 2 dimensions
pca = PCA(2) #2 dimensional projection
projected = pca.fit_transform(digits.data)
#now we can plot the 2 principle components
plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar();
show()

#choosing the number of components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
show() #our 2 componets loose a lot of the data, need 20 some for 90%

#PCA as noise filtering

def plot_digits(data):
    fig, axes = plt.subplots(4, 10, figsize=(10, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8),
                  cmap='binary', interpolation='nearest',
                  clim=(0, 16))
plot_digits(digits.data)
show()

#add noise
np.random.seed(42)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)
show()

#lets filter
pca = PCA(0.50).fit(noisy) #12 components

#reconstruct with inverse transform
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)
show()

#Example Eigenfaces

#get out face data
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

#we will use RandomizedPCA since this is a large dataset
#we will reduce from near 3000 to 150 components

from sklearn.decomposition import PCA as RandomizedPCA
pca = RandomizedPCA(150)
pca.fit(faces.data)

fig, axes = plt.subplots(3, 8, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

show()
#lets check the cumulative variance
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
#150 turns out to be around 90% of variance
#lets compare to the full data

# Compute the components and projected faces
pca = RandomizedPCA(150).fit(faces.data)
components = pca.transform(faces.data)       #filter
projected = pca.inverse_transform(components)#rebuild

# Plot the results
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),
                       subplot_kw={'xticks':[], 'yticks':[]},
                       gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i in range(10):
    ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
    ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
    
ax[0, 0].set_ylabel('full-dim\ninput')
ax[1, 0].set_ylabel('150-dim\nreconstruction');
show()
