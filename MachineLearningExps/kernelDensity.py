import matplotlib
matplotlib.use("TKagg")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


#motivating KDE: histograms
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

x = make_data(1000)

def show():
    plt.show()
    plt.clf()

hist = plt.hist(x, bins=30, density=True)
show()

density, bins, patches = hist
widths = bins[1:] - bins[:-1]

x = make_data(20)
bins = np.linspace(-5, 10, 10)

x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8]);
show()

from scipy.stats import norm
x_d = np.linspace(-4, 8, 1000)
density = sum(norm(xi).pdf(x_d) for xi in x)

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5]);
show()


#Kernel Density estimation in practice

from sklearn.neighbors import KernelDensity

#instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x[:, None])

#score_samples returns the log of pobability density
logprob = kde.score_samples(x_d[:, None])

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
show()

##selecting bandwith via cross-validation
#we can use gridsearchCV to optimize the bandwidth

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

bandwidths = 10 ** np.linspace(-1,1,100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                     {'bandwidth':bandwidths},
                     cv=LeaveOneOut())
grid.fit(x[:, None])
#whats the best bandwidth?
print(grid.best_params_) #1.123...

#Example: KDE on a sphere
from sklearn.datasets import fetch_species_distributions
data = fetch_species_distributions()

# Get matrices/arrays of species IDs and locations
latlon = np.vstack([data.train['dd lat'],
                    data.train['dd long']]).T
species = np.array([d.decode('ascii').startswith('micro')
                    for d in data.train['species']], dtype='int')

from mpl_toolkits.basemap import Basemap
#broken
from sklearn.datasets.species_distributions import construct_grids

xgrid, ygrid = construct_grids(data)

# plot coastlines with basemap
m = Basemap(projection='cyl', resolution='c',
            llcrnrlat=ygrid.min(), urcrnrlat=ygrid.max(),
            llcrnrlon=xgrid.min(), urcrnrlon=xgrid.max())
m.drawmapboundary(fill_color='#DDEEFF')
m.fillcontinents(color='#FFEEDD')
m.drawcoastlines(color='gray', zorder=2)
m.drawcountries(color='gray', zorder=2)

# plot locations
m.scatter(latlon[:, 1], latlon[:, 0], zorder=3,
          c=species, cmap='rainbow', latlon=True)

show()
