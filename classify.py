import json
import matplotlib.pyplot as plt
import numpy as np
import random
def plot(z):
    plt.plot(z, 'bo', z, 'k')
    plt.show()
def plot_d(x, y):
    plt.subplot(211)
    plt.plot(x, 'bo', x, 'k')
    plt.subplot(212)
    plt.plot(y, 'bo', y, 'k')
    plt.show()
def diff(d):
    return [d[i+1]-d[i] for i in range(len(d)-1)]
def dist(v1,v2):
    return sum(abs(v1[i]-v2[i]) for i in range(len(v1)))
def dtw_dist(v1,v2): # dtw: dynamic time warping
    n, m = len(v1), len(v2)
    D = np.full((n,m),float('inf'))
    D[0][0] = 0
    w = 10 # gives 0.943 score, w = 15 gives 0.949
    for i in range(1,n):
        for j in range(max(1,i-w),min(m,i+w)):
            cost = (v1[i-1]-v2[j-1])**2
            D[i][j] = cost + min([D[i-1][j], D[i][j-1], D[i-1][j-1]])
    d = D[n-1][m-1]**0.5
    return d
def truncate_z(s):
    # clip a single height series at the first min
    d = diff(s)
    c = 0
    i = 1
    while c == 0:
        z = d[i]*d[i+1]
        if z <= 0.0 and d[i-1] <= 0 and sum(d[i:i+5]) > 0:
            min_v = s[i]
            min_i = i
            for j in range(i+1,i+4):
                if s[j] < min_v:
                    min_v = s[j]
                    min_i = j
            c = min_i
        i += 1
    return s[c:270+c]
def truncate_a(s):
    # clip a single angle series at the first big min
    # filter out smaller mins
    d = diff(s)
    crit_points = []
    i = 1
    while len(crit_points) < 12:
        z = d[i]*d[i+1]
        if z <= 0.0 and d[i] <= 0 and sum(d[i+1:i+3]) > 0:
            min_v = s[i]
            min_i = i
            for j in range(i+1,i+3):
                if s[j] < min_v:
                    min_v = s[j]
                    min_i = j
            if min_i not in crit_points:
                crit_points.append(min_i)
        i += 1
    min_i = min(crit_points, key=lambda i:s[i])
    c = min(crit_points,key=lambda i: i + 1000*abs(s[i]-s[min_i]))
    return s[c:240+c]
def smooth(s):
    new_s = []
    r = 2
    for i in range(r,len(s)-r):
        new_s.append(sum(s[i-r:i+r])/(r*2+1))
    return new_s
def NN_series(g_data, p_data):
    N = len(g_data)
    M = len(p_data)
    score = N*M
    max_score = score
    freqs = [0]*N
    for p in range(M):
        dists = [(g,int(dtw_dist(g_data[g], p_data[p]))) for g in range(N)]
        ranks = [x[0] for x in sorted(dists, key=lambda tup: tup[1])]
        print(p, ranks[:6])
        freqs[ranks.index(p)] += 1
        score -= 2*ranks.index(p) # wrong classifications may give negative
    print(score/max_score)
    print(freqs)
##    plt.bar([i for i in range(N)], freqs)
    #z: euclidian dist gives .8, dtw gives 0.94 with w=10
    #angle: uhhhh

## Kmeans clustering
## clustering on whole series is bad: 250 features and only 27 points
def cluster_series(g_data, p_data):
    random.seed(0)
    M = kmeans(g_data, 3)
    #print(M)
    g_c = [km_classify(M,s) for s in g_data]
    p_c = [km_classify(M,s) for s in p_data]
    print(g_c)
    print(p_c)
    x = [1 if g_c[i] == p_c[i] else 0 for i in range(len(g_c))]
    print(sum(x)/len(g_c))
def km_classify(M, s):
    return min(M, key=lambda m:dist(M[m], s))
def kmeans(D, K): # D will be array of data points, K is # of clusters
    N = len(D)
    M = M_gen_pp(D,K) # generate centers
    Z = [-1]*N # cluster assignments
    delta = 1
    while delta > 0.01:
        Z = [min(M, key=lambda m:dist(M[m], D[n])) for n in range(N)] # assign to clusters
        print(Z)
        new_M = {k:avg([D[n] for n in range(N) if Z[n] == k]) for k in range(K)}
        delta = sum([dist(M[i], new_M[i]) for i in range(K)])
        M = new_M
    return M
def M_gen_pp(D,K):
    N = len(D)
    W = [1 for d in D] # weights
    Z = [-1 for d in D] # cluster assignments
    M = {}
    for k in range(K): # choose and update choice weighting
        M[k] = random.choices(D, W)[0]
        W = [min([dist(M[m], D[n]) for m in M]) for n in range(N)]
    return M
def avg(D): # D = list of vectors
    N = len(D)
    if N > 0:
        dim = len(D[0])
        return [sum(d[i] for d in D)/N for i in range(dim)]
    return None

# try partitioning series into double cycles, so we have around 27*6 training
# dtw works on cycles with different length
def partition_data(data):
    i = 0
##    for s in data:
##        plot(s)

# TRY:
# try dtw + KNN voting on partitioned cycles
# can we do clustering on partitioned cycles?

# TRIED:
# dtw with angle of inclination: very bad - data is too messy
# clustering with whole time series: also bad
        
# 34 subjects with ellipse data
##g_raw = json.load(open("a_gallery_10km.json", "r"))
##p_raw = json.load(open("a_probe_10km.json", "r"))
##g_n = [truncate_a(smooth(s)) for s in g_raw]
##p_n = [truncate_a(smooth(s)) for s in p_raw]
##for i in range(len(g_n)):
##    plot_d(g_n[i], p_n[i])
##fit_series(g_n, p_n)

# 28 subjects, 300 data point time series each of z-height
g_raw = json.load(open("z_gallery_10km.json", "r"))
p_raw = json.load(open("z_probe_10km.json", "r"))
g_n = [truncate_z(s) for s in g_raw]
p_n = [truncate_z(s) for s in p_raw]
##cluster_series(g_n,p_n)
NN_series(g_n, p_n)
