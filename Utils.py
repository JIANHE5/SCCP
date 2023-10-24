import FCM
import scipy.io
import numpy as np
import random as rn
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from obspy.signal.tf_misfit import cwt
from matplotlib.colors import ListedColormap


def seisplot_wig1(s, inc=1, scale=0.8, lw=1, highlight=False, lightstep=10, figsize=(7.5, 6)):
    
    nt = s.shape[0]
    xmax = s.shape[1]
    # vmax = np.max(s1)
    t = np.arange(nt)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in np.arange(0, xmax, inc):
        x1 = scale * s[:, i] + i
        
        ax.plot(x1, t, 'k', lw=lw)
        ax.fill_betweenx(t, x1, x2=i, where=x1 >= i, facecolor='k', interpolate=True)
        if highlight == True:
            if i % lightstep == 0:
                ax.plot(x1, t, 'r', lw=lw*2)
                ax.fill_betweenx(t, x1, x2=i, where=x1 >= i, facecolor='r', interpolate=True)
            if i == int(xmax/lightstep)*lightstep:
                ax.plot(x1, t, 'r', lw=lw*2, label='Training traces')
                ax.legend(loc='upper right', fontsize=12)
    # ax.invert_yaxis()
    # ax.set_xlim(-1, xmax)
    # ax.set_ylim(nt, 0)
    # ax.set_ylabel('Time samples', fontsize=13)
    # ax.set_xlabel('Traces', fontsize=13)
    fig.tight_layout()
    
    return fig, ax


def predict_label(signal, model, spectrum, nf, Filter):
    labelproposed = np.zeros((np.shape(signal)[0]))                 # (100,)
    clabelf = np.zeros((np.shape(signal)[0], np.shape(signal)[1]))  # (100, 2048)
    for ix in range(0, np.shape(signal)[0]):
        scalre = spectrum[ix]                                        # (2048, 256)
        scal1 = np.reshape(scalre, (1, np.shape(signal)[1], nf, 1))  # (1, 2048, 256, 1)

        outenc = model.predict(scal1)**2                             # (1, 2048, 1, 16)
        outenc = np.reshape(outenc, (np.shape(signal)[1], Filter))   # (2048, 16)

        # outenc = autoencoder.predict(scal1)**2
        # outenc = np.reshape(outenc, (np.shape(signal)[1],nf))

        var = []
        indout = []
        le = 8

        for io in range(0, Filter):
            xz = outenc[:, io]
            var.append(np.std(xz))                # 返回数组元素的标准差
        varord = np.flip(np.sort(var), axis=-1)   # 从大到小排列

        for iuy in range(0, le):
            indout.append(np.where(var == varord[iuy])[0][0])      # 大标准差的位置 (le,)

        outenc1 = outenc[:, indout]               # 选择标准差大的      (2048, le)
        clabelpre = np.zeros_like(outenc1)                         # (2048, le)
        for iux in range(0, le):

            iuxx = indout[iux]
            fet = outenc[:, iuxx]                # 选择标准差大的
            fet = fet / np.max(np.abs(fet))

            me = np.mean(fet)
            clabelxx = np.zeros_like(fet)
            indlarge = np.where(fet > 1*me)[0]
            clabelxx[indlarge] = 1
            clabelpre[:, iux] = clabelxx                     # (2048, le)
            # clabelpre[:,iux] = fet

        cluster = KMeans(n_clusters=2, random_state=0).fit(clabelpre)
        clabel = cluster.labels_
        acp = [len(np.where(clabel == 1)[0]), len(np.where(clabel == 0)[0])]
        ap = np.min(acp)
        ak = np.where(acp == ap)
        if ak[0][0] == 1:
            clabel = (clabel-1)*-1
        try:
            labelproposed[ix] = np.max([np.where(clabel == 1)[0][0], np.where(clabel == 0)[0][0]])
            labelproposed[ix] = np.where(clabel == 1)[0][0]
            clabelf[ix, 0:len(clabel)] = clabel
        except:
            labelproposed[ix] = 0
            clabelf[ix, 0:len(clabel)] = clabel
            
    return clabelf, labelproposed


def FCM_predict(signal, model, spectrum, nf, Filter):
    labelproposed = np.zeros((np.shape(signal)[0]))  # (100,)
    clabelf = np.zeros((np.shape(signal)[0], np.shape(signal)[1]))  # (100, 2048)
    for ix in range(0, np.shape(signal)[0]):
        scalre = spectrum[ix]  # (2048, 256)
        scal1 = np.reshape(scalre, (1, np.shape(signal)[1], nf, 1))  # (1, 2048, 256, 1)

        outenc = model.predict(scal1) ** 2  # (1, 2048, 1, 16)
        outenc = np.reshape(outenc, (np.shape(signal)[1], Filter))  # (2048, 16)

        # outenc = autoencoder.predict(scal1)**2
        # outenc = np.reshape(outenc, (np.shape(signal)[1],nf))

        var = []
        indout = []
        le = 8

        for io in range(0, Filter):
            xz = outenc[:, io]
            var.append(np.std(xz))  # 返回数组元素的标准差
        varord = np.flip(np.sort(var), axis=-1)  # 从大到小排列

        for iuy in range(0, le):
            indout.append(np.where(var == varord[iuy])[0][0])  # 大标准差的位置 (le,)

        outenc1 = outenc[:, indout]  # 选择标准差大的      (2048, le)
        clabelpre = np.zeros_like(outenc1)  # (2048, le)
        for iux in range(0, le):
            iuxx = indout[iux]
            fet = outenc[:, iuxx]  # 选择标准差大的
            fet = fet / np.max(np.abs(fet))

            me = np.mean(fet)
            clabelxx = np.zeros_like(fet)
            indlarge = np.where(fet > 1 * me)[0]
            clabelxx[indlarge] = 1
            clabelpre[:, iux] = clabelxx  # (2048, le)
            # clabelpre[:,iux] = fet

        res_U = FCM.fuzzy(clabelpre, 2, 2)
        clabel = [np.argmax(itm) for itm in res_U]
        acp = [len(np.where(clabel == 1)[0]), len(np.where(clabel == 0)[0])]
        ap = np.min(acp)
        ak = np.where(acp == ap)
        if ak[0][0] == 1:
            clabel = (clabel - 1) * -1
        try:
            labelproposed[ix] = np.max([np.where(clabel == 1)[0][0], np.where(clabel == 0)[0][0]])
            labelproposed[ix] = np.where(clabel == 1)[0][0]
            clabelf[ix, 0:len(clabel)] = clabel
        except:
            labelproposed[ix] = 0
            clabelf[ix, 0:len(clabel)] = clabel

    return clabelf, labelproposed