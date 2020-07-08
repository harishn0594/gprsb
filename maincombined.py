import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn import preprocessing
import GPy
import pandas as pd
from timeit import default_timer as timer
from GPy import util
from copy import deepcopy
"""
    This uses the training data of the combination of both surface profile and force curve. 
    Also included a test train split. 
"""


def needleman(exp_profile):
    z1 = 0
    z2 = 0

    dbsize = dataneedle.shape[0]

    nums1 = np.zeros((dbsize, 1))
    nums2 = np.zeros((dbsize, 1))
    nums3 = np.zeros((dbsize, 1))

    sep = 146                    # 146 for combined, 66 for use_prof

    # href = dbdata[0, sep]
    # href = 0.119071
    # fref = emod * (href ** 2)

    # dbdata[:, 0:sep] = dbdata[:, 0:sep] / fref
    # dbdata[:, sep:-2] = dbdata[:, sep:-2] / href

    for i in range(dbsize):
        # db_profile = dbdata[i, 0:-2]
        db_profile = dataneedle[i, 0:-2]
        # num1 = posterior(np.abs(exp_profile[0, 0:sep]/fref), np.abs(db_profile[0:sep]))
        # num2 = posterior(np.abs(exp_profile[0, sep:]/href), np.abs(db_profile[sep:]))

        # num1 = posterior(exp_profile[0, 0:sep]/fref, np.abs(db_profile[0:sep]))
        # num2 = posterior(exp_profile[0, sep:]/href, np.abs(db_profile[sep:]))

        # temp = np.concatenate([np.abs(exp_profile[0, 0:sep]/fref).reshape(1, -1), np.abs(exp_profile[0, sep:]/href).reshape(1, -1)], axis=1)
        temp = np.concatenate([(exp_profile[0, 0:sep]/fref).reshape(1, -1), (exp_profile[0, sep:]/href).reshape(1, -1)], axis=1)
        num3 = posterior(temp, db_profile)

        z3 = num3

        # z1 += num1
        # z2 += num2
        # # zsf += num1*num2
        # nums1[i, 0] = num1
        # nums2[i, 0] = num2
        nums3[i, 0] = num3

    # num_force = nums1/z1
    # num_prof = nums2 / z2
    # zsf = np.sum(np.multiply(num_force, num_prof))
    # postpro = num_force * num_prof / zsf

    # fig = plt.figure(20)
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(Egrid, Nugrid, num_er, cmap='viridis')

    # pos1 = nums1/z1
    # pos2 = nums2/z2
    # zsf = np.sum(np.multiply(pos1, pos2))
    # post = np.multiply(pos1, pos2)/zsf
    # post[post < 1e-4] = 0

    post = nums3/z3
    ind1 = np.argmax(post)

    store_y = dataneedle[ind1, -2]
    store_n = dataneedle[ind1, -1]

    # plt.figure(21)
    # plt.plot(np.arange(len(post)), post, label=str("Y %0.4f \n N %0.4f\n %d" % (store_y, store_n, ind1)))
    # plt.xlabel('DB Index')
    # plt.title('Posteriors')
    # plt.legend(loc='best')

    return store_y/emod, store_n


def posterior(exp1, db1, *args):
    # exp2 = args[0]
    # db2 = args[1]
    prior_sy = 1/siz
    prior_n = 1/siz

    # prior_sy = 0.001
    # prior_n = 0.001
    lhood1 = likelihood(exp1, db1)
    numr1 = lhood1 * prior_sy * prior_n

    # lhood2 = likelihood(exp2, db2)
    # num2 = lhood2 * prior_sy * prior_n
    return numr1


def likelihood(a, b):
    # b = preprocessing.minmax_scale(b, feature_range=(0.001, 1), axis=1)
    # a = preprocessing.minmax_scale(a, feature_range=(0.001, 1), axis=1)
    temp = a - b
    # var = np.sum([pow((a[kk]-b[kk]), 2) for kk in range(len(a))])/len(a)
    var = np.mean(0.1/100 * b)
    pdf1 = st.norm.pdf(a-b)
    # pdf = st.norm.pdf(temp)
    return np.product(pdf1)


def gpr(traindata, exp_profile, trainsize, do_x=0, do_y=0, *args):
    """
        data - takes in the data used to train the GPR model
        exp_profile - is the experimental surface residual profile of the specimen
        trainsize - size of the data to be used to train the model. for study purpose of influence of size on prediction
        args include the rows to be predicted from the data. usually from the test portion of the split.
    """
    # Using Gaussian Process
    # Build a model
    # do_x = 0
    # do_y = 0

    x_train = traindata[0:trainsize, 0:-2]
    y_train = traindata[0:trainsize, -2:]

    sep = 146                   # 146 for combined, 66 for use_prof,

    # href = x_train[0, sep]
    href = 0.119071
    fref = emod * (href**2)

    if do_x == 1:
        # x_scaler = preprocessing.MinMaxScaler()
        # x_train = x_scaler.fit_transform(x_train)
        # x_train = preprocessing.minmax_scale(x_train, axis=1)
        x_train[:, 0:sep] = x_train[:, 0:sep]/fref
        x_train[:, sep:] = x_train[:, sep:]/href
    else:
        None

    if do_y == 1:
        # ys_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # ys_train = ys_scaler.fit_transform(y_train[:, 0].reshape(-1, 1))
        ys_train = (y_train[:, 0]/emod).reshape(-1, 1)
        # N_scaler = preprocessing.MinMaxScaler(feature_range=(0.001, 1))
        # n_train = N_scaler.fit_transform(y_train[:, 1].reshape(-1, 1))
    else:
        ys_train = y_train[:, 0].reshape(-1, 1)

    n_train = y_train[:, 1].reshape(-1, 1)

    # y_train_mogp = np.concatenate([ys_train, n_train], axis=1) # Keep commented for separate models for ys and N

    kernel1 = GPy.kern.RBF(input_dim=datacols, variance=1, lengthscale=1)
    modelys = GPy.models.GPRegression(x_train, ys_train, kernel1)

    kernel2 = GPy.kern.RBF(input_dim=datacols)
    modeln = GPy.models.GPRegression(x_train, n_train, kernel2)

    # model_multi = GPy.models.GPCoregionalizedRegression(x_train, y_train_mogp, kernel=kernmogp)

    var1 = kernel1.variance
    var2 = kernel2.variance

    ls1 = kernel1.lengthscale
    ls2 = kernel2.lengthscale

    modelys.optimize()
    modeln.optimize()

    modelys.Gaussian_noise.variance = 0
    modeln.Gaussian_noise.variance = 0

    var1 = kernel1.variance
    var2 = kernel2.variance

    ls1 = kernel1.lengthscale
    ls2 = kernel2.lengthscale

    if do_x == 0:
        exper = exp_profile
    else:
        # exper = x_scaler.transform(exp_profile)
        # exper = preprocessing.minmax_scale(exp_profile, axis=1)
        exper = np.concatenate((exp_profile[:, 0:sep]/fref, exp_profile[:, sep:]/href), axis=1)

    meanys, vys = modelys.predict(exper, full_cov=False, include_likelihood=False)
    meann, vn = modeln.predict(exper, full_cov=False, include_likelihood=False)
    # means, vars = model_multi.predict(exper)

    ys_std = np.sqrt(vys)
    n_std = np.sqrt(vn)

    sypred = meanys
    npred = meann

    errsy = [i[0] for i in np.ndarray.tolist(ys_std)]
    errn = [i[0] for i in np.ndarray.tolist(n_std)]

    # removed from this point the bar plots. compare with combined_newtrain.py

    return sypred, errsy, npred, errn


def generatedist(mu, sigma):
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100).reshape(-1, 1)
    y = st.norm.pdf(x, mu, sigma).reshape(-1, 1)
    yscaler = preprocessing.MinMaxScaler()
    y = yscaler.fit_transform(y)
    return x, y


if __name__ == '__main__':
    emod = 100
    # n_db = np.linspace(0.001, 0.5, 20)
    # s0_db = np.linspace(1e-3 * emod, 2e-2 * emod, 21)
    siz = 20

    # findfile = str(siz) + 'x' + str(siz) + 'comb' + '.csv'
    # df = pd.read_csv(findfile, header=None, delimiter=',')
    df = pd.read_csv('combined_400.csv', header=None, delimiter=',')
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    data = df.values
    data[:, -2:] = np.round(data[:, -2:], 4)

    # df1 = pd.read_csv('combined_400.csv', header=None, delimiter=',')
    datatest = df.values[400:]
    datatest[:, -2:] = np.round(datatest[:, -2:], 4)
    datacols = len(data[0, 0:-2])

    # file = 'tpcomb' + str(siz) + '.txt'
    # with open(file, 'r') as openfile:
    #  temp = [int(float(i)) for i in openfile.read().splitlines()]

    # testrow = list(temp[5])
    # testrow = list(range(20))
    testrow = [10, 15]


    # Post processing
    act_sy = datatest[testrow, -2]
    act_n = datatest[testrow, -1]

    needle_sy = []
    needle_n = []

    test = datatest[testrow, :].reshape(len(testrow), -1)
    x_test = test[:, 0:-2].reshape(len(testrow), -1)
    y_test = test[:, -2:].reshape(len(testrow), -1)

    # noise_test = np.random.normal(0, 0.0002, size=x_test.shape)
    # # # noise_test = x_test * 0.8
    # # # noise_test = np.multiply(np.ones(x_test.shape) * 0.04, x_test)
    # x_test = x_test + noise_test                                                            # noise here

    # print('Actual parameters: sy:' + str(y_test[:, 0]) + ' N: ' + str(y_test[:, 1]))
    tsize = siz**2                                                                             # training size here
    nbar = 0

    datagpr = deepcopy(data)
    dataneedle = deepcopy(data[0:tsize, :])
    symeans = []
    sysds = []

    nmeans = []
    nsds = []
    kk = tsize
    # for jj in [0, 1]:
    #     if jj == 0:
    #         sy_p, esy, n_p, en = gpr(datagpr, x_test, kk, do_x=0, do_y=0)
    #     if jj == 1:
    #         sy_p, esy, n_p, en = gpr(datagpr, x_test, kk, do_x=1, do_y=1)
    #
    #     sy_mean = [i[0] for i in sy_p]
    #     sy_sd = [i for i in esy]
    #
    #     symeans.append(sy_mean)
    #     sysds.append(sy_sd)
    #
    #     n_mean = [i[0] for i in n_p]
    #     n_sd = [i for i in en]
    #
    #     nmeans.append(n_mean)
    #     nsds.append(n_sd)

    for kk in [tsize]:
        sy_p, esy, n_p, en = gpr(datagpr, x_test, kk, do_x=1, do_y=1)
        gpr_sy = [i[0] for i in sy_p]
        gpr_n = [i[0] for i in n_p]

    href = 0.119071
    fref = emod * (href ** 2)
    sep = 146

    dataneedle[:, 0:sep] = dataneedle[:, 0:sep] / fref
    dataneedle[:, sep:-2] = dataneedle[:, sep:-2] / href

    for it in testrow:
        test = datatest[it, :].reshape(1, -1)
        x_test = test[:, 0:datacols].reshape(1, -1)
        sy_temp, n_temp = needleman(x_test)
        needle_sy.append(sy_temp)
        needle_n.append(n_temp)


    def plotalltespoints():
        fig = plt.figure(500)
        plt.plot(list(range(len(testrow))), datatest[testrow, -2] / emod, list(range(len(testrow))), gpr_sy,
                 list(range(len(testrow))),
                 needle_sy)
        plt.xlabel('testpoints')
        plt.ylabel('yield strain')
        plt.legend(['actual', 'gpr', 'sb'])
        plt.savefig('testing.png')

        fig = plt.figure(600)
        plt.plot(list(range(len(testrow))), datatest[testrow, -1], list(range(len(testrow))), gpr_n,
                 list(range(len(testrow))),
                 needle_n)
        plt.xlabel('testpoints')
        plt.ylabel('hardening exps')
        plt.legend(['actual', 'gpr', 'sb'])
        plt.savefig('testing2.png')


    def plotfew():
        bar_width = 0.05
        x = [0 - bar_width, 0, 0 + bar_width]

        text = ['Actual', 'GPR', 'SB']
        for kkk in range(len(testrow)):
            ppp = 0
            fig = plt.figure(240 + kkk)
            plt.ylabel('Yield Strain')
            plt.title('Actual vs GPR vs Simplified Bayesian')
            plt.xticks([81])
            plt.xlim([-0.25, 0.25])
            exp1 = plt.bar(x[ppp], [act_sy[kkk] / emod], bar_width)
            exp2 = plt.bar(x[ppp + 1], [gpr_sy[kkk]], bar_width, yerr=esy[kkk] / emod)
            exp3 = plt.bar(x[ppp + 2], [needle_sy[kkk]], bar_width)
            plt.legend(text)
            plt.xlabel('Test Point %d' % (testrow[kkk]))
            plt.savefig(str(testrow[kkk]) + 'straincomb' + '.png')

            fig2 = plt.figure(250 + kkk)
            exp11 = plt.bar(x[ppp], [act_n[kkk]], bar_width)
            exp21 = plt.bar(x[ppp + 1], [gpr_n[kkk]], bar_width, yerr=en[kkk])
            exp31 = plt.bar(x[ppp + 2], [needle_n[kkk]], bar_width)

            plt.xticks([81])
            plt.xlim([-0.25, 0.25])
            plt.xlabel('Test Point %d' % (testrow[kkk]))
            plt.ylabel('Hardening Exponent')
            plt.title('Actual vs GPR vs Simplified Bayesian')
            plt.legend(text)
            plt.savefig(str(testrow[kkk]) + 'hardcomb' + '.png')

        # plotfew()


    # plotalltespoints()
    plotfew()

    print('Yield Strain')
    print([i/emod for i in act_sy])
    print('GPR: ')
    print([i for i in gpr_sy])
    print('Standard Deviation')
    print(esy)
    print('Needleman: ')
    print([i for i in needle_sy])
    print('---------------')

    print('Hardening Exponent')
    print(act_n)
    print('GPR: ')
    print(gpr_n)
    print('Std Deviation')
    print(en)
    print('Needleman: ')
    print(needle_n)
    print('---------------')

    plt.show()
