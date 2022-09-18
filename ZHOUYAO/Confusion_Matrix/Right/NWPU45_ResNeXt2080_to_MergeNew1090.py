import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

np.set_printoptions(threshold=np.inf)


# best_acc：0.,923968253968254
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=4)
    plt.yticks(tick_marks, classes, fontsize=4)

    sns.set(font_scale=0.2)  # 调整混淆矩阵中字体的大小
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig("../plot/NWPU45_resnext_2080_to_mergenew_1090_Fig2.png", dpi=500, format="png", bbox_inches='tight')
    plt.show()


cnf_matrix = np.array([[0.94336283, 0.01238938, 0., 0.00176991, 0., 0.00176991,
                        0., 0.00353982, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.00176991, 0.,
                        0., 0., 0., 0., 0.03362832, 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00176991, 0., ],
                       [0.00175747, 0.90158172, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.00351494, 0.,
                        0., 0., 0., 0., 0., 0.00175747,
                        0.00527241, 0., 0., 0., 0.00175747, 0.,
                        0., 0., 0.00527241, 0.0228471, 0., 0.00175747,
                        0.00878735, 0., 0.00527241, 0.00175747, 0.03514938, 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00351494, 0., ],
                       [0., 0., 0.93333333, 0., 0., 0.0017094,
                        0., 0., 0.0017094, 0., 0., 0.00683761,
                        0., 0.0017094, 0.0017094, 0.0017094, 0.00512821, 0.,
                        0., 0.0017094, 0., 0., 0.01538462, 0.0034188,
                        0., 0., 0., 0.01538462, 0., 0.,
                        0., 0., 0.0017094, 0., 0., 0.,
                        0., 0., 0., 0.0017094, 0.0034188, 0.0017094,
                        0.0017094, 0., 0., ],
                       [0., 0., 0.00177936, 0.95729537, 0., 0.,
                        0., 0., 0., 0., 0.00177936, 0.,
                        0., 0., 0.00711744, 0.00177936, 0.00177936, 0.,
                        0.00177936, 0., 0., 0., 0.00355872, 0.00177936,
                        0., 0., 0., 0.00355872, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00711744, 0., 0., 0.01067616,
                        0., 0., 0., ],
                       [0., 0.00181818, 0., 0., 0.97090909, 0.,
                        0., 0., 0., 0.00363636, 0., 0.,
                        0.00181818, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.00181818, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.01090909, 0., 0., 0.00363636,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00181818, 0.00363636],
                       [0., 0.00331126, 0.00165563, 0., 0.00331126, 0.89072848,
                        0., 0., 0., 0., 0.00331126, 0.,
                        0., 0., 0.02483444, 0., 0., 0.00993377,
                        0., 0., 0.00165563, 0., 0.00331126, 0.,
                        0., 0., 0.01324503, 0.00165563, 0., 0.00165563,
                        0., 0.00331126, 0., 0.00165563, 0.0115894, 0.,
                        0.0115894, 0., 0., 0.00331126, 0.00331126, 0.,
                        0., 0., 0.00662252],
                       [0., 0., 0.00175131, 0., 0.00175131, 0.,
                        0.98073555, 0., 0., 0., 0., 0.,
                        0.00700525, 0.00175131, 0., 0., 0., 0.,
                        0., 0., 0., 0.00175131, 0., 0.,
                        0., 0.00175131, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.00175131, 0., 0.00175131],
                       [0.00186567, 0., 0., 0.00373134, 0., 0.,
                        0., 0.82089552, 0., 0., 0.02238806, 0.00373134,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.00186567,
                        0., 0., 0., 0.12126866, 0., 0.,
                        0., 0.00373134, 0., 0.00186567, 0.00373134, 0.,
                        0., 0., 0.00373134, 0., 0.01119403, 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0.98568873, 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.00536673, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00536673, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.00178891, 0.,
                        0.00178891, 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.97806216, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00182815, 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00182815, 0., 0., 0.00731261, 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00914077, 0.00182815],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0.04920914, 0., 0., 0.86994728, 0.00527241,
                        0., 0., 0.00351494, 0., 0., 0.,
                        0.00527241, 0.00351494, 0., 0., 0., 0.00175747,
                        0., 0., 0.00527241, 0.02636204, 0.00527241, 0.,
                        0.00878735, 0.00702988, 0., 0., 0., 0.,
                        0., 0., 0., 0.00702988, 0., 0.,
                        0., 0.00175747, 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0.00185874, 0., 0., 0.00929368, 0.91635688,
                        0., 0., 0., 0., 0., 0.,
                        0.00371747, 0.00743494, 0., 0., 0., 0.0464684,
                        0.00371747, 0., 0., 0.00557621, 0., 0.,
                        0., 0., 0., 0.00185874, 0., 0.,
                        0., 0., 0., 0., 0., 0.00371747,
                        0., 0., 0., ],
                       [0., 0.0035461, 0., 0., 0.00177305, 0.,
                        0., 0., 0., 0.0035461, 0., 0.,
                        0.93262411, 0., 0., 0., 0.00177305, 0.,
                        0., 0., 0.0035461, 0.0035461, 0., 0.,
                        0., 0.03546099, 0., 0., 0., 0.,
                        0., 0., 0.00531915, 0., 0., 0.00177305,
                        0., 0.0035461, 0., 0., 0., 0.,
                        0., 0., 0.0035461],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.90556492, 0.00337268, 0.00168634, 0.00168634, 0.,
                        0., 0., 0., 0., 0.03709949, 0.00337268,
                        0.00337268, 0.00337268, 0., 0., 0., 0.,
                        0., 0., 0.00168634, 0., 0., 0.,
                        0., 0., 0.00505902, 0., 0., 0.,
                        0., 0., 0.03372681],
                       [0., 0.00206612, 0., 0., 0., 0.00413223,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00826446, 0.94834711, 0., 0.00206612, 0.,
                        0., 0.00206612, 0., 0., 0.00826446, 0.,
                        0., 0., 0.00826446, 0., 0., 0.,
                        0.00413223, 0., 0., 0., 0.00619835, 0.,
                        0.00206612, 0., 0.00413223, 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.00361664,
                        0., 0., 0.00180832, 0.98191682, 0., 0.,
                        0., 0., 0., 0., 0.00361664, 0.00180832,
                        0.00180832, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.00361664,
                        0., 0., 0.00180832],
                       [0., 0., 0.00560748, 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.00373832,
                        0., 0., 0., 0., 0.97570093, 0.,
                        0.00186916, 0., 0., 0., 0., 0.00186916,
                        0., 0., 0., 0.00560748, 0.00186916, 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.00186916, 0., 0.,
                        0., 0., 0.00186916],
                       [0., 0., 0., 0., 0., 0.00363636,
                        0., 0., 0., 0., 0.00363636, 0.,
                        0., 0.00181818, 0., 0., 0., 0.97818182,
                        0.00181818, 0., 0., 0., 0., 0.,
                        0.00181818, 0., 0., 0.00181818, 0.00181818, 0.,
                        0., 0., 0., 0., 0., 0.00181818,
                        0.00363636, 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0.00718133, 0.00179533, 0., 0., 0., 0.,
                        0., 0.00359066, 0., 0., 0.00359066, 0.,
                        0., 0., 0.00359066, 0., 0.00179533, 0.00179533,
                        0.89766607, 0.00359066, 0., 0., 0., 0.00359066,
                        0.00359066, 0., 0.00179533, 0.00897666, 0.005386, 0.00718133,
                        0.01615799, 0., 0., 0., 0.00179533, 0.,
                        0.005386, 0., 0., 0.00179533, 0.00897666, 0.,
                        0., 0.01077199, 0., ],
                       [0., 0.00175747, 0., 0.00527241, 0., 0.00175747,
                        0., 0.00351494, 0., 0., 0.01054482, 0.,
                        0., 0., 0.01405975, 0., 0., 0.,
                        0., 0.93673111, 0., 0., 0., 0.00351494,
                        0.00175747, 0., 0.00351494, 0.00527241, 0., 0.,
                        0., 0., 0., 0.00878735, 0.00175747, 0.,
                        0., 0., 0., 0., 0., 0.00175747,
                        0., 0., 0., ],
                       [0., 0.00718133, 0., 0., 0.01795332, 0.,
                        0., 0., 0., 0.00718133, 0., 0.,
                        0.00179533, 0., 0., 0., 0., 0.,
                        0., 0., 0.93895871, 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.01256732, 0., 0., 0.00179533,
                        0., 0.00179533, 0., 0., 0., 0.,
                        0., 0., 0.01077199],
                       [0., 0., 0., 0., 0.00584795, 0.00389864,
                        0., 0., 0.01169591, 0., 0., 0.00194932,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00194932, 0.95126706, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00779727, 0., 0., 0.,
                        0., 0.00194932, 0., 0., 0., 0.,
                        0., 0., 0.01364522],
                       [0., 0., 0., 0.00389105, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00972763, 0., 0.00194553, 0., 0.,
                        0., 0., 0., 0., 0.96692607, 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0.00972763, 0., 0., 0., 0.,
                        0.00389105, 0., 0., 0., 0., 0.,
                        0., 0., 0.00389105],
                       [0., 0., 0.0034904, 0.0052356, 0., 0.,
                        0., 0.0034904, 0., 0., 0., 0.03315881,
                        0., 0., 0.0017452, 0., 0., 0.,
                        0.0034904, 0.0069808, 0., 0., 0., 0.84816754,
                        0.0122164, 0., 0., 0.0122164, 0., 0.,
                        0., 0., 0.0017452, 0.0052356, 0., 0.,
                        0., 0., 0.05061082, 0., 0., 0.0122164,
                        0., 0., 0., ],
                       [0.00169492, 0., 0.00169492, 0., 0., 0.,
                        0., 0., 0., 0., 0.00169492, 0.04237288,
                        0., 0., 0., 0., 0., 0.00169492,
                        0.00847458, 0., 0., 0., 0., 0.02372881,
                        0.91355932, 0., 0., 0.00169492, 0.00169492, 0.,
                        0., 0., 0., 0.00169492, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00340136, 0.00170068, 0., 0.,
                        0.04251701, 0.00170068, 0., 0.00510204, 0., 0.,
                        0., 0., 0., 0., 0.00170068, 0.,
                        0., 0.89455782, 0., 0., 0., 0.,
                        0., 0.01360544, 0.01870748, 0., 0., 0.,
                        0., 0.00170068, 0., 0., 0., 0.,
                        0.0085034, 0., 0.00680272],
                       [0., 0., 0., 0., 0., 0.01247772,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.01426025, 0., 0.00356506, 0.,
                        0., 0.00356506, 0., 0., 0., 0.,
                        0., 0., 0.95187166, 0., 0.00178253, 0.00178253,
                        0., 0., 0., 0.00891266, 0., 0.,
                        0., 0., 0., 0., 0., 0.00178253,
                        0., 0., 0., ],
                       [0.0043956, 0., 0., 0.0021978, 0., 0.,
                        0., 0.14065934, 0., 0., 0.01098901, 0.0021978,
                        0., 0., 0.0021978, 0., 0.0021978, 0.,
                        0.0021978, 0., 0., 0., 0., 0.,
                        0.0021978, 0., 0., 0.82417582, 0., 0.,
                        0., 0.0043956, 0., 0., 0., 0.,
                        0.0021978, 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0.00509338, 0., 0.00169779, 0.0339559, 0., 0.,
                        0., 0., 0., 0., 0.00509338, 0.,
                        0., 0., 0.02716469, 0., 0., 0.00339559,
                        0.00169779, 0.00679117, 0., 0., 0.00169779, 0.,
                        0.00509338, 0., 0., 0.00339559, 0.92699491, 0.00169779,
                        0.00169779, 0., 0., 0., 0., 0.,
                        0., 0., 0.00169779, 0., 0.00169779, 0.,
                        0., 0.00169779, 0., ],
                       [0.00322581, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.00483871, 0.,
                        0., 0., 0.04032258, 0., 0., 0.,
                        0.01129032, 0., 0., 0., 0., 0.,
                        0.0016129, 0., 0.00322581, 0., 0.0016129, 0.8483871,
                        0.06129032, 0., 0., 0.0016129, 0.00645161, 0.,
                        0.0016129, 0., 0., 0., 0.00322581, 0.,
                        0.00645161, 0.00322581, 0.0016129],
                       [0.0017094, 0.00854701, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.01538462, 0.0017094,
                        0., 0., 0.00683761, 0.0017094, 0.0034188, 0.0034188,
                        0.02222222, 0.0034188, 0., 0., 0., 0.,
                        0., 0., 0.0034188, 0.02393162, 0., 0.03418803,
                        0.83418803, 0., 0., 0.00512821, 0.00683761, 0.,
                        0.00512821, 0., 0., 0.0017094, 0.0017094, 0.,
                        0., 0.01367521, 0.0017094],
                       [0., 0.00373832, 0., 0., 0., 0.,
                        0., 0., 0., 0.00186916, 0., 0.,
                        0., 0., 0.00186916, 0., 0., 0.,
                        0., 0., 0., 0., 0.00747664, 0.00186916,
                        0., 0., 0., 0.00373832, 0.00186916, 0.00186916,
                        0.00186916, 0.91962617, 0.00373832, 0., 0.01308411, 0.,
                        0.00186916, 0., 0., 0., 0., 0.,
                        0.03551402, 0., 0., ],
                       [0., 0.0051458, 0., 0.00171527, 0.00171527, 0.0051458,
                        0., 0.00171527, 0., 0.00171527, 0.00171527, 0.00343053,
                        0., 0.00686106, 0.0051458, 0.01200686, 0., 0.,
                        0.00171527, 0., 0.01543739, 0.02744425, 0.00171527, 0.,
                        0., 0., 0., 0.0102916, 0., 0.00171527,
                        0., 0.0102916, 0.86106346, 0., 0.00171527, 0.,
                        0.00171527, 0., 0., 0.0343053, 0., 0.,
                        0., 0., 0.01715266],
                       [0.00181488, 0., 0., 0., 0., 0.,
                        0., 0.00181488, 0., 0., 0.00181488, 0.00544465,
                        0., 0., 0.00181488, 0., 0., 0.,
                        0., 0.00362976, 0., 0., 0., 0.,
                        0., 0., 0.00181488, 0.00362976, 0., 0.,
                        0., 0., 0., 0.97640653, 0., 0.,
                        0., 0., 0., 0.00181488, 0., 0.,
                        0., 0., 0., ],
                       [0.00788955, 0.02169625, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00591716, 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.00197239, 0.,
                        0.00394477, 0., 0., 0., 0.95463511, 0.,
                        0.00197239, 0., 0., 0., 0., 0.,
                        0., 0., 0.00197239],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00364964, 0.00364964, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.99270073,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0.00488599, 0.00162866, 0., 0., 0.00977199, 0.,
                        0., 0.00488599, 0., 0.01302932, 0.00325733, 0.,
                        0., 0., 0., 0., 0., 0.01465798,
                        0.00325733, 0., 0.03094463, 0., 0., 0.,
                        0., 0., 0., 0.01302932, 0., 0.,
                        0., 0.00162866, 0.00162866, 0., 0.00162866, 0.00814332,
                        0.8713355, 0., 0., 0., 0.01140065, 0.,
                        0., 0., 0.00488599],
                       [0., 0., 0., 0., 0.00171527, 0.,
                        0., 0., 0., 0.0051458, 0., 0.00171527,
                        0.00343053, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.00857633, 0., 0.,
                        0., 0.01200686, 0., 0.00171527, 0., 0.,
                        0., 0., 0., 0., 0., 0.0102916,
                        0., 0.95197256, 0., 0., 0., 0.,
                        0., 0.00171527, 0.00171527],
                       [0., 0., 0.00359712, 0., 0., 0.,
                        0., 0.01258993, 0., 0., 0., 0.,
                        0., 0.00539568, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.01798561, 0.02517986,
                        0., 0., 0., 0.00179856, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.00179856, 0., 0.92625899, 0., 0., 0.00539568,
                        0., 0., 0., ],
                       [0., 0., 0.00340716, 0., 0.00170358, 0.,
                        0., 0.00340716, 0., 0., 0.00340716, 0.,
                        0., 0., 0., 0., 0.04258944, 0.,
                        0., 0.00170358, 0., 0., 0., 0.,
                        0., 0., 0., 0.01192504, 0., 0.,
                        0.00340716, 0., 0., 0., 0.00340716, 0.,
                        0., 0., 0., 0.92504259, 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0.00188679, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.00188679, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.00188679, 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.00188679, 0., 0.99245283, 0.,
                        0., 0., 0., ],
                       [0.00175131, 0., 0., 0.01225919, 0., 0.,
                        0., 0.00175131, 0., 0., 0.00175131, 0.00175131,
                        0., 0., 0.00175131, 0., 0., 0.,
                        0.00700525, 0.00175131, 0., 0., 0.00350263, 0.01225919,
                        0., 0., 0., 0.00525394, 0., 0.,
                        0.00175131, 0., 0., 0., 0., 0.,
                        0., 0., 0.00350263, 0.00350263, 0., 0.94045534,
                        0., 0., 0., ],
                       [0., 0.0016835, 0., 0., 0., 0.0016835,
                        0., 0., 0., 0., 0., 0.,
                        0.0016835, 0., 0.003367, 0.0016835, 0., 0.,
                        0.0016835, 0., 0., 0., 0., 0.,
                        0., 0.003367, 0., 0.003367, 0., 0.0016835,
                        0.0016835, 0.05723906, 0.02525253, 0., 0., 0.,
                        0., 0., 0.0016835, 0., 0., 0.,
                        0.89057239, 0., 0.003367],
                       [0.00509338, 0.00848896, 0., 0., 0., 0.00169779,
                        0., 0.00509338, 0., 0.00339559, 0.00848896, 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.01867572, 0.00169779, 0., 0., 0., 0.,
                        0., 0., 0., 0.01358234, 0., 0.00509338,
                        0.00848896, 0., 0., 0., 0., 0.,
                        0.00169779, 0., 0., 0.00339559, 0.01188455, 0.,
                        0., 0.90322581, 0., ],
                       [0., 0., 0., 0., 0., 0.00182815,
                        0., 0., 0., 0.00182815, 0., 0.,
                        0., 0.00548446, 0., 0.00182815, 0., 0.,
                        0., 0., 0.00365631, 0.07678245, 0.00365631, 0.,
                        0., 0.00365631, 0., 0., 0., 0.,
                        0., 0., 0.00548446, 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.89579525]])
print(cnf_matrix.shape)
attack_types = ['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church',
                'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
                'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island',
                'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace',
                'parking_lot', 'railway', "railway_station", "rectangular_farmland", "river", "roundabout", "runway",
                "sea_ice", "ship", "snowberg", "sparse_residential", "stadium", "storage_tank", "tennis_court",
                "terrace", "thermal_power_station", "wetland"]

plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Confusion Matrix')