import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import itertools
import seaborn as sns


# best: 0.964
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Greys):
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
    # print(cm)

    # fig = plt.figure(figsize=(6.4, 4.8))

    # plt.rc('font', family='Times New Roman')  # 设置字体

    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=-45, fontsize=7, ha="left", )  # , ha="left",
    plt.yticks(tick_marks, classes, fontsize=7, )

    # 修改刻度线的宽度与长度
    plt.tick_params(which='major', width=0.5)
    plt.tick_params(which='major', length=0.6)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')

    # 加上或去掉刻度线
    plt.tick_params(top=True, bottom=True, left=True, right=True)

    sns.set(font_scale=0.37)  # 调整混淆矩阵中字体的大小
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i][j] < 0.01:
            plt.text(j, i, '')
        else:
            plt.text(j, i, format(cm[i, j], fmt),
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()
    # fig.savefig("../plot/test_CM1.png", dpi=600, format='png', bbox_inches='tight')
    # plt.savefig("../plot/test_CM9.png", dpi=600, format='png', bbox_inches='tight')
    # plt.savefig("../plot/test_CM4.pdf", dpi=600, format='pdf', bbox_inches='tight')


cnf_matrix = np.array([[0.97260274, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01369863, 0., 0., 0., 0., 0.01369863, 0., 0., 0., ],
                       [0., 0.95384615, 0., 0., 0., 0., 0., 0., 0., 0.03076923, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01538462, 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0.97674419, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.02325581, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0.01369863, 0.98630137, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.97826087, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.02173913, 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.04081633, 0.93877551, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.02040816, 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.01428571, 0., 0.92857143, 0.02857143, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01428571, 0., 0., 0., 0.01428571, 0., ],
                       [0., 0., 0., 0., 0., 0., 0.01234568, 0., 0.95061728, 0., 0., 0., 0., 0., 0.01234568, 0., 0., 0., 0., 0., 0., 0.01234568, 0., 0., 0.01234568, 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0.01315789, 0., 0., 0., 0., 0., 0., 0., 0.01315789, 0., 0.97368421, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.02469136, 0., 0.01234568, 0., 0., 0., 0., 0.9382716, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01234568, 0., 0., 0.01234568, 0., 0., ],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0., 0., 0., 0.03389831, 0., 0., 0., 0., 0., 0.94915254, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01694915, 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 1., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0.02985075, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.89552239, 0.,
                        0., 0., 0., 0., 0.01492537, 0.,
                        0.02985075, 0., 0.02985075, 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.98734177,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.01265823, 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.97368421, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.02631579, 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 1., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 1., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.01851852,
                        0., 0., 0., 0., 0., 0.,
                        0.01851852, 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.90740741, 0.01851852, 0.,
                        0., 0., 0.01851852, 0., 0., 0.01851852],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.01639344, 0., 0.01639344, 0., 0.06557377, 0.,
                        0., 0., 0., 0., 0.86885246, 0.,
                        0., 0., 0.03278689, 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.01204819, 0.,
                        0., 0., 0., 0., 0., 0.97590361,
                        0., 0., 0.01204819, 0., 0., 0., ],
                       [0., 0., 0.03030303, 0., 0., 0.,
                        0., 0.01515152, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.06060606, 0.,
                        0., 0., 0., 0., 0.01515152, 0.,
                        0.83333333, 0., 0.04545455, 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0.01612903, 0., 0., 0., 0., 0.01612903,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.01612903, 0.,
                        0., 0.9516129, 0., 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0.01818182, 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0.98181818, 0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 1., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.01388889,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0.98611111, 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0.01190476, 0., 0.,
                        0., 0., 0., 0., 0., 0.98809524]])

# print(cnf_matrix.shape)
attack_types = ['Airp', 'Bare', 'Base', 'Beach', 'Bridge', 'Center', 'Church', 'Commer', 'Dense R', 'Desert',
                'Farm', 'Forest', 'Indus', 'Meadow', 'Medium R', 'Mount', 'Park', 'Parking', 'Play', 'Pond',
                'Port', 'Rail Sta', 'Resort', 'River', 'School', 'Sparse R', 'Square', 'Stadium', 'Storage',
                'Viaduct']

plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True)
