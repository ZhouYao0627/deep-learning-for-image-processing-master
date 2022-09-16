import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


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
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig("../plot/UCM21_8020_resnext_101_1.png", dpi=500, format="png", bbox_inches='tight')
    plt.show()


cnf_matrix = np.array([[1., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 1., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 1., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 1., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 1., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0.95238095, 0., 0., 0., 0., 0.,
                        0.04761905, 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 1., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 1., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 1., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 1., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 1.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0.95, 0., 0., 0., 0., 0.,
                        0.05, 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 1., 0., 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0.04761905, 0., 0., 0.,
                        0., 0., 0.95238095, 0., 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.04761905,
                        0., 0., 0., 0.95238095, 0., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 1., 0.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 1.,
                        0., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        1., 0., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 1., 0., ],
                       [0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0.,
                        0., 0., 1.]])
attack_types = ['Airport', 'BareLand', 'BaseballField', 'Beach', 'Bridge', 'Center', 'Church', 'Commercial',
                'DenseResidential', 'Desert', 'Farmland', 'Forest', 'Industrial', 'Meadow', 'MediumResidential',
                'Mountain', 'Park', 'Parking', 'Playground', 'Pond', 'Port']

plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='Confusion Matrix')
