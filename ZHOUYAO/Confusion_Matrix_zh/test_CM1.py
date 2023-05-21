# import numpy as np
# import matplotlib.pyplot as plt
# import itertools
# import seaborn as sns
#
#
# # best: 1.0 -> 99.87%
#
# # 绘制混淆矩阵
# def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Greys):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     Input
#     - cm : 计算出的混淆矩阵的值
#     - classes : 混淆矩阵中每一行每一列对应的列
#     - normalize : True:显示百分比, False:显示个数
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=5)
#     plt.yticks(tick_marks, classes, fontsize=5)
#
#     sns.set(font_scale=0.4)  # 调整混淆矩阵中字体的大小
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  verticalalignment='center',
#                  horizontalalignment='center',
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.gcf().subplots_adjust(bottom=0.3)
#     plt.tight_layout()
#     # plt.savefig("../plot/test_CM4.png", dpi=500, format="png", bbox_inches='tight')
#     plt.show()
#
#
# # 效果太好，需改一下
# cnf_matrix = np.array([[10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0.3, 0., 9.2, 0., 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0., 0., 0., 0.,],
#                        [0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 0., ],
#                        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., ]])
#
# attack_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
#
# plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True)


import matplotlib

print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())
