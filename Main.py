import matplotlib.pyplot as plt

RandomForest_K_folds = [0.9864864864864865,0.9806949806949807,0.9787234042553191,0.97678916827853,0.9729206963249516,0.9845261121856866,0.9729206963249516,0.9806576402321083,0.9574468085106383,0.9574468085106383]

box_plot_data = [RandomForest_K_folds]
plt.boxplot(box_plot_data)
plt.show()
