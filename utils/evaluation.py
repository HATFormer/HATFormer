import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score


def getHist(img,num_bins):
    hist, bins = np.histogram(img, bins=np.linspace(-25, 70,num_bins+1))
    return hist, bins

def drawFig_backup(hist1,hist2,bins,img_path,num_bins):
    f, ax = plt.subplots()
    plt.tight_layout()
    #import pdb;pdb.set_trace()
    ax.bar(bins[:-1], hist1, width=110/num_bins, color='blue', alpha=0.8, label='GroudTruth')
    ax.bar(bins[:-1], hist2, width=110/num_bins, color='yellow', alpha=0.8, label='Prediction')
    ax.set_xlabel("Height Value")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    #ax.set_title("Accumulated Pixel Value Distribution")
    ax.set_ylim([0, 300000])
    f.savefig(img_path.replace('png','svg'), dpi=300, pad_inches=0.05,
              bbox_inches='tight', format='svg')

def drawFig(hist1,hist2,bins,img_path,num_bins, ground_truth_heights, predicted_heights):
    # f, ax = plt.subplots()
    # plt.tight_layout()

    # 设置图形布局
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    # 主密度散点图
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    sns.kdeplot(
        x=ground_truth_heights,
        y=predicted_heights,
        cmap="viridis",
        fill=True,
        alpha=0.3,  # 设置透明度
        thresh=0.05,
        ax=main_ax
    )
    # 在密度图上叠加散点
    main_ax.scatter(ground_truth_heights, predicted_heights, color="blue", s=5, alpha=0.5)

    # 添加对角线 y = x
    main_ax.plot([-30, 30], [-35, 35], 'k--', linewidth=1)
    main_ax.set_xlabel("Ground Truth Height")
    main_ax.set_ylabel("Predicted Height")

    # 固定主图的横轴和纵轴范围为 -40 到 40
    main_ax.set_xlim(-30, 35)
    main_ax.set_ylim(-30, 35)

    # 在主图上添加颜色条
    # cbar = plt.colorbar(main_ax.collections[0], ax=main_ax, orientation="vertical", fraction=0.05)
    # cbar.set_label("Density")

    # 顶部的分布叠加图
    top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    sns.kdeplot(ground_truth_heights, color="green", ax=top_ax, label="Ground Truth heights")
    sns.kdeplot(predicted_heights, color="purple", ax=top_ax, label="Predicted heights")
    top_ax.legend()
    top_ax.set_ylabel("Frequency")
    top_ax.get_xaxis().set_visible(False)  # 隐藏顶部图的 x 轴标签

    # 设置顶部分布叠加图的纵轴范围为 0 到 0.03
    top_ax.set_ylim(0, 0.15)

    # 在主图上显示 R^2 值
    r2 = r2_score(ground_truth_heights, predicted_heights)
    main_ax.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', transform=main_ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    fig.savefig(img_path.replace('svg','pdf'), dpi=300, pad_inches=0.05,
              bbox_inches='tight', format='pdf')
def drawFig_smars(hist1,hist2,bins,img_path,num_bins, ground_truth_heights, predicted_heights):
    # f, ax = plt.subplots()
    # plt.tight_layout()

    # 设置图形布局
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    # 主密度散点图
    main_ax = fig.add_subplot(grid[1:4, 0:3])
    sns.kdeplot(
        x=ground_truth_heights,
        y=predicted_heights,
        cmap="viridis",
        fill=True,
        alpha=0.3,  # 设置透明度
        thresh=0.05,
        ax=main_ax
    )
    # 在密度图上叠加散点
    main_ax.scatter(ground_truth_heights, predicted_heights, color="blue", s=5, alpha=0.5)

    # 添加对角线 y = x
    main_ax.plot([-45, 45], [-45, 45], 'k--', linewidth=1)
    main_ax.set_xlabel("Ground Truth Height")
    main_ax.set_ylabel("Predicted Height")

    # 固定主图的横轴和纵轴范围为 -40 到 40
    main_ax.set_xlim(-45, 45)
    main_ax.set_ylim(-45, 45)

    # 在主图上添加颜色条
    # cbar = plt.colorbar(main_ax.collections[0], ax=main_ax, orientation="vertical", fraction=0.05)
    # cbar.set_label("Density")

    # 顶部的分布叠加图
    top_ax = fig.add_subplot(grid[0, 0:3], sharex=main_ax)
    sns.kdeplot(ground_truth_heights, color="green", ax=top_ax, label="Ground Truth heights")
    sns.kdeplot(predicted_heights, color="purple", ax=top_ax, label="Predicted heights")
    top_ax.legend()
    top_ax.set_ylabel("Frequency")
    top_ax.get_xaxis().set_visible(False)  # 隐藏顶部图的 x 轴标签

    # 设置顶部分布叠加图的纵轴范围为 0 到 0.03
    top_ax.set_ylim(0, 0.04)

    # 在主图上显示 R^2 值
    r2 = r2_score(ground_truth_heights, predicted_heights)
    main_ax.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', transform=main_ax.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    fig.savefig(img_path.replace('svg','pdf'), dpi=300, pad_inches=0.05,
              bbox_inches='tight', format='pdf')
def accHist(root):
    total_hist = np.zeros(110)
    img_list = os.listdir(root)
    for img_name in img_list:
        img_path = os.path.join(root, img_name)
        img = tiff.imread(img_path)
        mask = tiff.imread(img_path.replace('3d','2d'))
        if mask.sum()>0:
            hist, bins = getHist(img[mask!=0])
            total_hist += hist

    return total_hist, bins

if __name__ == '__main__':
    total_hist = np.zeros(110)
    roots = ['/home/liub/data/amsterdam_025/test/mask3d_1k',
             '/home/liub/data/rotterdam_025/test/mask3d_1k',
             '/home/liub/data/utrecht_025/test/mask3d_1k']
    
    for root in roots:
        hist, bins = accHist(root)
        total_hist += hist
    f, ax = plt.subplots()
    plt.tight_layout()

    ax.bar(bins[:-1], total_hist, width=1)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Accumulated Pixel Value Distribution")
    f.savefig('./utils/GTdistribution.png', dpi=300, pad_inches=0.05, bbox_inches='tight')
