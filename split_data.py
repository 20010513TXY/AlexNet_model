import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()#获取当前工作目录的路径
    data_root = os.path.join(cwd, "flower_data")
    print("data_root="+data_root)
    origin_flower_path = os.path.join(data_root, "flower_photos")
    print("origin_flower_path="+origin_flower_path)#origin_flower_path=E:\研\研究生\code\AlexNet\flower_data\flower_photos
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)
    #assert 关键字用于检查条件是否为 True。如果条件为 True，则程序继续执行。如果条件为 False，assert 将引发 AssertionError 异常。
    #后面部分是当出错时，报错的信息
    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]
    for cla in os.listdir(origin_flower_path):
        print(cla)
        print(os.path.isdir(os.path.join(origin_flower_path, cla)))
    #os.listdir():列出指定目录中的所有文件和子目录的名称  os.path.isdir():检查给定路径是否是一个目录
    # 建立保存训练集的文件夹
    #data_root=E:\研\研究生\code\AlexNet\flower_data
    train_root = os.path.join(data_root, "train")
    print(train_root)#train_root=E:\研\研究生\code\AlexNet\flower_data\train
    mk_file(train_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))#创建训练集中的文件夹-文件夹名即为类名

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in flower_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)#每类图片数量
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        #random.sample：从序列中随机选择指定数量的元素，而不重复选择相同的元素
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_root, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()