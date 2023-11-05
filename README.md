# AlexNet_model
Use a model named as AlexNet to classify some flowers
这个项目是使用AlexNet模型给一些花进行分类
①首先数据集下载网站：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz,数据集存放至flower_data文件夹中
②运行split_data.py对数据集进行分割，分为训练集和验证集
③运行model.py,是AlexNet的模型构建
④train.py是使用训练集对模型进行训练，并且可以看到每个epoch的训练误差，以及在验证集上的准确度
⑤predict.py使用一张花的图片进行类别预测
