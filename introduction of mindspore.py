import argparse
from mindspore import context

parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'])

args = parser.parse_known_args()[0]
context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

# mode=context.GRAPH_MODE这个模式设置分为动态图和静态图
# 动态图   PYNATIVE_MODE 静态图   GRAPH_MODE

import os
import requests

requests.packages.urllib3.disable_warnings()

def download_dataset(dataset_url, path): # 将数据集进行url方式下载
    filename = dataset_url.split("/")[-1]
    # 取用url最后一个斜杠后的内容作为文件名
    save_path = os.path.join(path, filename)
    #  "datasets/MNIST_Data/train" + "train-labels-idx1-ubyte"
    # 如果已经下载过该文件就不用执行了
    if os.path.exists(save_path):
        print("The {} file has already been downloaded and saved in the path {}".format(os.path.basename(dataset_url), path))
        return
    if not os.path.exists(path):
        os.makedirs(path)
    # 一种文件下载和写入的方式
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url), path))

train_path = "datasets/MNIST_Data/train"
test_path = "datasets/MNIST_Data/test"

download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-labels-idx1-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/train-images-idx3-ubyte", train_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-labels-idx1-ubyte", test_path)
download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte", test_path)


import mindspore.dataset as ds  # dataset模块
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype # 一种数据类型的定义函数


def create_dataset(data_path, batch_size=32, repeat_size=1, # repeat_size 数据重复个数
                   num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32    # 图片大小变为32*32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    #   output = image * rescale + shift. = 32/255+0
    rescale_op = CV.Rescale(rescale, shift)
    #   output = image * rescale + shift. = （0.1255）/0.3081- 0.1307 / 0.3081
    hwc2chw_op = CV.HWC2CHW()
    #   hwc变为chw
    type_cast_op = C.TypeCast(mstype.int32)
    # 数据类型int32

    # 使用map映射函数，将数据操作应用到数据集
    # 先对label进行处理，处理为int32数据类型，设置num-workers数量，input_columns更像是一种说明
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    # 再对图片进行处理：缩放-平移-平移-通道数变换，设置num-workers数量，input_columns更像是一种说明
    mnist_ds = mnist_ds.map(operations=[resize_op, rescale_op, rescale_nml_op, hwc2chw_op], input_columns="image", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch、repeat操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    # Dataset会取所有数据的前buffer_size数据项，填充 buffer
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    # 设置batch大小，多余的丢去
    mnist_ds = mnist_ds.repeat(count=repeat_size)


    return mnist_ds



import mindspore.nn as nn
# 对标torch.nn as nn
from mindspore.common.initializer import Normal
# 一种参数初始化的函数

class LeNet5(nn.Cell):  # nn.Module
    """
    Lenet网络结构 输出的种类数量num_class=10, 输入图片的通道数量num_channel=1
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算分为valid直接丢弃最后的pixel，same补充上以满足跟之前conv模板一样的处理，
        # pad就是两边补齐带入计算公式
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # input:32 kernel:6 stride:5 ceil[(32-6+1)/5]=6
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 实例化网络
net = LeNet5()


# 定义损失函数 交叉熵损失函数对标CROSSENTROPYLOSS
# 因为mindspore的loss返回的是一个batch大小的向量做一个mean处理就是数
# softmax_cross_entropy_with_logits传入的labels为稀疏标签，如one-hot标签[[0,0,1], [0,1,0]]；
# sparse_softmax_cross_entropy_with_logits传入的labels为非稀疏标签，
# 如三分类问题传入labels值为[2,1]，
# sparse为稀疏化的意思，即将非稀疏标签转化为稀疏标签，
# [2,1]中的2表示属于第3类，对应one-hot标签里的[0,0,1]，[2,1]中的1表示属于第2类，对应one-hot标签里的[0,1,0]；
# sparse=指定标签是否使用稀疏格式。
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')



# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)


from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
# 设置模型保存参数，1875个step一次保存模型，最多保存10个
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 应用模型保存参数，保存名字的前缀
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)


# 导入模型训练需要的库
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model

def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    print(ds_train)
    #  model.train(epoch数量，dataset类， 返回的参数【模型保存，监听函数】)， 125个step一次监听
    # dataset_sink_mode=True时，数据处理（dataset加载及其处理）会和网络计算构成Pipeline方式，
    # 即：数据处理在逐步处理数据时，处理完一个batch的数据，会把数据放到一个队列里，
    # 这个队列用于缓存已经处理好的数据，然后网络计算从这个队列里面取数据用于训练，那么此时数据处理与网络计算就Pipeline起来了，
    # 整个训练耗时就是数据处理 / 网络计算耗时最长的那个。
    # dataset_sink_mode = False时，数据处理（dataset加载及处理）会和网络计算构成串行的过程，
    # 即：数据处理在处理完一个batch后，把这个batch的数据传递给网络用于计算，在计算完成后，数据处理再处理下一个batch，
    # 然后把这个新的batch数据传递给网络用于计算，如此的循环往复，直到训练完。
    # 该方法的总耗时是数据处理的耗时 + 网络计算的耗时 = 训练总耗时。
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)




def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(os.path.join(data_path, "test"))
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("{}".format(acc))

train_epoch = 1
mnist_path = "./datasets/MNIST_Data"
dataset_size = 1
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
test_net(model, mnist_path)

