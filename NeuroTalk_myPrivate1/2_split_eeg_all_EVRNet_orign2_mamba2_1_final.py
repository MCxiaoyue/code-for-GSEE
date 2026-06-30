import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tqdm import tqdm
import pywt
import matplotlib.pyplot as plt
from kymatio import Scattering1D
from mamba.mamba import Mamba
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义各个模块
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpatialBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class MKRB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MKRB, self).__init__()

        # First part (3x3 conv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Second part (5x5 conv)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # 第一个卷积层提取特征向量 y0
        y0 = self.conv1(x)

        # 将 y0 与输入 x 融合
        fused = y0 + x

        fused = self.relu(fused)

        # 第二个卷积层进一步提取特征
        y1 = self.conv2(fused)

        # 输出最终的特征向量
        output = y1 + fused  # 残差连接

        return self.relu(output)


class EVRNet(nn.Module):
    def __init__(self, num_classes):
        super(EVRNet, self).__init__()
        self.mamba = Mamba(
            num_layers=1,  # Number of layers of the full model
            d_input=32,  # Dimension of each vector in the input sequence (i.e. token size)
            d_model=8,  # Dimension of the visible state space
            d_state=8,  # Dimension of the latent hidden states
            d_discr=16,  # Rank of the discretization matrix Δ
            ker_size=4,  # Kernel size of the convolution in the MambaBlock
            parallel=False,  # Whether to use the sequenial or the parallel implementation
        )
        self.spatial_block1 = SpatialBlock(1, 32, kernel_size=3, stride=2)
        # self.temporal_block2 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb1 = MKRB(32, 32)

        self.temporal_block1 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        # self.spatial_block2 = SpatialBlock(32, 32, kernel_size=3, stride=2)
        self.mkrb2 = MKRB(32, 32)

        self.temporal_block3 = TemporalBlock(32, 32, kernel_size=3, stride=2)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层

        self.dropout = nn.Dropout(0.1)  # 定义一个Dropout层

        self.fc_layer = nn.Linear(32, num_classes)  # 调整线性的输入大小


    def forward(self, x):
        # x = x.permute(0, 1, 3, 2)
        x = self.spatial_block1(x)  # (batch_size, 32, 96, 24)
        # x = self.temporal_block2(x)  # (batch_size, 32, 96, 24)
        x = self.mkrb1(x)  # (batch_size, 128, 96, 24)

        x = self.temporal_block1(x)  # (batch_size, 256, 48, 12)
        # x = self.spatial_block2(x)  # (batch_size, 512, 24, 6)
        x = self.mkrb2(x)  # (batch_size, 512, 48, 12)

        x = self.temporal_block3(x)

        x = self.avg_pooling(x)  # (batch_size, 512, 1, 1)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.dropout(x)  # 应用Dropout层

        x = x.unsqueeze(-2)

        x = self.mamba(x)[0]

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc_layer(x)

        return x


# 手动规定的单词到标签映射
word_to_label = {
    "my": 8,
    "dad": 0,
    "is": 1,
    "a": 2,
    "policeman": 3,
    "he": 4,
    "will": 5,
    "always": 6,
    "become": 7,
    "hero": 9
}

# 处理单个文件的函数
def process_file(file_path, image_data, labels):
    with open("./orign_eeg_data/"+file_path, mode='r') as f:
        lines = f.readlines()
        ls = []
        # 采集2560行
        for line in lines[1282:52844]:  # 前三行没用，所以从3开始，每秒128帧
            line_list = line.strip().split('\t')  # 再分割成列表
            # 采集24列中的特定几列
            columns = []
            for linetime in range(1, 25):  # [5, 6, 13, 14]
                columns.append(float(line_list[linetime]))
            ls.append(columns)  # 提取指定列并转换为浮点数

        eeg = np.array(ls)

        # 定义参数
        word_duration = 1.5  # 单词时长（秒）
        rest_within_word = 5  # 同一单词间的静音间隔（秒）
        rest_between_words = 10  # 不同单词间的静音间隔（秒）
        sampling_rate = 128  # 采样率（Hz）

        # 单词序列
        words = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "my", "hero"]
        repetitions = 5  # 每个单词重复次数

        # 计算每个单词和静音周期的采样点数量
        samples_per_word = int(word_duration * sampling_rate)
        samples_per_rest_within_word = int(rest_within_word * sampling_rate)
        samples_per_rest_between_words = int(rest_between_words * sampling_rate)

        # 分割数据
        word_data_segments = []
        current_index = 0
        for word in words:
            for _ in range(repetitions):
                # 添加单词部分
                word_start = current_index
                word_end = word_start + samples_per_word
                word_segment = eeg[word_start:word_end, :]
                word_data_segments.append(word_segment)

                # 分配标签
                label = word_to_label[word]  # 使用单词到标签的映射来分配标签
                labels.append(label)

                # 更新索引到下一个单词或静音部分
                if _ < repetitions - 1:
                    current_index += samples_per_word + samples_per_rest_within_word
                else:
                    current_index += samples_per_word + samples_per_rest_between_words

        for eeg in word_data_segments:
            # print(eeg.shape)

            # 对每个通道进行归一化
            scaler = StandardScaler()  # MinMaxScaler() RobustScaler() StandardScaler()
            segment_normalized = scaler.fit_transform(eeg.T).T  # 注意这里转置两次是为了保持原来的shape

            arr = segment_normalized
            image_data.append(arr.T)

            # # 不进行归一化
            # image_data.append(eeg.T)




# 定义一个函数来绘制并保存loss曲线图
def plot_and_save_loss_curves(train_loss_history, test_loss_history, filename='loss_curves.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(test_loss_history, label='Testing Loss')
    # plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset

# 固定所有随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置GPU的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    random.seed(seed)  # Python内置的随机模块种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回相同的结果

seed = 42
setup_seed(seed)

# 对于每个记录和每个通道，应用小波变换
def apply_wavelet_transform(data, wavelet='db4', level=1):
    """
    对输入的三维数据执行小波变换。

    参数:
    - data: 形状为(n_samples, n_channels, n_times)的numpy数组
    - wavelet: 使用的小波类型，默认'db4'
    - level: 分解尺度，默认1

    返回:
    - coeffs_array: 经过小波变换后的系数数组，形状与输入相同
    """
    n_samples, n_channels, n_times = data.shape
    coeffs_array = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_channels):
            # 获取单个记录和通道的数据
            signal = data[i, j, :]

            # 执行小波分解
            coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

            # 如果需要将所有系数重构回原始长度，可以使用以下方法
            # 注意：这一步不是必须的，取决于你的应用场景
            reconstructed_signal = pywt.waverec(coeffs, wavelet=wavelet)[:n_times]

            # 将重构后的信号或系数存入数组中
            # 这里我们选择存储重构后的信号，但根据需要也可以存储coeffs
            coeffs_array[i, j, :] = reconstructed_signal

    return coeffs_array

def plot_eeg_data_together(eeg_data, sample_index=0):
    """
    将给定索引的样本的所有通道的EEG数据绘制在同一张图上。

    参数:
        eeg_data: 包含EEG数据的numpy数组，形状为(samples, channels, timestamps)
        sample_index: 要可视化的样本的索引
    """
    eeg_sample = eeg_data[sample_index]
    num_channels = eeg_sample.shape[0]

    plt.figure(figsize=(15, 8))

    # 绘制每个通道的数据
    for channel_idx in range(num_channels):
        plt.plot(eeg_sample[channel_idx], label=f'Channel {channel_idx + 1}')

    plt.title(f'Sample {sample_index} - All Channels')
    plt.xlabel('Timestamp')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# 对于每个记录和每个通道，应用小波变换
def apply_wavelet_transform(data, wavelet='db4', level=2):
    """
    对输入的三维数据执行小波变换。

    参数:
    - data: 形状为(n_samples, n_channels, n_times)的numpy数组
    - wavelet: 使用的小波类型，默认'db4'
    - level: 分解尺度，默认1

    返回:
    - coeffs_array: 经过小波变换后的系数数组，形状与输入相同
    """
    n_samples, n_channels, n_times = data.shape
    coeffs_array = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_channels):
            # 获取单个记录和通道的数据
            signal = data[i, j, :]

            # 执行小波分解
            coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

            # 如果需要将所有系数重构回原始长度，可以使用以下方法
            # 注意：这一步不是必须的，取决于你的应用场景
            reconstructed_signal = pywt.waverec(coeffs, wavelet=wavelet)[:n_times]

            # 将重构后的信号或系数存入数组中
            # 这里我们选择存储重构后的信号，但根据需要也可以存储coeffs
            coeffs_array[i, j, :] = reconstructed_signal

    return coeffs_array

def apply_wavelet_packet_transform(data, wavelet='db4', max_level=2):
    """
    对输入的三维数据执行小波包变换。

    参数:
    - data: 形状为(n_samples, n_channels, n_times)的numpy数组
    - wavelet: 使用的小波类型，默认'db4'
    - max_level: 分解的最大尺度，默认1

    返回:
    - coeffs_array: 经过小波包变换后的重构信号或系数数组，形状与输入相同
    """
    n_samples, n_channels, n_times = data.shape
    coeffs_array = np.zeros_like(data)

    for i in range(n_samples):
        for j in range(n_channels):
            # 获取单个记录和通道的数据
            signal = data[i, j, :]

            # 创建小波包对象
            wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)

            # 执行小波包分解
            wp.decompose()

            # 重构原始长度的信号
            # 注意：这里我们选择重构回原始信号，但也可以根据需要提取特定节点的系数
            reconstructed_signal = wp.reconstruct(update=False)[:n_times]

            # 将重构后的信号存入数组中
            coeffs_array[i, j, :] = reconstructed_signal

    return coeffs_array


def apply_wavelet_scattering_transform(data, J=2, Q=1):
    """
    对输入的三维数据执行小波散射变换。

    参数:
    - data: 形状为(n_samples, n_channels, n_times)的numpy数组
    - J: 散射变换的最大尺度参数，默认值为2
    - Q: 小波滤波器组的品质因数，默认值为1

    返回:
    - scattering_coeffs_array: 经过小波散射变换后的系数数组，形状可能与输入不同但保持三维
    """
    n_samples, n_channels, n_times = data.shape

    # 初始化Scattering1D对象
    scattering = Scattering1D(J=J, shape=(n_times,), Q=Q)

    scattering_coeffs_list = []

    for i in range(n_samples):
        sample_coeffs_list = []
        for j in range(n_channels):
            signal = data[i, j, :]

            # 将信号转换为torch张量并添加批量维度
            signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)

            # 执行小波散射变换
            scattering_coeffs_tensor = scattering(signal_tensor)

            # 移除批量维度，并将结果转换为numpy数组
            scattering_coeffs = scattering_coeffs_tensor.squeeze(0)  # .cpu().numpy()

            # 如果需要，可以对散射系数进行处理（例如展平）
            # 这里我们假设展平所有散射系数以便于后续处理
            flattened_coeffs = scattering_coeffs.reshape(-1)

            # 保存当前通道的结果
            sample_coeffs_list.append(flattened_coeffs)

        # 合并当前样本的所有通道的结果
        # 确保每个样本的每个通道的散射系数具有相同的长度
        max_length = max([coeffs.shape[0] for coeffs in sample_coeffs_list])
        padded_sample_coeffs_list = [np.pad(coeffs, (0, max_length - coeffs.shape[0]), 'constant')
                                     for coeffs in sample_coeffs_list]
        sample_coeffs_array = np.stack(padded_sample_coeffs_list, axis=0)
        scattering_coeffs_list.append(sample_coeffs_array)

    # 合并所有样本的结果
    scattering_coeffs_array = np.stack(scattering_coeffs_list, axis=0)

    return scattering_coeffs_array


if __name__ == '__main__':
    # 在循环外部初始化列表
    train_losses = []
    test_losses = []

    # 文件路径列表
    file_paths = ['vDLY-001_1.txt', 'vLCL-001_1.txt', 'vLHJ-001_1.txt', 'vLHJ-002_1.txt', 'vLCL-004_1.txt',
                  'vYHC02_1.txt', 'vLCL-01_1.txt', 'vLCL-02_1.txt', 'vCLB-001_1.txt', 'vYHC-001_1.txt', 'vLCL-003_1.txt']

    # 创建一个空列表来存储转换后的图像数据和对应的标签
    image_data = []
    labels = []
    # 处理所有文件
    for file_path in file_paths:
        process_file(file_path, image_data, labels)

    # 转换为numpy数组
    image_data_array = np.array(image_data)
    labels_array = np.array(labels)

    print(image_data_array.shape)
    print(labels_array.shape)


    # 测试数据的索引
    test_indices = [i for i in range(0, image_data_array.shape[0], 10)]  # 第5, 55, 105, 155个数据（基于0索引） # 114, 165, 5
    print(test_indices)

    # 切分数据
    train_data = np.delete(image_data_array, test_indices, axis=0)
    test_data = image_data_array[test_indices]

    # 切分标签
    train_labels = np.delete(labels_array, test_indices, axis=0)
    test_labels = labels_array[test_indices]

    # 转换为Tensor
    train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 192, 24)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 192, 24)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    print(test_labels)

    # 创建数据加载器
    torch.manual_seed(42)  # 设置固定的随机种子
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    # # 应用小波变换
    # image_data_array = apply_wavelet_transform(image_data_array)

    # # 应用小波包变换
    # image_data_array = apply_wavelet_packet_transform(image_data_array)

    # # 应用小波扩散变换
    # image_data_array = apply_wavelet_scattering_transform(image_data_array)

    # 使用函数可视化第一个样本的EEG数据
    plot_eeg_data_together(image_data_array)

    # 初始化模型、损失函数和优化器
    num_classes = len(word_to_label.keys())
    model = EVRNet(num_classes=num_classes).to(device)  # 将模型移动到 GPU
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移动到 GPU
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # betas=(0.95, 0.999), eps=1e-08

    # 训练模型
    num_epochs = 2000
    best_test_acc = 0.0  # 用于记录最佳测试集准确率

    for epoch in range(num_epochs):
        train_accuracies = []
        test_accuracies = []

        all_train_targets = []
        all_train_predictions = []
        all_test_targets = []
        all_test_predictions = []
        batch_train_f1_scores = []
        batch_test_f1_scores = []

        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # 训练部分
        for inputs, targets in tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]'):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)  # 获取预测概率
            # print(targets.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 计算训练集上的正确预测数
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            # 计算每个批次的准确率，并添加到列表中
            batch_accuracy = 100 * correct_train / total_train
            train_accuracies.append(batch_accuracy)

            all_train_targets.extend(targets.cpu().numpy())
            all_train_predictions.extend(predicted.cpu().numpy())
            # 计算并保存当前批次的F1-score
            batch_f1 = 100 * f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            batch_train_f1_scores.append(batch_f1)

        # 计算训练集上的准确率
        train_accuracy = 100 * correct_train / total_train
        # 在你的训练循环内，计算完running_loss之后添加这行代码：
        train_losses.append(running_loss / len(train_loader))
        # 在计算完测试集上的准确率之后，添加如下代码以记录测试loss:
        test_loss = 0
        # 测试部分
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()

                # 计算每个批次的准确率，并添加到列表中
                batch_accuracy = 100 * correct_test / total_test
                test_accuracies.append(batch_accuracy)
                all_test_targets.extend(targets.cpu().numpy())
                all_test_predictions.extend(predicted.cpu().numpy())
                # 计算并保存当前批次的F1-score
                batch_f1 = 100 * f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
                batch_test_f1_scores.append(batch_f1)

                loss = criterion(outputs, targets)  # 计算测试集的loss
                test_loss += loss.item() * inputs.size(0)

        # 计算测试集上的准确率
        test_accuracy = 100 * correct_test / total_test
        test_loss /= len(test_loader.dataset)  # 平均loss
        test_losses.append(test_loss)

        # 如果当前测试集准确率优于之前的最佳准确率，则保存模型
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            best_model_weights = model.state_dict()  # 保存当前模型权重
            # 立即保存最佳模型权重到文件
            torch.save(best_model_weights, 'classification_checkpoint/best_model_weights.pth')
            print(f'Saved new best model with Test Acc: {best_test_acc:.2f}%')  # 可选：打印保存信息


        # 转换为numpy数组并计算标准差
        if len(train_accuracies) > 1:
            train_accuracy_std = np.std(train_accuracies, ddof=1)  # ddof=1 表示样本标准差
            test_accuracy_std = np.std(test_accuracies, ddof=1)
        else:
            train_accuracy_std = 0  # 避免除以零的情况
            test_accuracy_std = 0  # 避免除以零的情况

        # 在epoch结束时计算F1-score
        train_f1 = 100 * f1_score(all_train_targets, all_train_predictions, average='weighted')  # 使用'weighted'或其他适合的方式
        f1_train_std = np.std(batch_train_f1_scores)  # 计算F1-score的标准差
        test_f1 = 100 * f1_score(all_test_targets, all_test_predictions, average='weighted')  # 使用'weighted'或其他适合的方式
        f1_test_std = np.std(batch_test_f1_scores)  # 计算F1-score的标准差
        # 打印结果
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Loss: {running_loss / len(train_loader):.4f}, '
              f'Train Acc: {train_accuracy:.2f}% ± {train_accuracy_std:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}% ± {test_accuracy_std:.2f}%, '
              f'Train F1: {train_f1:.2f} ± {f1_train_std:.2f}%, '
              f'Test F1: {test_f1:.2f} ± {f1_test_std:.2f}%, '
              )


    model = EVRNet(num_classes=num_classes).to(device)  # 将模型移动到 GPU
    # 训练完成后，使用最佳模型权重更新模型
    model.load_state_dict(torch.load('classification_checkpoint/best_model_weights.pth'))
    model.eval()

    print('Training complete and best weights saved.')

    # 测试模型
    correct_test = 0
    total_test = 0
    all_predicted_labels = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing Final Model'):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()

            # 收集预测标签和真实标签
            all_predicted_labels.extend(predicted.cpu().numpy())
            all_true_labels.extend(targets.cpu().numpy())

            probabilities = torch.softmax(outputs, dim=1)  # 获取预测概率
            all_probabilities.extend(probabilities.cpu().numpy())


    # 计算最终测试集上的准确率
    final_test_accuracy = 100 * correct_test / total_test
    print(f'Final Test Accuracy with Best Weights: {final_test_accuracy:.2f}%')

    # 绘制ROC曲线
    n_classes = len(np.unique(all_true_labels))
    # 定义10种颜色以区分10个类别
    colors = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'brown']

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(all_true_labels) == i, np.array(all_probabilities)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class')
    plt.legend(loc="lower right")

    # 保存图像文件
    plt.savefig('classification_checkpoint/best_model_roc.png')  # 保存图像文件
    plt.show()  # 显示图像

    # 生成混淆矩阵数据
    cm = confusion_matrix(all_true_labels, all_predicted_labels)

    # 创建反向映射，即从数值标签到单词的映射
    label_to_word = {v: k for k, v in word_to_label.items()}

    # 获取排序后的标签列表（确保与混淆矩阵的顺序匹配）
    labels = [label_to_word[i] for i in range(len(word_to_label))]

    # 将混淆矩阵转换为百分比形式
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # 创建一个带百分号标注的数组用于图表显示
    annot = np.array([[f"{v:.2f}%" for v in row] for row in cm_percent])

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=annot, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.title('Confusion Matrix with Word Labels (Percentage)')

    # 旋转x轴标签以提高可读性
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 调整布局，防止标签被裁剪
    plt.tight_layout()

    # 保存图像文件
    plt.savefig('classification_checkpoint/confusion_matrix_with_words_percentage.png')  # 保存图像文件
    plt.show()  # 显示图像

    # 打印预测标签和真实标签
    for i in range(len(all_predicted_labels)):
        print(f'Predicted: {all_predicted_labels[i]}, True: {all_true_labels[i]}')
        print('======================================')

    # 在训练完成后调用此函数
    plot_and_save_loss_curves(train_losses, test_losses, filename='classification_checkpoint/loss_curves.png')

    print("Loss curves have been saved.")