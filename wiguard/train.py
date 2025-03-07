import torch
import logging
import sys
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from ptflops import get_model_complexity_info

from wiguard.dataset import CSIDataset, MIX
from wiguard.model.Transformer import Transformer
# from wiguard import config

torch.manual_seed(0)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))

SUBCARRIES = 64  # 子载波数
LABELS_NUM = 3
EPOCHS_NUM = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 8
ENC_SEQ_LEN = 6  # 编码器序列长度
DEC_SQL_LEN = 4  # 解码器序列长度
DIM_VAL = 32  # value的维度
DIM_ATTN = 8  # attention的维度
N_HEADS = 4  # 多头注意力的头数
N_ENCODER_LAYERS =4  # 编码器层数
N_DECODER_LAYERS = 4  # 解码器层数
WEIGHT_DECAY = 1e-4  # 权重衰减




model = Transformer(dim_val=DIM_VAL,
                    dim_attn=DIM_ATTN,
                    input_size=SUBCARRIES,
                    dec_seq_len=DEC_SQL_LEN,
                    out_seq_len=LABELS_NUM,
                    n_decoder_layers=N_DECODER_LAYERS,
                    n_encoder_layers=N_ENCODER_LAYERS,
                    n_heads=N_HEADS)
model = model.float().to(device)



def test_train():

    # 日志文件
    log_dir = os.path.join('./logs', sys.argv[1])
    # 准备数据集
    if MIX: data_path = './data/train'
    else: data_path = os.path.join('./data/train', sys.argv[2])
    print(data_path)
    csi_dataset = CSIDataset(data_path)
    # print(len(csi_dataset))
    total_size = len(csi_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    # print(train_size, val_size)
    train_dataset, val_dataset = random_split(
        csi_dataset, [train_size, val_size])  # 分割数据集

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # print(len(train_loader), len(val_loader))
    logging.info("Train size: {}, val size: {}".format(
        len(train_loader), len(val_loader)))

    # Loss Function CrossEntropy
    loss_fn = CrossEntropyLoss()

    # Optimizer AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # outputs: (batch_size, num_label)

    writer = SummaryWriter(log_dir)

    total_train_step = 0
    total_val_step = 0
    for epoch in range(EPOCHS_NUM):
        model.train()
        total_train_loss = 0
        logging.info("Epoch: {}".format(epoch))
        for data, label in train_loader:
            data = data.float().to(device)
            label =label.float().to(device)

            # print(data.shape, label.shape)
            # print(model(data.to(device)))

            outputs = model(data)
            # print(pred)
            loss = loss_fn(outputs, label.long()).float()
            total_train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1

        model.eval()
        total_valid_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data, label in val_loader:
                data = data.float().to(device)
                label = label.float().to(device)
                outputs = model(data)
                # print(outputs, label)
                total_accuracy += outputs.argmax(dim=1).eq(label).sum()
                valid_loss = loss_fn(outputs, label.long()).float()
                total_valid_loss += valid_loss

        print("step: {}".format(total_train_step),
              "train loss: {}".format(total_train_loss/len(train_loader)))
        print("step: {}".format(total_train_step),
              "valid loss: {}".format(total_valid_loss/len(val_loader)))
        print("step: {}".format(total_train_step),
              "accuracy: {}".format(total_accuracy/val_size))

        writer.add_scalar('Loss/train', total_train_loss /
                          len(train_loader), epoch)
        writer.add_scalar('Loss/val', total_valid_loss/len(val_loader), epoch)
        writer.add_scalar('Accuracy/val', total_accuracy/val_size, epoch)

        if KeyboardInterrupt: 
            model_path = os.path.join('./models/', sys.argv[1]) + '.pth'
            torch.save(model.state_dict(), model_path)
        


    model_path = os.path.join('./models/', sys.argv[1]) + '.pth'
    torch.save(model.state_dict(), model_path)

def flops():
    flops, params = get_model_complexity_info(model, ( 80, 64), as_strings=True, print_per_layer_stat=True) 
    print('flops: ', flops, 'params: ', params)

if __name__ == '__main__':

    # 预测单个文件
    # test_predict_file(sys.argv[1])

    # 测试模型复杂度
    # flops()

    # 训练模型
    test_train() 

    # 测试模型
    # test_predict()
