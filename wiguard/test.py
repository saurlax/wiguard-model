import torch
import logging
import sys
import os
from torch.nn import functional as F
from torch.utils.data import DataLoader
from wiguard.dataset import process_single_csv_file, CSIDataset, MIX
from wiguard.model.Transformer import Transformer
from wiguard.finetune import Build

torch.manual_seed(0)
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Device: {}".format(device))

SUBCARRIES = 64  # 子载波数
LABELS_NUM = 3
LEARNING_RATE = 0.0001
BATCH_SIZE = 8
ENC_SEQ_LEN = 6  # 编码器序列长度
DEC_SQL_LEN = 4  # 解码器序列长度
DIM_VAL = 32  # value的维度
DIM_ATTN = 8  # attention的维度
N_HEADS = 4  # 多头注意力的头数
N_ENCODER_LAYERS = 2  # 编码器层数
N_DECODER_LAYERS = 2  # 解码器层数
LORA = True  # 是否使用LoRa

model = Transformer(dim_val=DIM_VAL,
                    dim_attn=DIM_ATTN,
                    input_size=SUBCARRIES,
                    dec_seq_len=DEC_SQL_LEN,
                    out_seq_len=LABELS_NUM,
                    n_decoder_layers=N_DECODER_LAYERS,
                    n_encoder_layers=N_ENCODER_LAYERS,
                    n_heads=N_HEADS)
model = model.float().to(device)

pth_path = os.path.join('./models/', sys.argv[1]) + '.pth'

def test_predict():
    '''
    使用传入的csi数据，批量处理，用于模型预测
    '''
    if MIX: data_path = './data/test'
    else: data_path = os.path.join('./data/test', sys.argv[1])

    if LORA:
        Build(model, lora_alpha=16)
        
        pth_path = os.path.join('./models/lora', sys.argv[1]) + '.pth'
    else:
        pth_path = os.path.join('./models/', sys.argv[1]) + '.pth'

    if (not torch.cuda.is_available()):
        model.load_state_dict(torch.load(pth_path, map_location='cpu'), strict=False)
    else:
        model.load_state_dict(torch.load(pth_path), strict=False)

    # model_state_dict = model.state_dict()
    # for name, param in model_state_dict.items():
    #     print(f"Layer: {name}, Shape: {param.shape}")

    
    csi_dataset = CSIDataset(data_path)
    val_loader = DataLoader(csi_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_size = len(csi_dataset)
    logging.info("Test size: {}".format(len(val_loader)))
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data, label in val_loader:
            # print(data)
            data = data.float().to(device)
            label = label.float().to(device)
            outputs = model(data)
            print(outputs.argmax(dim=1), label)
            total_accuracy += outputs.argmax(dim=1).eq(label).sum()

        print("accuracy: {}".format(total_accuracy/val_size))



def test_predict_file(csv_path):
    '''
    使用存放于csv文件里的csi数据，批量处理，用于模型训练
    '''

    if (not torch.cuda.is_available()):
        model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(pth_path))
    amplitude_data = process_single_csv_file(csv_path)

    amplitude_data = torch.tensor(amplitude_data).float().to(device)
    amplitude_data = amplitude_data.unsqueeze(0)

    output = model(amplitude_data)
    pred = F.log_softmax(output, dim=1).argmax(dim=1)
    # print(pred)
    if pred[0] == 0:
        res = 'empty'
    elif pred[0] == 1:
        res = 'fall'
    else:
        res = 'walk'
    print(res)
    return res

if __name__ == '__main__':
    test_predict()