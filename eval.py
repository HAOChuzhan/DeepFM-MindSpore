"""eval_criteo."""
import os
import sys
import time
import argparse

from mindspore import context
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.deepfm import ModelBuilder, AUCMetric
from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import create_dataset, DataType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser(description='CTR Prediction')
# 需要修改最新模型的checkpoint位置
parser.add_argument('--checkpoint_path', type=str, default="checkpoint/deepfm_5-9_49.ckpt", help='Checkpoint file path')
parser.add_argument('--dataset_path', type=str, default="dataset/mindrecord", help='Dataset path')
parser.add_argument('--device_target', type=str, default="CPU", choices=("Ascend", "GPU", "CPU"),
                    help="device target, support Ascend, GPU and CPU.")
args_opt, _ = parser.parse_known_args()
device_id = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=device_id)


def add_write(file_path, print_str):
    with open(file_path, 'a+', encoding='utf-8') as file_out:
        file_out.write(print_str + '\n')


if __name__ == '__main__':
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    ds_eval = create_dataset(args_opt.dataset_path, train_mode=False,
                             epochs=1, batch_size=train_config.batch_size,
                             data_type=DataType(data_config.data_format))
    if model_config.convert_dtype: # 是否进行平台转换
        model_config.convert_dtype = args_opt.device_target != "CPU"
    model_builder = ModelBuilder(model_config, train_config)
    train_net, eval_net = model_builder.get_train_eval_net()
    train_net.set_train()
    eval_net.set_train(False)
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})
    param_dict = load_checkpoint(args_opt.checkpoint_path)
    load_param_into_net(eval_net, param_dict)

    start = time.time()
    res = model.eval(ds_eval)
    eval_time = time.time() - start
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    out_str = f'{time_str} AUC: {list(res.values())[0]}, eval time: {eval_time}s.'
    print(out_str)
    add_write('./auc.log', str(out_str))
