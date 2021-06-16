"""train_criteo."""
import os
import sys
import argparse

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.common import set_seed

from src.deepfm import ModelBuilder, AUCMetric
from src.config import DataConfig, ModelConfig, TrainConfig
from src.dataset import create_dataset, DataType
from src.callback import EvalCallBack, LossCallBack

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--dataset_path', type=str, default="dataset/mindrecord", help='Dataset path')
parser.add_argument('--ckpt_path', type=str, default="checkpoint", help='Checkpoint path')
parser.add_argument('--eval_file_name', type=str, default="./auc.log",
                    help='Auc log file path. Default: "./auc.log"')
parser.add_argument('--loss_file_name', type=str, default="./loss.log",
                    help='Loss log file path. Default: "./loss.log"')
parser.add_argument('--do_eval', type=str, default='True',
                    help='Do evaluation or not, only support "True" or "False". Default: "True"')
parser.add_argument('--device_target', type=str, default="CPU", choices=("Ascend", "GPU", "CPU"),
                    help="device target, support Ascend, GPU and CPU.")
args_opt, _ = parser.parse_known_args()
rank_size = int(os.environ.get("RANK_SIZE", 1))

set_seed(1)

if __name__ == '__main__':
    data_config = DataConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

   
    # if args_opt.device_target == "Ascend":
    #     device_id = int(os.getenv('DEVICE_ID'))
    #     context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=device_id)
    # elif args_opt.device_target == "GPU":
    #     context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target=args_opt.device_target)
    # else:
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    rank_size = None
    rank_id = None

    ds_train = create_dataset(args_opt.dataset_path,
                              train_mode=True,
                              epochs=1,
                              batch_size=train_config.batch_size,
                              data_type=DataType(data_config.data_format),
                              rank_size=rank_size,
                              rank_id=rank_id)

    steps_size = ds_train.get_dataset_size()

    if model_config.convert_dtype:
        model_config.convert_dtype = args_opt.device_target != "CPU"
    model_builder = ModelBuilder(model_config, train_config)
    train_net, eval_net = model_builder.get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallBack(loss_file_path=args_opt.loss_file_name)
    callback_list = [time_callback, loss_callback]

    if train_config.save_checkpoint:
        if rank_size:
            train_config.ckpt_file_name_prefix = train_config.ckpt_file_name_prefix + str(get_rank())
            args_opt.ckpt_path = os.path.join(args_opt.ckpt_path, 'ckpt_' + str(get_rank()) + '/')
        if args_opt.device_target != "Ascend":
            config_ck = CheckpointConfig(save_checkpoint_steps=steps_size,
                                         keep_checkpoint_max=train_config.keep_checkpoint_max)
        else:
            config_ck = CheckpointConfig(save_checkpoint_steps=train_config.save_checkpoint_steps,
                                         keep_checkpoint_max=train_config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix=train_config.ckpt_file_name_prefix,
                                  directory=args_opt.ckpt_path,
                                  config=config_ck)
        callback_list.append(ckpt_cb)

    if args_opt.do_eval:
        ds_eval = create_dataset(args_opt.dataset_path, train_mode=False,
                                 epochs=1,
                                 batch_size=train_config.batch_size,
                                 data_type=DataType(data_config.data_format))
        eval_callback = EvalCallBack(model, ds_eval, auc_metric,
                                     eval_file_path=args_opt.eval_file_name)
        callback_list.append(eval_callback)
    model.train(train_config.train_epochs, ds_train, callbacks=callback_list, dataset_sink_mode=False)
    
