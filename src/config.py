# Copyright 2020 Huawei Technologies Co., Ltd
"""
network config setting, will be used in train.py and eval.py
"""

class DataConfig:
    """data config"""
    data_vocab_size = 184965
    train_num_of_parts = 21
    test_num_of_parts = 3
    batch_size = 16000
    data_field_size = 39
    data_format = 1

class ModelConfig:
    """model config"""
    batch_size = DataConfig.batch_size
    data_field_size = DataConfig.data_field_size
    data_vocab_size = DataConfig.data_vocab_size
    data_emb_dim = 80 # 80
    deep_layer_args = [[1024, 512, 256, 128], "relu"]
    init_args = [-0.01, 0.01]
    weight_bias_init = ['normal', 'normal']
    keep_prob = 0.9
    convert_dtype = True

class TrainConfig:
    """train config"""
    batch_size = DataConfig.batch_size
    l2_coef = 8e-5
    learning_rate = 5e-4 # 5e-4
    epsilon = 5e-8
    loss_scale = 1024.0

    train_epochs = 20

    save_checkpoint = True
    ckpt_file_name_prefix = "deepfm"
    save_checkpoint_steps = 1
    keep_checkpoint_max = 50

    eval_callback = True
    loss_callback = True
