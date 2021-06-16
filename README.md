# MindSpore
这里是对于MinsSpore框架中部分模型的复现，参考其中model_zoo相关模型。
#### deepFM模型
- 数据预处理 输入：	python src/preprocess_data.py 
- 模型训练 输入：    	python train.py
- 模型评估 输入：    	python eval.py                                # 需要修改对应--checkpoint_path中的最新ckpt文件位置
