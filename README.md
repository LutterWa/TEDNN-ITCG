# TEDNN-ITCG

迁移-集成 飞行时间约束制导律

collect_main.py: 自动采集预训练模型数据脚本（需将collect_data.py中TrEN标志置为False）
ppo_train.py: 强化学习训练与测试脚本，通过调节OMissile的继承对象切换source task与target task, super.__init__()函数中的输入变量k表示对基础气动系数的调节
collect_data: 采集数据使用的脚本与导弹模型
predictor_module: TEDNN核心代码
corrector_module: PPO核心代码
