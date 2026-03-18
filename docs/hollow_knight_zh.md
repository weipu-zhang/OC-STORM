# 空洞骑士配置说明

> English version: [hollow_knight.md](hollow_knight.md)

> **游戏版本提示：** 请使用空洞骑士版本 **1.5.78.11833**。当前最新版本缺少 mod API 支持以及与大多数 mod 的适配。如果 mod 生态大规模向新版本迁移，我们也会同步跟进；否则将维持现状。

## 硬件要求

GPU 性能最好不低于 3090，不然可能难以达到 9 FPS 的策略执行频率（也可以调低帧率尝试）。

## 总体说明

空洞骑士的训练使用 `ray` 在游戏节点和训练节点之间通信。游戏运行在一个节点上，模型训练与推理运行在另一个节点上。论文中的大多数实验均在单台 Windows 机器上完成：Hyper-V 虚拟机运行游戏，WSL 运行 PyTorch 训练与推理，`ray` 在两者之间传递截图观测和动作。

这种分布式架构看起来很麻烦但是有以下几个比较重要的优势：
1. PyTorch 在 WSL 里训练比在 PowerShell 更快。
2. Windows 里面向游戏窗口发送键盘输入是要求该窗口保持在最前台的。若不使用虚拟机或独立的游戏机器，训练过程会占用桌面焦点，就没法干别的事情或者看训练曲线了。
3. 分布式有助于进一步scale up：比如 `train_async.py` 可以接入多卡性能更强的非 Windows 节点来加速训练。

我们写了一个 mod 来提取击中敌人和受伤的信号来计算奖励。关于 mod 的安装，首先从 https://github.com/hk-modding/api/releases 安装 mod API（建议B站搜索相关的视频或图文教程），然后吧 `game_mod/` 里面的  `HKRLEnv` 安装到对应路径里。

## 环境配置

游戏环境和训练环境必须使用相同版本的 `python` 和 `ray`。

1. 在游戏节点（如虚拟机）上执行 `pip install -r requirements-game.txt` 安装游戏端依赖，主要是截屏和键盘模拟相关的包。
2. 在训练节点上执行 `./scripts/start_ray.sh` 启动 ray。
3. 在游戏节点上执行 `./scripts/win_start_ray.ps1`，并根据实际情况修改其中的 conda 环境名和 IP 地址。
4. 可选：执行 `ray status` 确认连接状态。
5. 执行 `./scripts/train.sh` 启动训练。`train.py` 和 `train_async.py` 均支持 Hollow Knight。`train.py` 主要面向单卡场景：在打 boss 期间不进行训练以提高推理频率，一局结束后再执行策略更新；`train_async.py` 需要两块 GPU，一块持续进行策略更新，另一块负责推理。
