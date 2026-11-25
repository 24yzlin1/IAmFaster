# IAmFaster

基于双路线手势识别的跑酷游戏。玩家面对摄像头做出握拳、张开手掌、左右移动手部等手势，控制 KayKit 3D 角色在无尽跑道上躲避障碍；同时附有一个独立的桌面工具 **Gesture Studio**，覆盖手势数据采集、模型训练与推理测试的完整闭环。

## 功能特性

- **跑酷游戏**：跳跃、下滑、变道等核心玩法，障碍、计分
- **双输入后端**：键盘与摄像头手势可随时切换，游戏逻辑零改动（MVP 架构 + `InputBackend` 协议）
- **双识别路线**：
  - **Mask CNN** —— YCrCb 肤色分割 + 128×128 mask 图 + 卷积网络
  - **Landmark MLP** —— MediaPipe 21 关键点归一化为 42 维特征 + 小型 MLP
  - 两条路线共用同一预测接口，加载哪个模型文件即走哪条路线（自动识别）
- **Gesture Studio**：数据采集（骨架点/手势框/分割预览，一帧双格式落盘）、模型训练（路线/epoch/学习率可调，JSONL 训练日志）、推理测试

## 技术栈

| 层次 | 技术 |
|---|---|
| 语言与工程 | Python 3.13+、uv |
| 游戏本体 | Pygame |
| 3D 角色 | KayKit 资源 + 自研蒙皮渲染器（骨骼动画 + 3D 着色器，moderngl） |
| 桌面工具 | PySide6 |
| 深度学习 | PyTorch |
| 计算机视觉 | MediaPipe Tasks（hand landmarker）、OpenCV（DirectShow） |

## 环境配置与部署

需要 Python 3.13+ 与摄像头。使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
uv sync
```

## 运行方法

```bash
# 启动游戏（默认键盘控制，配置文件中可切换手势模型路径）
uv run python main.py

# 启动 Gesture Studio（数据采集 / 模型训练 / 推理测试）
uv run python studio.py
```

### 操作说明

| 手势 | 动作 |
|---|---|
| 张开手掌（Palm） | 跳跃 |
| 握拳（Fist） | 下滑 |
| 手部左移 / 右移 | 变道 |

键盘模式：方向键对应相同动作。

### 训练自己的手势模型

1. 打开 Gesture Studio → **Data Collection**：选摄像头、选手势标签、录制（每帧同时保存裁剪图与关键点特征）
2. → **Training**：选特征路线（Mask CNN / Landmark MLP）、设置 epoch 与学习率、开始训练；训练日志与最优模型输出至 `snapshot/`
3. → **Inference**：加载模型验证识别效果
4. 在游戏配置中将模型路径指向训练产物（如 `gesture_model/gesture_landmark.pth`）即可用手势游玩

已有纯图片数据集可用训练面板的 *Convert dataset to landmarks...* 一键转换为关键点特征。

## 项目结构

```
├── main.py               # 游戏入口
├── studio.py             # Gesture Studio 入口
├── game/
│   ├── models/           # 玩法状态（MVP - Model）
│   ├── presenters/       # 流程协调（MVP - Presenter）
│   ├── views/            # 渲染与 UI（MVP - View）
│   ├── scenes/           # 场景切换
│   └── input_backends/   # 键盘 / 手势输入后端
├── utils/                # 手势识别与训练核心（检测、特征、训练器、预测器）
├── studio/               # Gesture Studio 面板与后台线程
├── dataset/              # 采集数据（jpg 裁剪图 + npy 关键点特征）
└── snapshot/             # 训练日志、checkpoint 与模型元数据
```

## License

本项目代码以 [MIT License](LICENSE) 发布。

第三方资源：角色模型来自 [KayKit](https://kaylousberg.itch.io/)（CC0）；MediaPipe、PyTorch、Pygame、PySide6、OpenCV 等依赖遵循各自的开源协议。
