# 项目文件结构

本文档记录了项目的实际文件结构和用途。

## 核心Python模块

| 文件 | 大小 | 用途 |
|------|------|------|
| `models.py` | 5.4 KB | 基础神经网络架构（LocalGNO, HR_TOA_BOA_Model） |
| `models_universal.py` | 9.4 KB | **推荐使用**：多分辨率训练实现和数据加载器 |
| `utils.py` | 7.1 KB | 工具函数（归一化、积分、插值等） |
| `data.py` | 6.0 KB | 数据加载和预处理（AtmosphericDataLoader） |
| `config.py` | 1.6 KB | 配置数据类（ModelConfig, TrainingConfig等） |

## 脚本

| 文件 | 大小 | 用途 |
|------|------|------|
| `run_finetune.py` | 8.0 KB | 自动化微调流程，支持迁移学习 |
| `quickstart.sh` | 1.7 KB | 快速启动脚本 |

## 测试

| 文件 | 大小 | 用途 |
|------|------|------|
| `test_utils.py` | 3.8 KB | 工具函数单元测试（pytest） |

## 文档

| 文件 | 大小 | 用途 |
|------|------|------|
| `README.md` | 4.8 KB | 项目主文档和使用指南 |
| `DATA.md` | 5.7 KB | 数据准备和格式说明 |
| `CORRECT_SOLUTION.md` | 7.8 KB | **重要**：可变分辨率问题的正确解决方案 |
| `技术解说文档.md` | 92 KB | 详细的技术文档和架构说明 |
| `FILE_STRUCTURE.md` | 本文件 | 项目文件结构说明 |

## 配置文件

| 文件 | 大小 | 用途 |
|------|------|------|
| `requirements.txt` | 315 B | Python依赖包列表 |
| `setup.py` | 2.0 KB | 包安装配置 |
| `.gitignore` | 600 B | Git忽略规则 |
| `LICENSE` | 1.1 KB | MIT许可证 |

## 预训练模型

| 目录/文件 | 大小 | 用途 |
|-----------|------|------|
| `pretrained_ckpt/` | - | 预训练模型检查点目录 |
| `pretrained_ckpt/base_model_10k.pt` | 1.9 MB | 在10k大气廓线上预训练的基础模型 |

## 总大小

- **代码**: ~42 KB（Python模块和脚本）
- **文档**: ~110 KB（Markdown文件）
- **配置**: ~4 KB（requirements, setup, gitignore）
- **预训练模型**: 1.9 MB
- **总计（不含.git）**: ~2.1 MB

## 使用指南

### 新用户
1. `README.md` - 从这里开始了解项目
2. `DATA.md` - 学习如何准备数据
3. `quickstart.sh` - 快速设置和测试
4. `run_finetune.py` - 主要使用的脚本

### 开发者
1. `models.py`, `models_universal.py` - 核心模型架构
2. `utils.py`, `data.py`, `config.py` - 支持模块
3. `test_utils.py` - 运行和扩展测试
4. `技术解说文档.md` - 深入理解架构细节

### 关于可变分辨率
- **必读**: `CORRECT_SOLUTION.md` - 解释了如何正确处理不同分辨率的数据
- **推荐使用**: `models_universal.py` 中的多分辨率训练方法

## 已删除的文件

以下文件已被删除，因为它们包含错误或过时的解决方案：
- `models_adaptive.py` - 自适应邻域方案（已证明不正确）
- `models_physical_distance.py` - 物理距离邻域方案（已证明不正确）
- `models_resolution_aware.py` - 分辨率嵌入方案（不推荐）
- `test_resolution_methods.py` - 测试错误方案的脚本
- `README_RESOLUTION.md` - 介绍错误方案的文档
- `RESOLUTION_GUIDE.py` - 过时的指南文件
- `FILE_MANIFEST.md` - 过时的文件清单

---

最后更新：2026-03-12
