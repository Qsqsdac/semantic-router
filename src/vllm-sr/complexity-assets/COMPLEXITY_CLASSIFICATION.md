# 数据集生成脚本 - 复杂度分类改进

## 更新说明

脚本已从**纯长度指标**改为**内容特征+长度的集成指标**来判断请求复杂度。

### 新的复杂度计算方法

#### 1. 内容特征评分 (`content_based_complexity`)

基于以下四个维度计算 0-1 的分数：

- **推理步骤数** (权重 30%)：检测关键词 first, second, step, finally 等
- **逻辑连接词密度** (权重 20%)：检测 since, thus, if, however, therefore 等
- **结构化输出** (权重 25%)：检测编号列表、符号列表、LaTeX 公式
- **代码块** (权重 25%)：检测代码块标记 (```...```)

#### 2. 集成分数计算 (`classify_complexity`)

```
最终得分 = 0.7 × 内容特征分数 + 0.3 × 长度分数
长度分数 = min(单词数 / 300, 1.0)
```

#### 3. 三级分类

| 分类 | 条件 | 默认参数 |
|------|------|--------|
| easy | 得分 < 0.33 | `--easy-threshold 0.33` |
| medium | 0.33 ≤ 得分 < 0.66 | 默认 |
| hard | 得分 ≥ 0.66 | `--hard-threshold 0.66` |

### 使用示例

#### 默认参数运行
```bash
python make_dataset.py \
  --dataset lmsys/chatbot_arena_conversations \
  --sample-size 10000
```

#### 使用更严格的阈值（如用户建议的 0.1 - 0.9）
```bash
python make_dataset.py \
  --dataset lmsys/chatbot_arena_conversations \
  --sample-size 10000 \
  --easy-threshold 0.1 \
  --hard-threshold 0.9
```

这样只有明确的简单/复杂请求会被分类，中间的模棱两可的请求会被 easy 类过滤掉。

#### 自定义其他参数
```bash
python make_dataset.py \
  --dataset lmsys/chatbot_arena_conversations \
  --sample-size 10000 \
  --train-ratio 0.8 \
  --easy-threshold 0.2 \
  --hard-threshold 0.7 \
  --output-dir ./my_dataset
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--easy-threshold` | float | 0.33 | easy 分类的上限阈值 |
| `--hard-threshold` | float | 0.66 | hard 分类的下限阈值 |
| `--sample-size` | int | 10000 | 生成的总样本数 |
| `--train-ratio` | float | 0.9 | 训练集比例 |
| `--seed` | int | 42 | 随机种子 |
| `--output-dir` | path | `.build/complexity` | 输出目录 |

### 测试脚本

运行测试以查看不同类型文本的分类结果：

```bash
# 快速测试
python test_complexity.py

# 详细调试
python debug_test.py
```

### 实现细节

- 所有关键词检测都是**英文**（数据集是全英文）
- 内容特征权重总和为 1.0，避免重复计算
- 长度在最终集成分数中权重为 30%，确保长短结合
- 分数计算避免上溢（使用 `min(score, 1.0)`）
