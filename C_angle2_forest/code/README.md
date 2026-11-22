# Source code — Angle 2 / Random Forest

这一目录是 Angle 2 / Random Forest 的公开源码：数据读入、调参与评估、分组置换重要性、图表。

不在这里的：幻灯片生成与摘要排版工具；它们只排版已算好的数字，不产生分析结果。

## 文件职责

| 文件 | 作用 | 产出 |
|---|---|---|
| `common.py` | 唯一的数据入口与公共工具：调用克隆进来的 `shared_preprocessing`，pooled OOF 指标、分组置换、RF 工厂、路径常量 | — |
| `shared_preprocessing.py` | **自 A_data 交付包克隆**（逐字节一致）。数据读入、固定划分校验、预处理管线、编码列到解释组的映射 | — |
| `build_dataset.py` | **自 A_data 交付包克隆**（逐字节一致）。被上一模块引用其常量与校验函数 | — |
| `tune_rf.py` | 12 配置 × 10 折网格，按 **pooled OOF RMSE** 选参，附留一折选参稳定性审计 | `outputs/rf_tuning_main.csv`、`rf_tuning_folds_main.csv`、`oof_all_configs_main.csv`、`rf_selection_main.json`、`unknown_by_fold_main.csv` |
| `evaluate_rf.py` | 用冻结配置逐折重拟合：OOF 预测、6 组原始变量置换（30 重复）、MDI 汇总、两种尺度排名对照 | `outputs/predictions.csv`、`metrics.csv`、`fold_metrics.csv`、`rf_permutation_*`、`rf_impurity_*`、`rf_importance_*` |
| `evaluate_conditional.py` | 条件置换（只在同 `brand_model` 内重排 age / mileage / capacity），量化"超出车型的信息" | `outputs/rf_permutation_conditional_*.csv` |
| `selection_check.py` | 留一折重新选参，量化单层十折的选择乐观量 | `outputs/selection_stability.csv`、`.json` |
| `finalize_rf.py` | 全开发集重拟合冻结配置，**测试集只评估一次**；需要 `evaluate_rf.py` 先写出 `outputs/metrics.csv`；写出完整溯源 | `outputs/model_config.json`、`test_metrics.csv`、`test_predictions.csv`、`final_dev_metrics.csv` |
| `figures.py` | 6 张图（调参热图、组置换、两尺度对照、折间离散、raw vs conditional、配对 ΔRMSE） | `figures/fig01…fig06.png` |
| `merge_angle2.py` | 把 B_angle2_linear 的线性模型预测与森林结果合并：先校验 `row_id`／折／`y_true` 三方一致，再算四模型 pooled 指标、配对 bootstrap 区间、Lasso 选择按解释组汇总 | `outputs/angle2_models.csv`、`angle2_lasso_groups.csv` |

## 来源与克隆

`shared_preprocessing.py` 与 `build_dataset.py` 不是本组的独立工作，而是 A_data 交付包 `member-a-v1` 的原件。为保证附录能独立运行、也避免 A 日后修改导致读数变化，按**字节完全一致**的方式复制进来：

| 克隆文件 | 源路径 | SHA-256 |
|---|---|---|
| `shared_preprocessing.py` | `A_data/code/shared_preprocessing.py` | `42d3c955c42112b9a50bed23847b8ef5aedf0a421ef5e4cc356b4c0979f2d40d` |
| `build_dataset.py` | `A_data/code/build_dataset.py` | `3761169be788936aa8d61e93d8484d21329a83062f859d5146b234f1c886efbd` |

B_angle2_linear 只读引用：`merge_angle2.py` 从 `B_angle2_linear/results/`（`predictions.csv`、`fold_metrics.csv`、`lasso_selected_features.csv`）读取，不修改对方文件；缺这些文件时脚本会停下并说明，不会静默跳过。

数据本身不复制：代码默认从 `data/` 读取（`cleaned.csv` 与 `split_manifest.csv`，哈希写在 `outputs/model_config.json` 的 `data.sha256`）。若脱离本仓库结构单独运行，设置环境变量即可：

```bash
export MEMBER_A_DATA="/path/to/data"      # 含 cleaned.csv 与 split_manifest.csv 的目录
python -B tune_rf.py
```

`eda.py` 未被引用，故未克隆。

## 运行顺序

```bash
cd C_angle2_forest/code
python -B tune_rf.py          # 约 10 分钟（12 配置 × 10 折 × 500 棵树）
python -B evaluate_rf.py      # 约 4 分钟（含 1800 次置换预测）
python -B evaluate_conditional.py   # 约 3 分钟
python -B finalize_rf.py      # 秒级；会读测试集，务必在前三步之后
python -B selection_check.py  # 秒级
python -B merge_angle2.py     # 秒级；需要 B_angle2_linear 的 predictions.csv
python -B figures.py          # 秒级
```

结果摘要工具不包含在本公开仓；不要按不存在的路径执行。

`tune_rf.py` 支持 `--limit-folds` / `--limit-configs` / `--tag`，用于小规模自测（自测产物用不同 tag，避免覆盖正式结果）。

## 输出文件（14 个，按用途分三类）

| 文件 | 内容 | 谁读 |
|---|---|---|
| `predictions.csv` | 开发集 OOF 预测（row_id, cv_fold, y_true, y_pred_price） | 配对比较、误差集中度 |
| `test_predictions.csv` | 测试集逐行预测 | 测试评估、A 做散点图 |
| `metrics.csv` | 一个模型的所有指标：pooled、平均折、样本内、测试（`evaluation_split` + `aggregation` 区分） | 幻灯片、摘要、A |
| `fold_metrics.csv` | 十折逐折指标 | 折间波动、B 的合并表 |
| `model_config.json` | 数据哈希、预处理、超参、搜索空间、选参准则、版本 | 复现、E 复用 |
| `rf_tuning_main.csv` | 12 个配置的完整网格（pooled + 折均值/SD） | 选参、热图 |
| `rf_selection_main.json` | 冻结配置 + 留一折选参审计 + 每折实际取用列数 | 选参说明 |
| `oof_all_configs_main.csv` | 12 个配置的逐行 OOF 预测（1.1 MB） | `selection_check.py` 复算稳定性 |
| `selection_stability.json` | 留一折选参的逐折结果与结论 | 方法限制 |
| `unknown_by_fold_main.csv` | 每折验证集中未见过类别的行数 | 限制说明 |
| `importance.csv` | **长表**：group × measure（permutation_raw / permutation_conditional / impurity_mdi_share）× 值、折间 SD、重复内 SD | 幻灯片、摘要 |
| `importance_by_fold.csv` | 同一长表的逐折明细 | 复核 SD 的来源 |
| `angle2_models.csv` | 四模型指标 + 相对森林的配对差与 95% 区间 + 折间均值/SD | 幻灯片页 1、摘要 |
| `angle2_lasso_groups.csv` | Lasso 选中的 60 列按解释组汇总 | 幻灯片页 2、摘要 |

早先一版把重要性拆成 6 个文件（raw／MDI／conditional × 汇总／逐折）、指标拆成 3 个（开发／样本内／测试），又把 Lasso 分组对照拆成 5 个；这些都是同一批数字的不同切法，现已并成长表或并入上面 14 个文件。

## 依赖与环境

- Python 3.12.8；`scikit-learn 1.9.1`、`numpy 2.5.4`、`pandas 3.1.0`、`scipy 1.16.3`、`matplotlib 3.10.9`（完整记录见 `outputs/model_config.json` 的 `software` 字段）。
- 除 numpy / pandas / sklearn / matplotlib 外只依赖标准库；A 的两个模块已克隆在本目录，**可直接运行**，不需要再取得 A 的代码。
- `n_jobs` 固定为 1：沙箱禁止命名管道，joblib 并行会报 `PermissionError [WinError 5]`。单线程下 500 棵树在 4,310 × 203 上约 4–6 秒，整轮网格 10 分钟可接受。

## 代码语言约定

全部注释、docstring、变量名与输出文件名使用英文。

## 口径（复现时必须一致，否则数字不可比）

1. 只用 `load_development` 的 4,789 条与固定十折；不重划分、不重抽种子（42）。
2. 选参与汇报都用 **pooled OOF**：十折验证预测拼起来算一次 RMSE，**不是**平均折 RMSE。
3. 选定配置的 OOF 标 *selected on CV*，不宣称无偏。`selection_check.py` 只审计留一折选参的稳定性，不单独计算一个 optimism 数值；不要把 +0.87% 说成该脚本的正式输出。
4. 预处理在每个训练折内拟合；RF 不缩放。
5. 测试集在配置冻结后只评估一次。
6. 置换针对**原始列**行置换后重新 `transform`，不直接打乱 one-hot 列。

## 这份源码支撑的结论

- 冻结配置 `max_features=0.3`、`min_samples_leaf=1`、`n_estimators=500`：开发集 pooled OOF RMSE 2,539,108 BDT（R² 0.7512）。
- 与 B 的线性模型在同一批 4,789 行上比：森林 2.54M，OLS 2.8882M、Ridge 2.9762M、Lasso 2.9332M，即比最好的线性模型低 12.1%；配对 bootstrap 差 0.35M、95% 区间 [0.00, 0.76]M。
- 原始置换里 capacity 最强（1.55M BDT），但同车型内置换后只剩 9%；age 保留 43%、mileage 25%。
- Lasso 保留的 60 个编码列中 47 个是车辆身份、占系数绝对值 95.4%：两种方法都指向"车型标签"，但在 age／mileage 上分歧。
- MDI 与置换的组排名不一致（Spearman 0.77）：vehicle_identity 在 MDI 第一、置换第三。
