# Quantitative Research Protocol

本文件定义研究流程。目标不是找到历史收益最高的参数，而是建立一个可复现、可证伪、能拒绝弱模型的实验系统。任何研究结果都只能用于 Trading 212 Demo。

## 研究问题

主假设：横截面动量、风险状态和成交量/波动率特征的正则化集成，在计入换手成本后，能否比同期 SPY 获得更好的风险调整收益？

判定前固定以下规则：

- 外层验证：2023-01-01 至 2025-12-31，扩展窗口 walk-forward。
- 预测周期与再平衡周期：21 个交易日。
- 内层验证：252 个交易日。
- 训练/验证隔离：21 日 purge 加 21 日 embargo。
- 候选模型：Ridge、Elastic Net、Histogram Gradient Boosting、Extra Trees、Random Fourier Features + Ridge。
- 模型权重：仅依据当时内部验证的横截面 Spearman IC，正 IC 平方后归一化；没有正 IC 时等权。
- 交易成本：基础情景单边 10 bps，并压力测试 5、25、50 bps。
- 基准：SPY 和当前股票池等权组合。
- 多重试验：五个模型加三项策略设计选择，共计八次试验进入 Deflated Sharpe 诊断。

## 因果边界

在日期 `t` 形成预测时：

1. 特征只能使用 `t` 及以前的 OHLCV 和宏观代理数据。
2. 训练样本的 21 日未来标签必须已经成熟。
3. 内部验证窗口之前留出 purge 与 embargo 隔离带。
4. 当天收盘形成的目标仓位只影响下一交易日收益。
5. 参数和随机种子由 `research_config.json` 固定；外样本结果出来后不自动回调参数。

自动测试会通过篡改未来价格来验证过去特征完全不变，并检查隔离窗口、仓位上限、板块上限和统计结果的确定性。

## 系统结构

```text
research_config.json
        |
        v
data.py -> data manifest + SHA-256
        |
        v
features.py -> 64 causal asset/context/rank features
        |
        v
models.py -> nested validation + dynamic ensemble
        |
        v
portfolio.py -> Ledoit-Wolf covariance + constrained optimizer
        |
        v
statistics.py -> PSR/Deflated Sharpe + block bootstrap
        |
        v
runner.py -> report + daily audit trail + promotion decision
```

## Demo 影子晋级

研究候选必须同时满足：

- Deflated Sharpe probability ≥ 95%；
- 区块自助法中年化收益超过 SPY 的概率 ≥ 70%；
- 最大回撤 ≤ 30%；
- 单边 25 bps 成本下 Sharpe > 0.5。

全部通过也只允许进入 Demo 影子观察，不会立刻影响订单。随后还要通过独立的前向预测批次门槛。任何门槛失败时状态为 `RESEARCH_ONLY`。

## 运行与产物

```bash
python3 run_research.py --refresh-data
python3 -m unittest -v
```

主要产物位于 `outputs/research_project/`：

- `latest_report.md`：人类可读报告；
- `latest_report.json`：完整指标、模型诊断和每次再平衡决策；
- `daily_returns.csv`：每日毛收益、净收益、基准、换手和权益曲线；
- `data_manifest.json`：数据范围、可用标的和内容哈希；
- `resolved_config.json`：实际运行配置；
- `candidate_decision.json`：机器可读晋级判定；
- `experiment_registry.jsonl`：不可覆盖的实验登记记录。

## 已知局限

- 50 只股票是当前候选池，存在存活者偏差与事后选择偏差。
- Yahoo 日线数据适合研究，但不等同于 Trading 212 的实际成交、点差和滑点。
- 固定 bps 成本无法完整模拟容量、冲击成本和极端行情流动性。
- 三年外样本仍然偏短，尤其不足以覆盖多种长期市场周期。
- 日线候选与分钟级 Demo 策略不是同一个数据生成过程，不能直接移植回测收益。
- 统计概率依赖模型假设，不能解释为未来赚钱概率。

## 方法依据

- Gu, Kelly, Xiu, “Empirical Asset Pricing via Machine Learning”: https://www.nber.org/papers/w25398
- Harvey, Liu, Zhu, “... and the Cross-Section of Expected Returns”: https://www.nber.org/papers/w20592
- Frazzini, Israel, Moskowitz, “Trading Costs of Asset Pricing Anomalies”: https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini%2C%20Israel%20and%20Moskowitz.pdf
- Bailey et al., “The Probability of Backtest Overfitting”: https://doi.org/10.21314/JCF.2016.322
