# Trading 212 Demo 模拟交易终端

这是一个只连接 Trading 212 **Paper Trading (Demo)** 的本地终端。代码中的服务器地址固定为：

`https://demo.trading212.com/api/v0`

它不会连接真实账户，也不需要 OpenAI API。

## 1. 创建 Demo API 密钥

1. 在 Trading 212 切换到 Demo/Practice 模式。
2. 打开 `Settings -> API (Beta)`。
3. 生成一组 Demo 专用的 API Key 和 API Secret。
4. 至少授予账户、持仓和订单所需权限。建议限制为你自己的可信 IP。

Demo 与真实账户的密钥可能不同。不要把 Key、Secret、账户密码或 2FA 发到聊天中。

## 2. 本地配置

```bash
cd /Users/yq225/Documents/Codex/2026-07-20/trading212
python3 -m pip install -r requirements.txt
cp .env.example .env
```

用本地文本编辑器打开 `.env`，填入：

```dotenv
T212_API_KEY=你的DemoKey
T212_API_SECRET=你的DemoSecret
```

`.env` 已被 `.gitignore` 排除。

## 3. 验证连接

```bash
python3 t212_demo.py check
```

## 4. 搜索股票的 Trading 212 ticker

Trading 212 下单使用自己的 ticker，例如 `AAPL_US_EQ`，不是简单的 `AAPL`：

```bash
python3 t212_demo.py search Apple
python3 t212_demo.py search AAPL
```

标的列表由 Trading 212 每 10 分钟刷新一次，本程序也缓存 10 分钟。强制刷新可加 `--refresh`。

## 5. 实时监控模拟持仓

```bash
python3 t212_demo.py positions --watch
```

持仓接口官方限制为每秒 1 次，因此程序的最短刷新间隔是 1.1 秒。按 `Ctrl+C` 停止。

## 6. 模拟下单

不加 `--confirm-demo` 时只预览，不发送订单：

```bash
python3 t212_demo.py market buy AAPL_US_EQ 0.1
```

确认后发送到 Demo：

```bash
python3 t212_demo.py market buy AAPL_US_EQ 0.1 --confirm-demo
python3 t212_demo.py market sell AAPL_US_EQ 0.1 --confirm-demo
python3 t212_demo.py limit buy AAPL_US_EQ 0.1 180 --confirm-demo
```

查看和撤销待成交模拟订单：

```bash
python3 t212_demo.py orders
python3 t212_demo.py cancel 订单ID --confirm-demo
```

## 重要限制

- Trading 212 Public API 目前为 Beta，只支持 Invest 和 Stocks ISA 类型。
- API 只能按股票数量下单，不支持按金额下单。
- 市价单接口不是幂等的；重复发送可能产生重复模拟订单。因此程序不会自动重试下单。
- Public API 没有为所有未持仓股票提供实时行情流。`positions --watch` 显示的是持仓接口返回的当前价；自动策略若要在买入前持续计算信号，需要另接合规的实时行情源。
- 该项目仅用于模拟交易和技术测试，不代表投资建议。

## 7. 持续自动模拟交易

`auto_trader.py` 使用 `rational_momentum_ml_v3` 多因子长仓模型：

- 每 60 秒采样，分别计算约 15 分钟、1 小时和 4 小时的对数收益。
- 用近期实现波动率标准化三个周期的动量，再加入上涨样本占比，形成可解释的综合评分。
- 对所有主策略股票做横截面排名，只持有分数超过门槛的前 6 只。
- 通过反波动率配置权重；单股最多 30%，单板块最多 40%，保留至少 5% 现金。
- 用候选股票中位数趋势判断 `RISK_ON / NEUTRAL / RISK_OFF`，相应把目标总仓位调整为 95% / 65% / 25%。
- 组合回撤达到 8% 时把目标总仓位压到 40% 以下，达到 12% 时压到 15% 以下。
- 使用 8% 硬止损和 5%–15% 波动率自适应移动止损，不固定止盈，让强趋势继续运行。
- 最多每 15 分钟再平衡一次；偏差小于资产的 1.5% 或 £20 时不交易，单笔最多约 £300。
- 不设置每日订单数量上限，但通过再平衡间隔、冷却时间和偏差门槛控制无效换手。
- 只在常规交易时段提交订单，不使用盘前盘后交易。

启动真实 Demo 订单执行：

```bash
python3 auto_trader.py run --execute-demo
```

查看成果或停止：

```bash
python3 auto_trader.py status
python3 auto_trader.py stop
```

所有快照、信号和模拟订单记录保存在 `outputs/auto_trader/`。参数可在 `strategy.json` 中调整。

当前主策略股票记录在 `universe.json`。模型启动后需要约 4 小时形成完整的实时采样窗口；预热期间不会根据不完整信号追涨，只会处理硬止损和超过 30% 的集中仓位。策略升级时会清空旧模型的价格样本，避免把不同采样频率的数据混在一起。

市场休市时程序每 60 秒检查一次是否重新开盘，不提交订单。网络或 API 失败时使用最长 5 分钟的指数退避，并在恢复后继续运行，避免断网期间高频空转。

### 自动市场侦察

`universe.json` 维护 50 只股票、11 个板块的候选池：8 只原有股票继续运行主策略，另外 42 只作为侦察标的。开盘期间最多每 15 分钟新增一个约 £1.35–£2.50 的 Demo 探针仓，并始终保留策略要求的现金缓冲。候选股至少积累 241 个一分钟样本后，使用与主策略相同的波动率标准化动量评分；评分达到 0.75 且长周期收益为正的最强股票才会晋升。

## 8. 研究回测

`research_backtest.py` 使用 2020–2022 年作为模型选择区间，从预先声明的三个参数组中选择训练期 Sharpe 最高者，然后只在 2023–2025 年做样本外验证。回测使用月度再平衡、0.1% 单边换手成本和日线版同类信号：

```bash
python3 research_backtest.py --refresh
```

结果写入 `outputs/research_backtest.json`。当前平衡参数在 2023–2025 样本外区间年化约 11.3%、最大回撤约 18.2%，但同期 SPY 年化约 23.2%。这说明风险控制有效但没有证明策略能战胜被动指数；结果还存在存活者偏差、当前股票池选择偏差以及日线模型与实时模型频率不同等限制。

## 9. 实验机器学习模型

`experimental_model.py` 集成两个结构不同的模型：

- 标准化后的 Ridge 回归，用强正则化捕捉较稳定的线性关系。
- 深度限制为 3 的直方图梯度提升树，用于捕捉动量与波动率之间的非线性交互。

模型的 12 个因果特征只使用预测时点之前的价格，包括 5/15/60/120 周期收益、15/60 周期波动率、上涨比例、60 周期回撤、价格与均值距离以及短长动量交互。训练标签延迟 15 个样本，未成熟的未来收益绝不会进入训练集。

实时模型默认处于 `WARMUP`，训练样本足够后进入 `SHADOW`，只记录预测而不影响订单。最近至少 20 个独立预测批次、40 个入选结果的方向命中率达到 53%，并且扣除 0.1% 假设往返成本后的平均收益为正，才进入 `APPROVED`。获批后它最多影响综合评分的 20%；原有仓位、板块、回撤和止损限制仍然有效。指标恶化时会自动退回影子模式。

样本外日线等价验证：

```bash
python3 experimental_backtest.py
```

2023–2025 扩展窗口验证会在每次月度再平衡时只使用当时已经成熟的训练标签。当前结果为年化约 36.3%、Sharpe 约 1.99、最大回撤约 22.2%；同期 SPY 年化约 23.2%、Sharpe 约 1.44。该结果比基础模型强，但仍包含当前股票池的存活者和选择偏差，且日线等价验证不能保证分钟实时模型获得同样表现，因此实时前向准入门槛不可跳过。

## 测试

```bash
python3 -m unittest -v
```
