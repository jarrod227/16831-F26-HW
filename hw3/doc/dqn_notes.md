# HW3 学习笔记：DQN / DDQN

## 1. 从 Q 表到 Q 网络

表格 Q-learning 的更新规则：

```
Q[s][a] ← Q[s][a] + α · ( r + γ · max_a' Q[s'][a'] − Q[s][a] )
```

LunarLander 的状态是 9 维连续向量，表格存不下，于是用神经网络 `Q(s; φ)` 代替这张表：
输入 s，一次输出所有动作的 Q 值（LunarLander 有 4 个动作，输出就是 4 个数）。

"改一格"变成"把网络输出往目标值拉一步"，本质是一个回归问题：

```
loss = Huber( Q(s; φ)[a] ,  y )          y = r + γ · max_a' Q(s'; ·)[a']
                 ↑ 要学的              ↑ 学习的目标（TD target）
```

对应代码（`critics/dqn_critic.py`）：

```python
qa_t_values = self.q_net(ob_no)                                   # 查表：s 下 4 个动作的 Q 值
q_t_values  = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1))   # 取实际执行的那个动作 a 的值
loss = self.loss(q_t_values, target)
```

## 2. 问题的根：目标里也有 Q

Q 在拿自己当老师教自己。表格版没事，因为改 `Q[s][a]` 这一格，`Q[s'][·]` 纹丝不动。
换成神经网络后，改一次参数**所有**状态的输出都动了：你刚往目标走一步，目标自己也挪了。
像射箭时靶子跟着手一起晃，永远射不中。

## 3. 解法一：Target network —— 把老师冻住

复印一份老师，钉在墙上：

| | 本质 | 什么时候变 | 干什么用 |
|---|---|---|---|
| **online** `q_net`（φ） | 正在学的学生 | 每一步 `optimizer.step()` | 选动作、算 loss 左边 `Q(s,a)` |
| **target** `q_net_target`（φ⁻） | online 的定期快照 | 每 `target_update_freq` 步被整个覆盖 | 只用来算 loss 右边的目标值 |

```python
# critics/dqn_critic.py
self.q_net        = network_initializer(self.ob_dim, self.ac_dim)   # 学生
self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)   # 墙上的复印件

def update_target_network(self):          # 复印：target ← online
    for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
        target_param.data.copy_(param.data)
```

```python
# agents/dqn_agent.py —— 每 N 次参数更新换一次老师，DQN 和 DDQN 都一样
if self.num_param_updates % self.target_update_freq == 0:
    self.critic.update_target_network()
```

**要点：target 网络不是"学得慢"，它根本不学，只是定期被覆盖。**
两者是同一个网络的新版本和旧版本，不是两个不同的东西。

时间线示例（N = 1000）：第 1000 → 1999 步，学生从 φ_1000 学到 φ_1999，老师全程 φ⁻ = φ_1000；
第 2000 步复印一次，老师变成 φ_2000。

## 4. 解法二：Double-Q —— 把"挑"和"评"拆开

### 问题：max 会偏高

网络输出是带噪声的近似。对 4 个带噪声的数取 `max`，结果**系统性偏大**：
噪声为正的那个动作更容易被选中，选中它的同时又用它那个偏大的值。
例：4 个动作真值都是 0，估计是 `[+0.3, −0.2, +0.1, −0.4]`，`max` 取到 0.3，但真值是 0。
偏差沿 Bellman 方程一步步往前传、累积。这就是作业 PDF 里说的"没有 double-Q 时 return 到 125 后经常往下掉"的原因之一。

### 修法

目标值 `y = r + γ · Q(s', a')` 要回答两个问题：(1) 在 s' 该选哪个动作？**挑**；(2) 这个动作值多少？**评**。

| | 挑 a' 用的参数 | 评分用的参数 |
|---|---|---|
| DQN | 老师 φ⁻ | 老师 φ⁻ |
| DDQN | **学生 φ_k**（此刻的） | 老师 φ⁻ |

```
DQN : y = r + γ · max_a Q(s', a ; φ⁻)

DDQN: a* = argmax_a Q(s', a ; φ_k)        ← 学生挑
      y  = r + γ · Q(s', a* ; φ⁻)         ← 老师给这个动作打分
```

一个人既挑又评，会偏向挑"自己恰好高估了的那个"。换一个人来挑，两个网络的误差不同步，这种便宜就占不到了。

对应代码（`critics/dqn_critic.py`）：

```python
qa_tp1_values = self.q_net_target(next_ob_no)        # 老师：s' 下 4 个动作的 Q 值，如 [1.2, 0.8, 1.5, 0.3]

if self.double_q:
    qa_tp1_online = self.q_net(next_ob_no)           # 学生：如 [1.4, 0.7, 1.1, 0.5]
    ac_tp1 = qa_tp1_online.argmax(dim=1)             # 学生说最好的是第 0 个
    q_tp1 = torch.gather(qa_tp1_values, 1,
                         ac_tp1.unsqueeze(1)).squeeze(1)   # 去老师那行取第 0 个 → 1.2
else:
    q_tp1, _ = qa_tp1_values.max(dim=1)              # 老师自己取 max → 1.5

target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
```

注意例子里 DQN 得 1.5，DDQN 得 1.2：DDQN 取的是"学生认为最好的动作"在老师那里的值，往往不是老师那行的最大值，整体比 DQN 偏小一点，正好抵消 `max` 的偏高。

### "老师挑"和"直接 max"是一回事

`max_a Q(s', a; φ⁻)` 其实隐含了两步：

```
max_a Q(s', a; φ⁻)  =  Q(s', a*; φ⁻)    其中 a* = argmax_a Q(s', a; φ⁻)
                                                   ↑ 挑：找最大值的下标
                              ↑ 评：取那个下标对应的值
```

挑和评用同一个网络时，两步合起来就是一个 `max`，DQN 一行搞定。
DDQN 挑用 φ_k、评用 φ⁻，不是同一个网络，合不成一个 `max`，所以必须拆开写成 `argmax` + `gather`。

**DQN 是 `老师.max()`；DDQN 是 `老师[学生.argmax()]`。**

### 用"两张表"过一遍 DDQN

对同一个 s'，查两张表：

```
学生表 Q(s', ·; φ_k)  = [1.4, 0.7, 1.1, 0.5]   → argmax → a* = 0        （挑）
老师表 Q(s', ·; φ⁻)   = [1.2, 0.8, 1.5, 0.3]   → 取 a*=0 那格 → 1.2     （评）

y = r + γ · 1.2
```

DQN 只查老师一张表，直接取最大的 1.5。

两个细节：
1. 两张表是**同一张表的新旧版本**，不是两个独立训练的网络。老师每 `target_update_freq` 步被学生覆盖一次。
2. 学生表**只用来挑下标，它的数值不进目标值**。`y` 里的 Q 永远来自老师，DQN 和 DDQN 一致，
   所以 DDQN 没有破坏 target network 带来的"目标稳定"。

### 常见误解（我自己踩过的）

- ❌ "DDQN 就是更新慢一点" —— 更新频率、网络结构、学习率全一样。
- ❌ "DDQN 每隔几步更新老师，DQN 不更新" —— 两者都每 `target_update_freq` 步更新老师，`--double_q` 不碰这件事。
- ❌ "DDQN 的老师用上一步的 φ_k" —— 老师永远是 φ⁻（N 步前的拷贝）；DDQN 只是**挑动作**这一步改问学生 φ_k。

### 一句话

**target network 解决"目标在动"，double-Q 解决"目标偏高"。**
DDQN 和老师更新频率无关，只是把"挑动作"从老师手里换给了学生。

## 5. 其余几行

- `(1 - terminal_n)`：终止状态没有下一步，把 `γ·Q(s',·)` 截成 0，y = r。
- `target.detach()`：目标值不参与反向传播，梯度只流进 `q_net`。
- ε-greedy（`agents/dqn_agent.py`）：以 ε 概率随机探索，ε 按 `exploration_schedule` 随时间衰减；
  `t < learning_starts` 时也全随机，先把 replay buffer 填起来。
- `argmax_policy.py`：评估/执行时直接取 `q_net` 输出的 argmax。

## 6. 实验里预期看到的

Q1 画 DQN vs DDQN 各 3 seed 均值 ± std。预期：DQN 可能更早冲到 ~125 然后回落或震荡大，
DDQN 上升稍慢但更稳、后期更高。写报告时用第 4 节"过估计"的逻辑解释。
