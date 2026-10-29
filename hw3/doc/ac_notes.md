# HW3 学习笔记：Actor-Critic

## 1. 从 hw2 的策略梯度出发

hw2 更新策略的公式：

```
∇J ≈ mean[ ∇ log π(a|s) · A(s,a) ]
```

**A 大的动作，提高它的概率；A 小（负）的动作，降低它的概率。**
A 叫 advantage，衡量"这个动作比平均水平好多少"。

hw2 里 A 这么算：

```
A(s_t, a_t) = (从 t 开始实际拿到的 reward 之和)  −  V(s_t)
              ↑ reward-to-go，用一整条真实轨迹算出来
```

问题：reward-to-go 是**一整条轨迹后面所有随机性的总和**——后面每一步策略的随机采样、
环境的随机性全混进来。同一个 (s, a) 跑两次，可能一次 +80 一次 −30。这就是"方差大"。
方差大 → 梯度估计噪声大 → 需要很多样本、学得慢。

## 2. 核心想法：用估计代替实际求和

reward-to-go 是"从 s_t 开始未来 reward 之和"，而 V(s) 定义上就是"从 s 开始未来 reward 之和的期望"。
直接拿 V 代替：

```
reward-to-go ≈ r_t + γ · V(s_{t+1})
                ↑ 真实的一步      ↑ 剩下的全部交给 V 估计
```

于是 advantage 变成：

```
A(s_t, a_t) = r_t + γ·V(s_{t+1}) − V(s_t)
```

只有 r_t 一个随机量，其余是网络输出的确定值。方差小很多。
代价是 V 估不准时会有偏差——RL 里经典的 **bias-variance 权衡**。

对应代码 `agents/ac_agent.py` → `estimate_advantage`：

```python
def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
    v_t   = self.critic.forward_np(ob_no)         # V(s)
    v_tp1 = self.critic.forward_np(next_ob_no)    # V(s')
    q_n   = re_n + self.gamma * v_tp1 * (1 - terminal_n)   # r + γV(s')，终止时截掉 V(s')
    adv_n = q_n - v_t                             # A = Q − V

    if self.standardize_advantages:
        adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
    return adv_n
```

## 3. 这里用 Q 了吗？

用了，但**没有单独学一个 Q 网络**。Q 是用 V 拼出来的。

```
Q(s, a) = 在 s 做动作 a 之后，未来 reward 之和的期望
V(s)    = 在 s 按策略 π 行动，未来 reward 之和的期望
```

精确关系——Q 可以拆成"这一步的 reward"加"下一个状态的 V"：

```
Q(s, a) = r(s, a) + γ · V(s')
```

做完 a 拿到 r，到了 s'，之后按策略走，剩下的期望就是 V(s')。
代码里那行变量名就叫 `q_n`。advantage 的本来定义就是 **A = Q − V**：

```
A(s,a) = Q(s,a) − V(s) = [r + γV(s')] − V(s)
```

"这个动作比平均水平好多少" = "做这个动作能拿到的" − "平均能拿到的"。

**为什么不直接学 Q 网络？**
1. 省一个网络。有 V，Q 一行就能算。
2. V 比 Q 好学。V 只依赖 s；Q 依赖 (s, a)，连续动作时 a 有无穷多取值，要覆盖的空间大得多。

| | DQN | Actor-Critic（本作业） |
|---|---|---|
| 学的网络 | Q(s, a) | V(s) |
| Q 从哪来 | 网络直接输出 | `r + γV(s')` 拼出来 |
| Q 拿来干嘛 | 直接 argmax 当策略 | 减去 V 得 A，喂给 actor 做策略梯度 |

DQN 必须学 Q，因为它没有单独的策略，全靠 `argmax_a Q(s,a)` 选动作，V 不带动作做不到。
Actor-critic 有自己的策略网络，不靠 Q 选动作，只学 V 就够。
（有些变体确实学 Q 而不是 V，如 SAC/DDPG，那是另一套思路。）

## 4. 两个角色

| | 是什么 | 学什么 | 代码 |
|---|---|---|---|
| **Actor** | 策略网络 π(a\|s) | 用 A 做策略梯度，和 hw2 一样 | `policies/MLP_policy.py` → `MLPPolicyAC.update` |
| **Critic** | 价值网络 V(s) | 回归到 TD 目标 `r + γV(s')` | `critics/bootstrapped_continuous_critic.py` → `update` |

Actor 负责"做决定"，Critic 负责"打分"告诉 Actor 刚才那步好不好。

### Actor

```python
class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        adv_n = ptu.from_numpy(adv_n)

        action_distribution = self.forward(observations)     # π(·|s)：离散是 Categorical，连续是高斯
        log_prob = action_distribution.log_prob(actions)     # log π(a|s)
        loss = -(log_prob * adv_n).mean()                    # 最大化 E[log π · A]，取负给 optimizer 最小化

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

和 hw2 的 `MLPPolicyPG.update` 完全一样，只是 advantage 从外面传进来，没有 nn_baseline（baseline 的活交给 critic 了）。

### Critic：又是"自己教自己"

Critic 的目标值 `y = r + γ · V(s')` 里面有 V 自己——和 DQN 一样的问题：你往目标走一步，目标跟着动。

DQN 的解法是冻结的 target 网络。这里用更轻量的办法：**先冻住目标算几步，再重算目标**：

```python
def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
    ob_no      = ptu.from_numpy(ob_no)
    next_ob_no = ptu.from_numpy(next_ob_no)
    reward_n   = ptu.from_numpy(reward_n)
    terminal_n = ptu.from_numpy(terminal_n)

    for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):  # 10 × 10 = 100 步
        if i % self.num_grad_steps_per_target_update == 0:        # 每 10 步（含第 0 步）
            with torch.no_grad():                                 # 冻住：目标是常数，不传梯度
                v_tp1 = self(next_ob_no)
                target_n = reward_n + self.gamma * v_tp1 * (1 - terminal_n)   # 重算目标

        v_t = self(ob_no)
        loss = self.loss(v_t, target_n)                           # MSE，向固定目标回归

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    return loss.item()
```

命令行里 `-ntu 10 -ngsptu 10` 就是 `num_target_updates`（重算目标 10 次）和
`num_grad_steps_per_target_update`（每次目标固定后走 10 步梯度）。
`with torch.no_grad()` 等价于 DQN 里的 `target.detach()`。

作业 PDF 里描述的过程：
1. 用当前 V 算目标
2. 向目标回归几步
3. 重复 1、2 若干次

PDF 明确说"这个迭代过程对训练 critic 至关重要"——如果 ntu=1 只算一次目标，V 学到的是基于旧 V 的一个很差的估计。

## 5. `train` 把它们串起来

```python
def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
    loss = OrderedDict()
    for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
        critic_loss = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)   # ① 先训 critic

    adv_n = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)               # ② 用训好的 critic 算 A

    for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
        actor_loss = self.actor.update(ob_no, ac_na, adv_n)                            # ③ 用 A 更新 actor

    loss['Loss_Critic'] = critic_loss
    loss['Loss_Actor'] = actor_loss
    return loss
```

顺序重要：**先 critic 后 actor**。Critic 要先对当前策略打分打得准，actor 才有可靠的信号。

## 6. 和 DQN 对照

| | DQN | Actor-Critic |
|---|---|---|
| 学的是什么 | Q(s,a)，策略隐含在 argmax 里 | 显式的策略 π + 价值 V |
| 动作空间 | 离散（要对 a 取 max） | 离散/连续都行 |
| "目标里有自己"怎么处理 | 冻结的 target 网络 | 每 ngsptu 步重算一次目标 |
| 数据 | replay buffer，off-policy | 当前策略刚采的数据，on-policy |

### "AC 就是 DQN 把 Q 换成 V"？——对了一半

**一样的部分：critic 的 TD 回归**

```
DQN   : Q(s,a)  →  r + γ · max_a' Q(s',a')
Critic: V(s)    →  r + γ · V(s')
```

两边都是"网络输出 → 向 `r + γ·(下一步的自己)` 回归"，都有"目标里有自己"的问题，
都用"把目标冻住一会儿"处理（DQN 用 target 网络，这里每 ngsptu 步重算一次）。

细微差别：DQN 的目标里有 **max**，V 的目标里没有。V 定义就是"按当前策略走"，
下一步做什么由 π 决定，不用挑最优。所以 critic 不存在"过估计"问题，不需要 double 技巧。

**不一样的部分：学完之后怎么用**

| | DQN | Actor-Critic |
|---|---|---|
| 学到 Q/V 之后 | **直接完事**。策略就是 `argmax_a Q(s,a)` | **只走了一半**。V 不带动作，没法直接选动作 |
| 动作怎么来 | 从 Q 里 argmax 出来 | 另有策略网络 π，用 `A = r + γV(s') − V(s)` 做策略梯度训它 |

DQN 是**一个网络、一个更新公式**；actor-critic 是**两个网络、两个更新公式**：

```
① critic:  V(s) 向 r + γV(s') 回归            ← 和 DQN 相似
② actor :  ∇ log π(a|s) · [r + γV(s') − V(s)]  ← DQN 里没有，来自 hw2
```

"Q 变成 V、更新公式相似"只描述了 ①。真正的新东西是 ②：
**V 只负责给动作打分，它自己不选动作，选动作的是 actor。**
这也是它能处理连续动作的原因——没法对连续 a 取 max，但策略网络可以直接输出一个高斯分布。

### "AC 本质还是学 Q table"？——反了

Actor-critic **本质是学策略**，学 V 只是辅助。

**RL 的三条路：**

| 路线 | 学什么 | 动作怎么来 | 代表 |
|---|---|---|---|
| **Value-based** | 只学 Q(s,a) | `argmax_a Q` 隐含出来，没有显式策略 | DQN（hw3 Part 1） |
| **Policy-based** | 只学 π(a\|s) | 直接从 π 采样 | REINFORCE / 策略梯度（hw2） |
| **Actor-Critic** | 学 π **和** V | 从 π 采样，V 只用来帮 π 算梯度 | hw3 Part 2 |

Actor-critic 是 policy-based 的延伸，不是 value-based 的延伸。

**怎么判断谁是本质：把一边拿掉试试。**

- 拿掉 critic（V）：退化成 hw2 的策略梯度，advantage 用 reward-to-go 算。**照样能跑**，就是方差大、慢一点。
  hw2 的 CartPole 就是这么做的，也到了 200。
- 拿掉 actor（π）：只剩 V(s)。V 不带动作，**没法选动作**，什么都干不了。

所以 actor 不可或缺，critic 是可去掉的辅助。本质是学策略。

**V 在 AC 里到底干嘛。** 回到策略梯度公式：

```
∇J = mean[ ∇ log π(a|s) · A(s,a) ]
                          ↑ 只是一个权重，告诉 actor 往哪个方向、推多大
```

A 怎么算都不改变"我在学 π"这件事：
- hw2：A = 真实 reward 之和 − baseline（方差大）
- hw3：A = r + γV(s') − V(s)（方差小）

V 的全部作用是让这个**权重**更准、更稳。它不是被学来"用"的，是被学来"帮 actor 算梯度"的。
DQN 的 Q 学完直接当策略用，完全不同。

**和 Q table 的关系。** critic 的更新公式长得像 DQN，是因为**任何价值函数都靠 Bellman 方程学**，
这是 TD 学习的通用骨架，不是 DQN 专属。但：

- DQN：学 Q → Q 是最终产物 → 策略 = argmax Q
- AC：学 V → V 是中间产物 → 喂给 actor → **π 才是最终产物**

训练完 AC，部署的是 π 网络，V 可以扔掉。训练完 DQN，部署的是 Q 网络，没别的可部署。

**一句话：DQN 是"学价值，策略顺带出来"；Actor-critic 是"学策略，价值顺带帮忙"。**
两者学价值的手法相似，但谁主谁辅正好相反。

## 7. 实验里的现象

CartPole（Q2）跑出来 `Loss_Critic` 几十到两百、`Loss_Actor` 接近 0 且正负跳动，都正常：
- actor loss 本身没有"越小越好"的意义，只是梯度的载体
- critic loss 是 V 的回归误差，reward 尺度大它就大

Q2 CartPole-v0 期望 `Eval_AverageReturn` → 200；Q3 InvertedPendulum-v4 期望 100 轮后 ≈ 1000，
第 20 轮左右应已 ≥ 100。
