## 对Loss函数推导的理解
首先我们要度量或者说利用起来人类相比于 $y_w$ 对 $y_l$ 更加偏好的信息,使用BT模型

$ P(y_w > y_l | x) = \frac{1}{1 + exp(r(x,y_l) - r(x,y_w))} $

根据对数似然估计则有

$ L_p(r_\phi , D) = -E_{x,y_w,y_l -d }[\log\sigma(r_\phi(x,y_w)- r_\phi(x,y_l)] $

而进入到RL Fine——Tuning阶段我们有这样的优化目标

$ \max_{\pi_\theta} E_{x-d , y-\pi_{\theta}(y|x)}[r_\phi(x,y)] - \beta D_{KL}[\pi_\theta(y|x) ||\pi_{ref}(y|x)]  $

经过一番化简后我们得到

$ \min_{\pi_\theta}E_{x-d}[D_{KL}(\pi_\theta(y|x) ||\pi^*(y|x)) ] $表示我们希望的$\pi_\theta$要接近于$\pi^*$

其中 $ \pi^*(y|x) = \frac{\pi_{ref}(y|x)exp(\frac{r(x,y)}{\beta})}{\sum_y\pi_{ref}(y|x)exp(\frac{r(x,y)}{\beta})} $
这一项将优化奖励与优化概率联系起来
根据这一项得到

$r(x,y)=\beta \ln \frac{\pi^*(y|x)}{\pi_{ref}(y|x)}  + \beta\ln Z(x)$

其中$Z(x)= \sum_y\pi_{ref}(y|x)exp(\frac{r(x,y)}{\beta})$在前面的化简中用到过

结合刚刚建立的奖励与概率之间的联系和之前的损失函数得到

$ L_{DPO} = -E_{x,y_w,y_l ~ D}[\ln \sigma(\beta\ln \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)})-\beta\ln \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}]$

##  对加入KL散度的理解
- KL散度：一种衡量两个概率分布之间差异的统计量
- 1. 通过KL散度防止新的模型偏离原来的参考模型太远
  2. 通过增加KL散度防止生成集中在某几个少数的模型学习到的最优解，增添回答的多样性
  3. KL散度项限制了策略在参数空间中的更新范围。它不允许模型为了迎合奖励函数而进行剧烈的、不受约束的参数调整，防止了过拟合

## 对RL在LLM上运用的理解
-  运用一：偏好学习，通过人类相对偏好把模型输出朝人类更喜欢的方向调整。本文就是以此为中心提出DPO，目标是让生成更符合人类偏好
-  运用二：属性控制，例如情感控制（比如把评论改为更正面），本文的 controlled sentiment 任务就是示例。
-  运用三：训练安全/无害策略，可以用偏好数据把模型往安全、礼貌、精炼等方向偏移
-  本文也解释了RL可以学习人类偏好的reward，本文通过提出DPO算法建立了reward与策略之间的联系。证明了在策略空间做最大似然就同时进行了对reward的学习
-  RL应该不能在未经预训练的基础上完成对LLM的训练，文中使用DPO算法是建立在得到$\pi_{ref}$的基础上的需要经过监督式微调
## 论文中实验部分
- 在情感控制、摘要和对话任务上，DPO 相比 PPO 或 Preferred-FT能在更低的KL偏差下得到更多奖励
- 在跨分布迁移（CNN/DailyMail 新闻摘要）实验中，DPO在没有额外未标注提示的情况下仍显著优于PPO
- 人工评测验证表明 GPT-4 评判与人类一致性高，从侧面证明 DPO 的输出质量在真实人类偏好下也稳健
综上模型经过 DPO 这种后训练后，在未经偏好训练的数据输入（分布外任务或提示）上仍保持或提升性能