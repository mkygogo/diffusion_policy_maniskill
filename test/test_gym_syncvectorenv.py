import gymnasium as gym
import numpy as np

# 创建训练用的向量化环境（4个并行环境）
train_envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])

# 创建单独的可视化环境
vis_env = gym.make("CartPole-v1", render_mode="human")

# 简单的随机策略模型（实际使用时替换为你的RL模型）
class DummyModel:
    def predict(self, obs):
        # 这里使用随机策略作为示例
        # obs可能是单个观测(可视化时)或批量观测(训练时)
        if len(obs.shape) == 1:  # 单个观测
            return np.random.randint(0, 2)
        else:  # 批量观测
            return np.random.randint(0, 2, size=obs.shape[0])

model = DummyModel()

# 训练循环
num_episodes = 50
for episode in range(num_episodes):
    # 向量化环境reset返回批量观测
    obs = train_envs.reset()
    done = np.zeros(4, dtype=bool)
    episode_rewards = np.zeros(4)
    
    # 训练阶段
    while not all(done):
        # 获取批量动作 (形状应为 (4,))
        actions = model.predict(obs)
        
        # 向量化环境step
        obs, rewards, dones, infos = train_envs.step(actions)
        
        # 累计奖励
        episode_rewards += rewards * (1 - done)  # 只累计未完成环境的奖励
        done = np.logical_or(done, dones)
    
    print(f"Episode {episode+1}, Avg Reward: {np.mean(episode_rewards):.1f}")
    
    # 每10个episode用可视化环境测试
    if (episode + 1) % 10 == 0:
        print("Running visualization...")
        obs = vis_env.reset()
        total_reward = 0
        
        while True:
            action = model.predict(obs)
            obs, reward, done, _ = vis_env.step(action)
            total_reward += reward
            
            if done:
                print(f"Visualization episode reward: {total_reward}")
                break

# 关闭环境
train_envs.close()
vis_env.close()

