import numpy as np

# Tạo class agent sử dụng epsilon-greedy để chọn hành động
class EpsilonGreedyAgent:
  def __init__(self, num_actions, epsilon=0.1):
    self.num_actions = num_actions 
    self.epsilon = epsilon 
    self.action_values = np.zeros(num_actions) 
    self.action_counts = np.zeros(num_actions) 

  def select_action(self): 
    if np.random.rand() < self.epsilon: 
      action = np.random.randint(self.num_actions)
    else: 
          action = np.argmax(self.action_values)
    return action

  def update_value(self, action, reward):
      self.action_counts[action] += 1 
      self.action_values[action] += (1 / self.action_counts[action]) * (reward - self.action_values[action])
      
# Mô phỏng môi trường multi-armed bandit
class MultiArmedBandit:
    def __init__(self, num_arms):
        self.num_arms = num_arms 
        self.true_action_values = np.random.normal(0, 1, num_arms) 

    # Mỗi khi chọn một cánh tay (action), môi trường sẽ trả về một phần thưởng. Phần thưởng này được lấy mẫu ngẫu nhiên từ phân phối chuẩn với trung bình là giá trị thật của cánh tay đó và phương sai là 1.
    def get_reward(self, action):
        reward = np.random.normal(self.true_action_values[action], 1)
        return reward
    
if __name__ == '__main__':
    # Khởi tạo môi trường và agent
    num_arms = 10 
    num_steps = 1000 
    agent = EpsilonGreedyAgent(num_arms) 

    bandit = MultiArmedBandit(num_arms) 
    total_rewards = 0 
    for step in range(num_steps):
        action = agent.select_action() 
        reward = bandit.get_reward(action) 
        agent.update_value(action, reward) 
        total_rewards += reward 

    print("Total rewards obtained:", total_rewards) 
    print("Estimated action values:", agent.action_values)