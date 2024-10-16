import numpy as np

# Tạo GridWorld (thế giới lưới) đơn giản kích thước 3x3
class GridWorld:
    def __init__(self):
        self.grid_size = (3, 3)
       
        self.num_actions = 4  
        self.rewards = np.array([
            [0, 2, 0],
            [0, 0, 0],
            [0, 1, 0]
        ])

    # Trả về phần thưởng tại trạng thái cụ thể. Ví dụ, nếu trạng thái là (2, 1), nó sẽ trả về 1
    def get_reward(self, state):
        return self.rewards[state[0], state[1]]
    
# Lưu giá trị của mỗi trạng thái dưới dạng mảng 2D có cùng size với GridWorld
class ValueFunction:
    def __init__(self, grid_size):
        self.values = np.zeros(grid_size) 
 
    def update_value(self, state, new_value):
        self.values[state[0], state[1]] = new_value

    def get_value(self, state):
        return self.values[state[0], state[1]]
    
if __name__ == '__main__':
    # Tạo môi trường GridWorld
    grid_world = GridWorld()

    value_function = ValueFunction(grid_world.grid_size)

    for i in range(grid_world.grid_size[0]):
        for j in range(grid_world.grid_size[1]):
            state = (i, j) # Vị trí hiện tại trong lưới
            value_function.update_value(state, grid_world.get_reward(state))

    print("Initial Value Function:")
    print(value_function.values)