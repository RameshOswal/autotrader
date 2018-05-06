import numpy as np
class RandomModel:
    def __init__(self, num_weights=5, num_time_steps=10):
        self.num_weights = num_weights
        self.num_time_steps = num_time_steps
        self.weights = np.ones((num_time_steps, num_weights))
    def compute_allocation_weights(self):
        self.weights = np.random.uniform(size=(self.num_time_steps, self.num_weights), high=1, low=0)
        #normalize the weights
        self.weights = self.weights/self.weights.sum(axis=1).reshape(-1,1)
        return self.weights
    
if __name__ == "__main__":
    randomModel = RandomModel()
    weights = randomModel.compute_allocation_weights()#.sum(axis=1)
    print(weights.shape)
    print(weights)