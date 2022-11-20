class BaseLayer():
    def __init__(self):
        self.trainable = False
        self.weights = np.array([])
        self.bach_size = 0