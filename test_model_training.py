import unittest
from model_training import train_model
from sklearn.linear_model import LogisticRegression

class test_model_training(unittest.TestCase):
    def test_training_multiple_model(self):
        X_train = [[0,0], [1,1]]
        y_train = [0, 1]
        X_test = [[0,0], [1,1]]
        y_test = [0, 1]
        
        some_model, some_score = train_model(LogisticRegression, X_train, X_test, y_train, y_test, solver='lbfgs')
        
        self.assertIsInstance(some_model, LogisticRegression)
        self.assertGreater(some_score, 99)
        
if __name__ == "__main__":
    unittest.main()
        