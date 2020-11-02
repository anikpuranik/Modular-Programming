import unittest
from Titanic_Refactored import prepare_and_test_data


class TestFunctional(unittest.TestCase):
    #@unittest.skip("temporarily skipping test")
    def test_data_preparation_and_model_training(self):
        scores = prepare_and_test_data()
        for score in scores:
            self.assertLess(70, score)
            
if __name__ == "__main__":
    unittest.main()