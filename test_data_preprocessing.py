import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from data_preprocessing import extract_title_from_name, add_is_alone_column, continous_category, add_categorical_columns, fill_nan_with_median, fill_nans

class TestDataPreprocessing(unittest.TestCase):
    def test_extract_title_from_name(self):
        input_df = pd.DataFrame({
            "Name": ["Smith, Mr. Owen Harris  ", "Heikkinen, Miss. Laina", "Allen, Mlle. Maisie",
                     "Allen, Ms. Maisie", "Allen, Mme. Maisie"
                     
                     # rare titles
                     "Smith, Lady. Owen Harris   ", "Heikkinen, Countless. X ", "Allen, Capt. Maisie",
                     "Smith, Col. Owen Harris    ", "Heikkinen, Don. Laina   ", "Allen, Dr. Maisie",
                     "Smith, Major. Owen Harris  ", "Heikkinen, Rev. Laina   ", "Allen, Sir. Maisie",
                     "Smith, Jonkheer. Owen Bob  ", "Heikkinen, Dona. Laina  "
                     ]
        })
        
        actual_df = extract_title_from_name(input_df)
        
        expected_df = pd.DataFrame({
            "Name": ["Smith, Mr. Owen Harris  ", "Heikkinen, Miss. Laina", "Allen, Mlle. Maisie",
                     "Allen, Ms. Maisie", "Allen, Mme. Maisie"
                     
                     "Smith, Lady. Owen Harris   ", "Heikkinen, Countless. X ", "Allen, Capt. Maisie",
                     "Smith, Col. Owen Harris    ", "Heikkinen, Don. Laina   ", "Allen, Dr. Maisie",
                     "Smith, Major. Owen Harris  ", "Heikkinen, Rev. Laina   ", "Allen, Sir. Maisie",
                     "Smith, Jonkheer. Owen Bob  ", "Heikkinen, Dona. Laina  "
                     ],
            "Title": ["Mr", "Miss", "Miss",
                      "Miss", "Mrs",
                      
                      "Rare", "Rare", "Rare",
                      "Rare", "Rare", "Rare",
                      "Rare", "Rare", "Rare",
                      "Rare"]
            })
        
        assert_frame_equal(expected_df, actual_df)
        
    def test_add_is_alone_column(self):
        input_df = pd.DataFrame({
            "Parch": [0, 1, 0],
            "SibSp": [0, 0, 1]
            })
        
        actual_df = add_is_alone_column(input_df)
        
        expected_df = pd.DataFrame({
            "Parch":[0, 1, 0],
            "SibSp":[0, 0, 1],
            "IsAlone":[1, 0, 0]
            })
        
        assert_frame_equal(expected_df, actual_df)
        
    def test_continous_category(self):
        series = pd.Series([5, 20, 10, 25])
        
        assert_series_equal(
            pd.Series([1, 2, 1, 2]), continous_category(series, n_bias=2)
        )
        
    def test_add_cateogical_columns(self):
        input_df = pd.DataFrame({
            "Title": ["Mr", "Mrs", "Miss", "Rare", "Master"],
            "Embarked" : ["S", "Q", "C", "Q", "C"],
            "Sex" : ["female", "male", "male", "female", "female"]
        })
        
        expected_df = pd.DataFrame({
            "Title": [1, 3, 2, 5, 4],
            "Embarked": [0, 2, 1, 2, 1],
            "Sex": [1, 0, 0, 1, 1]
        })
                
        assert_frame_equal(expected_df, add_categorical_columns(input_df))
        
    def test_fill_nan_with_median(self):
        input_df = pd.DataFrame({
           "SomeColumn": [0, 10, 20, np.nan],
           "AnotherColumn": [0, 1, 2, np.nan],
           "YetAnotherColumn": [0, 1, 2, np.nan]
        })

        expected_df = pd.DataFrame({
            "SomeColumn": [0, 10, 20, 10],
            "AnotherColumn": [0, 1, 2, 1],
            "YetAnotherColumn": [0, 1, 2, np.nan]
        })
        
        actual_df = fill_nan_with_median(
            input_df, ["SomeColumn", "AnotherColumn"]
        )
                
        assert_frame_equal(expected_df, actual_df, check_dtype=False)
        
    def test_fill_nans_with_zero(self):
        input_df = pd.DataFrame({
            "Title": [1, np.nan, 2, 3, np.nan],
            "Embarked": [1, 2, 2, np.nan, np.nan],
            "SomeColumn": [0, 10, 20, np.nan, np.nan]
        })
        
        expected_df = pd.DataFrame({
            "Title": [1, 0, 2, 3, 0],
            "Embarked": [1, 2, 2, 2 ,2],
            "SomeColumn": [0, 10, 20, 10, 10]
        })
        
        assert_frame_equal(expected_df, fill_nans(input_df, median_columns=["SomeColumn"]), check_dtype=False)
        
if __name__ == "__main__":
    unittest.main()

    
    
    
    
    
    
    
    
    