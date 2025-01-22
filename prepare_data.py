from src.data_cleaner import DataCleaner

dir_path = "data/raw/"
output_path = "data/processed2/"

data_cleaner = DataCleaner(dir_path, output_path)
data_cleaner.load_data()
data_cleaner.clean_data()
data_cleaner.prepare_data()
data_cleaner.save_data()
