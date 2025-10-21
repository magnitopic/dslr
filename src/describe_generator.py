import pandas as pd # type: ignore

from Descriptive_funtions import *

def create_describe_file():
    data = pd.read_csv("./data/dataset_train.csv")
    print(data.describe())
    print(f"count:  {count(data['Index'].to_list()):.5f}")
    print(f"mean:  {mean(data['Index'].to_list()):.5f}")
    print(f"std:  {std(data['Index'].to_list()):.5f}")
    print(f"min:  {min(data['Index'].to_list()):.5f}")
    print(f"percentile-25:  {percentile(data['Index'].to_list(), 25):.5f}")
    print(f"percentile-50:  {percentile(data['Index'].to_list(), 50):.5f}")
    print(f"percentile-75:  {percentile(data['Index'].to_list(), 75):.5f}")
    print(f"max:  {max(data['Index'].to_list()):.5f}")

if __name__ == "__main__":
    create_describe_file()