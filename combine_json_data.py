from io import StringIO
import pandas as pd

def get_json_df(json_path, new_file_name):
    with open(json_path, 'r', encoding='utf-8') as file:
        json_content = file.read()

    json_io = StringIO(json_content)
    df = pd.read_json(json_io, lines=True)
    df = df["title"]
    df.to_csv(f"{new_file_name}", index=False)
    

get_json_df("data/valid.json", "data/stage_valid.csv")
get_json_df("data/train.json", "data/stage_train.csv")
get_json_df("data/test.json", "data/stage_test.csv")


def combine_df(df1, df2, new_file_name):
    df1 = pd.read_csv(df1)
    df2 = pd.read_csv(df2)
    df = pd.concat([df1, df2])
    df.to_csv(f"{new_file_name}", index=False)
    
combine_df("data/stage_valid.csv", "data/stage_train.csv", "data/stage_valid_train.csv")
combine_df("data/stage_valid_train.csv", "data/stage_test.csv", "data/full_maven.csv")