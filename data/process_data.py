import sys
import pandas as pd
from sqlalchemy import create_engine


def process_data(file_1, file_2, database):
    # load messages dataset
    messages = pd.read_csv(file_1)
    # load messages dataset
    categories = pd.read_csv(file_2)
    
    print("Files loaded successfully...")
    
    df = pd.merge(messages, categories, on='id')
    
    categories = df['categories'].str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # Asign columns names
    categories.columns = [x[:-2] for x in row.values]
    
    for column in categories:
    # set each value to be the last character of the string and convert to numeric
        categories[column] = categories[column].str[-1].astype(int)

    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    
    print("Data cleaned successfully...")
    
    engine = create_engine(f'sqlite:///{database}')
    df.to_sql('cleaned_data', engine, index=False, if_exists='replace')
    
    print("Data stored... ETL proccess completed!")

    
if __name__ == "__main__":
    file_1, file_2, database = sys.argv[1:]  # get filename of dataset
    process_data(file_1, file_2, database)   # run ETL