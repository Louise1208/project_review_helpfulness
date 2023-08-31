import mysql.connector
import pandas as pd
from sklearn.model_selection  import train_test_split
import torch

# 从MySQL数据库中读取数据
def read_data_from_mysql(table_name):
    # 设置本地MySQL数据库连接信息
    host = 'localhost'  # 替换为本地MySQL数据库的主机名
    user = 'root'  # 替换为数据库用户名
    password = '991208louise'  # 替换为数据库密码
    database = 'amazon'  # 替换为数据库名称

    # 建立数据库连接
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
    # Read data from MySQL into a DataFrame
    query = f"SELECT * FROM {table_name};"
    df = pd.read_sql(query, conn)
    # df.drop(columns=['reviewerID'], inplace=True)

    # Close the database connection
    conn.close()
    temp=1
    if table_name=='features_cell_phones' or table_name=='features_video_games':
        temp=0.1
    elif table_name=='features_books' or table_name=='features_electronics':
        temp=0.01

    df.drop(columns=['cleaned_review'], inplace=True)
    _, df = train_test_split(df, test_size=temp, random_state=42)
    print(f'load {table_name} from MySQL: {df.shape}!')

    return df


# split data into train, validation and test sets, the ratio is 8:1:1 , bast for large dataset
def split_data(df):
    y=df['helpfulness_score']
    X=df.drop(columns=['helpfulness_score'])
    test_ratio = 0.2
    validation_ratio = 0.1
    # split data into train, validation and test sets
    X_temp, X_test, y_temp,y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
    X_train,X_validation,y_train,y_validation = train_test_split(X_temp, y_temp, test_size=validation_ratio, random_state=42)
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def null_check(df):
    # 检查是否有空值
    null_values = df.isnull().any()

    # 输出包含空值的列
    columns_with_null = null_values[null_values]
    if columns_with_null.empty:
        print('No null values found in the dataset.')
    else:
        print('Null values found in the following columns:\n', columns_with_null)
#          fill null with mean
        df.fillna(df.mean(), inplace=True)
    return df

def features_selection(table_name,df,tag='all'):
    df_selected = None
    if tag == 'Gini':
        if table_name == 'features_cell_phones':
            # Timeline, Subjectivity, Review volume, Review length, Avg. sentence length
            # 选择需要的特征列
            selected_features = ['reviewText','helpfulness_score',"timeline", "subjectivity", "review_volume", "review_length", "avg_sentence_length"]
            # 从原始 DataFrame 中提取需要的特征列
            df_selected = df[selected_features]
        elif table_name == 'features_electronics':
            # Timeline, Subjectivity, Review volume, Review length, Overall
            selected_features = ['reviewText','helpfulness_score',"timeline", "subjectivity", "review_volume", "review_length", "overall"]
            df_selected = df[selected_features]
        elif table_name == 'features_video_games':
            # Review volume, Review length, Timeline, Subjectivity, Avg. sentence length
            selected_features = ['reviewText','helpfulness_score',"review_volume", "review_length", "timeline", "subjectivity", "avg_sentence_length"]
            df_selected = df[selected_features]
        elif table_name == 'features_books':
            # Review volume, Review length, Overall, Timeline, Subjectivity
            selected_features = ['reviewText','helpfulness_score',"review_volume", "review_length", "overall", "timeline", "subjectivity"]
            df_selected = df[selected_features]
    elif tag == 'PCC':
        if table_name == 'features_cell_phones':
            # Overall,readability, Num. of sentences, ,Timeline, Review length
            # 选择需要的特征列
            selected_features = ['reviewText','helpfulness_score',"overall",  "readability","num_sentences","timeline", "review_length"]
            # 从原始 DataFrame 中提取需要的特征列
            df_selected = df[selected_features]
        elif table_name == 'features_electronics':
            # Overall, Review length, Num. of sentences, Readability,timeline
            selected_features = ['reviewText','helpfulness_score',"overall", "review_length", "num_sentences", "readability","timeline"]
            df_selected = df[selected_features]
        elif table_name == 'features_video_games':
            #Review length, Num. of sentences, Readability, Overall, Timeline
            selected_features = ['reviewText','helpfulness_score',"review_length", "num_sentences", "readability", "overall",'timeline']
            df_selected = df[selected_features]
        elif table_name == 'features_books':
            # Overall, Review length, Num. of sentences, Readability,avg_sentence_length
            selected_features = ['reviewText','helpfulness_score',"overall", "review_length", "num_sentences", "readability", "avg_sentence_length"]
            df_selected = df[selected_features]
    elif tag=='SPC':
        if table_name == 'features_cell_phones':
    #         review Volume,Overall, Timeline, Review length, Num. of sentences
            selected_features = ['reviewText','helpfulness_score',"review_volume", "overall", "timeline", "review_length", "num_sentences"]
            df_selected = df[selected_features]
        elif table_name == 'features_electronics':
    #         review Volume,Overall, readability, Review length, Num. of sentences
            selected_features = ['reviewText','helpfulness_score',"review_volume", "overall", "readability", "review_length", "num_sentences"]
            df_selected = df[selected_features]
        elif table_name == 'features_video_games':
    #         review Volume,Overall, readability, Review length, Num. of sentences
            selected_features = ['reviewText','helpfulness_score',"review_volume", "overall", "timeline", "review_length", "num_sentences"]
            df_selected = df[selected_features]
        elif table_name == 'features_books':
            # review Volume,Overall, readability, Review length, Num. of sentences
            selected_features = ['reviewText','helpfulness_score',"review_volume", "overall", "readability", "review_length", "num_sentences"]
            df_selected = df[selected_features]
    elif tag=='all':
            df_selected=df
    elif tag=='none':
            df_selected=df[['reviewText','helpfulness_score']]

    # print(df_selected.dtypes)
    # print(df_selected.shape)
    # print(df_selected.head())
    print('features selection finished!')
    return df_selected


def data_load(table_name,tag='all'):
    df=read_data_from_mysql(table_name)
    df=features_selection(tag=tag,table_name=table_name,df=df)
    df=null_check(df)
    X_train, X_validation, X_test, y_train, y_validation, y_test=split_data(df)
    print('shape of X_train:',X_train.shape)
    print('shape of X_validation:', X_validation.shape)
    print('shape of X_test:', X_test.shape)
    print('shape of y_train:', y_train.shape)
    print('shape of y_validation:', y_validation.shape)
    print('shape of y_test:', y_test.shape)
    print(X_train.columns)
    # with open('evaluation_result_SVR.txt', 'a') as f:
    #     f.write(table_name)
    #     f.write('       tag: '+tag + '\n')
    return X_train, X_validation, X_test, y_train, y_validation, y_test


