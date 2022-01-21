import pandas as pd
from tqdm import tqdm


def clear_dataframe(df_names):
    cleared_df=pd.DataFrame(columns=['filename', 'gender'])

    
    for name in df_names:
        df= pd.read_csv("data/"+ name)

        for ind in tqdm(df.index,desc=str(name)):

            if((not  pd.isnull(df['gender'].iloc[ind]))):
                if df['gender'].iloc[ind] != 'other':
                    filename, gender =  df['filename'].iloc[ind], df['gender'].iloc[ind]#name[:-4] +'/'+
        
                    cleared_df = cleared_df.append({'filename': filename,'gender': gender}, ignore_index=True)

    #prepared_df.to_csv("prepared_dataframe.csv")
    print('\ncleared df info:')
    get_df_info(cleared_df)

    return cleared_df

def get_df_info(df):

    print("total samples:",  len(df))
    print("total male samples:", len(df[df['gender'] == 'male']))
    print("total female samples:",len(df[df['gender'] == 'female']))





def balance_classes(df):
    balanced_df=pd.DataFrame(columns=['filename', 'gender'])

    male_samples_num = len(df[df['gender'] == 'male'])
    female_samples_num = len(df[df['gender'] == 'female'])

    if male_samples_num > female_samples_num:
        superior_gender = 'male'
        less_gender_samples_num = female_samples_num
    else:
        superior_gender = 'female'
        less_gender_samples_num = male_samples_num

    more_gender_samples_num= 0
    for ind in tqdm(df.index,desc='balancing classes'):
            filename, gender = df['filename'].iloc[ind], df['gender'].iloc[ind]
            
            if gender == superior_gender:

                more_gender_samples_num+=1
                if more_gender_samples_num <= less_gender_samples_num:

                    balanced_df = balanced_df.append({'filename': filename,'gender': gender}, ignore_index=True)
                
            else:   
                balanced_df = balanced_df.append({'filename': filename,'gender': gender}, ignore_index=True)
    
    print('\nbalanced df info:')
    get_df_info(balanced_df)

    return balanced_df


def divide_into_2_frames(df):
    test_df = pd.DataFrame(columns=['filename', 'gender'])
    indexes = []

    male_samples_num,female_samples_num = 0,0

    for ind in tqdm(df.index,desc='dividing into 2 frames'):

        if df['gender'].iloc[ind] == 'male' and male_samples_num<7500:
            filename, gender =  df['filename'].iloc[ind], df['gender'].iloc[ind]

            test_df = test_df.append({'filename': filename,'gender': gender}, ignore_index=True)

            male_samples_num+=1
            indexes.append(ind)

        if df['gender'].iloc[ind] == 'female' and female_samples_num<7500:
            filename, gender =  df['filename'].iloc[ind], df['gender'].iloc[ind]

            test_df = test_df.append({'filename': filename,'gender': gender}, ignore_index=True)

            female_samples_num+=1
            indexes.append(ind)
    
    train_df = df.drop(df.index[indexes])

    print('\ntrain df info:')
    get_df_info(train_df)

    print('\ntest df info:')
    get_df_info(test_df)

    return train_df, test_df
    



df_names = ["cv-other-train.csv","cv-valid-train.csv"]
cleared_df = clear_dataframe(df_names)
balanced_df = balance_classes(cleared_df)

# balanced_df.to_csv("balanced_dataframe.csv")

train_df,test_df = divide_into_2_frames(balanced_df)

train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

train_df.to_csv("train_dataframe.csv")
test_df.to_csv("test_dataframe.csv")


