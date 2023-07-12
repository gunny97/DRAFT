import os
import random
import pandas as pd




def load_all_data():
    mypath = f"/home/local/anaconda3/envs/crawl/crawler/results_output"

    all_data_list = []
    for category in os.listdir(mypath):
        if category != 'general':
            for sub_cat in os.listdir(mypath + '/' + category):
                # all_data_list.append(os.listdir(mypath + '/' + category + '/' + sub_cat))
                file_name = [mypath + '/' + category + '/' + sub_cat + '/' + f for f in os.listdir(mypath + '/' + category + '/' + sub_cat)]
                all_data_list.append(file_name)

    all_data_list = sum(all_data_list,[])

    return all_data_list

def main(all_data_list, main_cat, sub_cat):
    mypath = f"/home/local/anaconda3/envs/crawl/crawler/results_output/{main_cat+'/'+sub_cat}"
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for file in onlyfiles:
        path = mypath + "/" + file
        data = pd.read_csv(path)

        pos_num = data.shape[0]
        easy_neg_num = pos_num // 2
        hard_neg_num = pos_num// 2
        

        # Easy sample
        easy_data_list = [ele for ele in all_data_list if sub_cat not in ele]
        easy_neg_data = pd.read_csv(random.sample(easy_data_list, 1)[0])
        while easy_neg_data.shape[0] < easy_neg_num:
            tmp = pd.read_csv(random.sample(easy_data_list, 1)[0])
            easy_neg_data = pd.concat([easy_neg_data,tmp],axis=0)
            easy_neg_data.drop_duplicates(['text'],keep='first',inplace=True)
            
        easy_neg_data = easy_neg_data.iloc[:easy_neg_num]

        # hard sample
        hard_data_list = [ele for ele in all_data_list if sub_cat in ele]
        hard_path = random.sample(hard_data_list, 1)[0]
        hard_neg_data = pd.read_csv(hard_path)
        while hard_neg_data.shape[0] < hard_neg_num:
            tmp = pd.read_csv(random.sample(hard_data_list, 1)[0])
            hard_neg_data = pd.concat([hard_neg_data,tmp],axis=0)
            hard_neg_data.drop_duplicates(['text'],keep='first',inplace=True)

        hard_neg_data = hard_neg_data.iloc[:hard_neg_num]


        data['columns'] = 0
        easy_neg_data['columns'] = 1
        hard_neg_data['columns'] = 2
        test_data = pd.concat([data,easy_neg_data,hard_neg_data],axis=0)
        test_data.drop(['Unnamed: 0','length'],axis=1,inplace=True)
        test_data.loc[test_data.shape[0]+1] = [hard_path.split('/')[-1].replace('.csv',''),0]
        test_data.to_csv(f"/home/local/anaconda3/envs/crawl/crawler/test_data/{main_cat + '/' + file.split('.csv')[0]}.csv")


if __name__ == "__main__":
    
    all_data_list = load_all_data()
    main_category_list = ['world', 'science', 'history', 'lifestyle', 'nature', 'general']
    for main_cat in main_category_list:
        if main_cat == 'world':
            edge_cateogry_list = ['cities', 'countries', 'landmarks', 'us-states']
        elif main_cat == 'science':
            edge_cateogry_list = ['biology', 'chemistry', 'geography', 'physics', 'technology']
        elif main_cat == 'history':
            edge_cateogry_list = ['culture', 'historical-events', 'people', 'religion']
        elif main_cat == 'lifestyle':
            edge_cateogry_list = ['entertainment', 'food', 'health', 'sports']
        elif main_cat == 'nature':
            edge_cateogry_list = ['animals', 'human-body', 'plants', 'universe']

        for sub_cat in edge_cateogry_list:
            if main_cat == 'general':
                pass
            else:
                main(all_data_list, main_cat , sub_cat)
