import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import torch
import argparse
import time

parser = argparse.ArgumentParser(description='BGM Recommendation given Text')

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--ckpt_path',
                        type=str,
                        default="monologg/kobigbird-bert-base",
                        help='need existed pretrained model ckpt')

        parser.add_argument('--finetuned_model_ckpt',
                        type=str,
                        default="bigbird_0707",
                        help='need existed finetuned model ckpt')

        parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        help='enter device type')

        parser.add_argument('--num_songs',
                            type=int,
                            default=3,
                            help='number of recommened song')
        return parser


def candidate_song_data():
    candidate_data = pd.read_csv('/home/keonwoo/anaconda3/envs/bgmRS/data/Audio library - 500.csv',encoding='utf-8')
    candidate_data.drop(['Unnamed: 0','순서'],axis=1,inplace=True)
    candidate_data = candidate_data.fillna(0)

    mood_list = np.unique(candidate_data['메인무드'])
    mood_list_ko = ['쿨한', '신나는', '우스운', '멋진', '평화로운', '로맨틱한', '무서운, 소름 끼치는, 미스터리한', '감동적인', '희망적인']
    mood_dict = dict(zip(mood_list, mood_list_ko))

    def change_label(x):
        return mood_dict[x]
        
    candidate_data['메인무드'] = candidate_data['메인무드'].apply(lambda x: change_label(x))
    return candidate_data


def recommend_mood(text, model, tokenizer):

    candidate_data = candidate_song_data()

    if '\n' in text:
        text = text.replace('\n','')

    label_dict = {
        '희망적인' : 0,  
        '감동적인' : 1, 
        '무서운, 소름 끼치는, 미스터리한' : 2,
        '로맨틱한' : 3, 
        '평화로운' : 4,
        '멋진' : 5, 
        '우스운' : 6,
        '신나는' : 7, 
        '쿨한' : 8 
    }

    reversed_dict = dict(map(reversed, label_dict.items()))

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    model.eval()
    with torch.no_grad():
        model_output = model(**encoded_input)

    output = torch.argmax(model_output[0][0])

    pred_mood = reversed_dict[int(output.numpy())]

    recommended_song_df = candidate_data[candidate_data['메인무드'] == pred_mood]

    print("적절한 테마를 고르시오: ", np.unique(recommended_song_df['테마']))
    
    return recommended_song_df

def recommend_song(recommended_song_df, tema, num_song=3):
    # tema = 'News / Current Affair'
    # num_song = 3

    final_recommended = recommended_song_df[recommended_song_df['테마'] == tema]

    if len(final_recommended) < num_song:
        print(final_recommended['곡제목(임시)'])
        RS = final_recommended['곡제목(임시)']
    else:
        print(final_recommended.sample(n=num_song)['곡제목(임시)'])
        RS = final_recommended.sample(n=num_song)['곡제목(임시)']

    return RS 

def main(text, ckpt_path, finetuned_model_ckpt, num_songs, device):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(f"/home/keonwoo/anaconda3/envs/bgmRS/ckpt/{finetuned_model_ckpt}", num_labels=9)
    model.to(device)

    st1 = time.time()
    recommended_song_df = recommend_mood(text, model, tokenizer)
    print('Recommend Mood Execution time:', time.time()-st1, 'seconds')
    tema = str(input())

    st2 = time.time()
    RS = recommend_song(recommended_song_df, tema, num_song=num_songs)
    print('Recommend Songs Execution time:', time.time()-st2, 'seconds')

    return RS

if __name__ == '__main__':
    

    parser = ArgsBase.add_model_specific_args(parser)
    args = parser.parse_args()


    text =  """
    강호순과 조두순의 공통점은? 둘 다 동물을 학대 했다는 점이다. 2006년부터 3년간 부녀자 등 10명을 살해한 강호순은 첫 범행에 앞서 자신이 운영하는 개 사육장에서 개들을 잔인하게 죽였다고 알려졌다. 강호순은 프로파일러 면담 중에도 "개를 많이 죽이다 보니 사람 죽이는 것도 아무렇지 않고 살인 욕구를 자제할 수 없었다"고 했다.

    아동성범죄자 조두순은 반려견의 눈을 찔러 죽였다. 조두순은 조사 중에도 검사가 '술에 취해 자신도 모르게 이상한 행동을 한 적이 있냐'고 묻자 "강아지에게 병을 집어던져 죽인 적이 두 번 있었다" "그중 한 마리의 눈을 빗자루 몽둥이로 찔러 죽였다"라고 했다.

    전문가들은 동물을 향한 폭력성이 사람에게 향할 수 있다고 경고한다. 인명피해가 없다고 동물학대범를 가벼이 여겨서는 안되는 이유다.
    """
    main(text, args.ckpt_path, args.finetuned_model_ckpt, args.num_songs, args.device)
    
    # python text2bgm.py --ckpt_path=/home/keonwoo/anaconda3/envs/KoDiffCSE/sroberta_change_lr --finetuned_model_ckpt=koDiffCSE_0802 --device=cpu --num_songs=3