from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
from sentence_splitter import SentenceSplitter, split_text_into_sentences
from tqdm.auto import tqdm



def page_url_crawl(page_url):

    URL_list = []
    go = True
    i = 1

    while go:
        try:
            if i == 1:
                print(f'{i}th page url crawling start')
                response = requests.get(page_url)
                soup = bs(response.text, 'html.parser')
                elements  = soup.select('#contents > div.facts-wrapper')
                fine_element = elements[0].select('div:nth-child(1) > div > a')
                url = [ele.attrs['href'].lstrip().rstrip() for ele in fine_element]
                URL_list.append(url)
                print(f'{i}th page url crawling done')
                i += 1
            else:
                print(f'{i}th page url crawling start')
                # import pdb 
                # pdb.set_trace()
                page_url_over_i = page_url + f'page/{i}/'
                response = requests.get(page_url_over_i)
                soup = bs(response.text, 'html.parser')
                elements  = soup.select('#contents > div.facts-wrapper')
                fine_element = elements[0].select('div:nth-child(1) > div > a')
                url = [ele.attrs['href'].lstrip().rstrip() for ele in fine_element]

                URL_list.append(url)
                print(f'{i}th page url crawling done')
                i += 1
        except:
            print(f'there is no {i}th page')
            go = False

    URL_list = sum(URL_list, [])
    return URL_list

def content_crawl(content_url):

    response = requests.get(content_url)
    soup = bs(response.text, 'html.parser')
    elements  = soup.select('#contents > div.content-wrap-flex > div.single-title-desc-wrap')
    if len(elements) == 1:
        title = [ele.text for ele in elements[0].select('h2')]
        paragraph = [ele.text for ele in elements[0].select('p')]
        query = [soup.select( f"#contents > div.content-wrap-flex > div.tabs > div.tabs-stage > #tab-1 > ol >li:nth-child({idx+1})")[0].text for idx in range(5)]

    else:
        title = [elements[i].select('h2')[0].text for i in range(len(elements))]
        paragraph = [elements[i].select('p')[0].text for i in range(len(elements))]
        query = [soup.select( f"#contents > div.content-wrap-flex > div.tabs > div.tabs-stage > #tab-1 > ol >li:nth-child({idx+1})")[0].text for idx in range(5)]

    # sentence_list = [splitter.split(para) for para in paragraph]
    # sentence_list = sum(sentence_list, [])
    sentence_list = paragraph
    sentence_list.extend(title) 
    
    return sentence_list, query

if __name__ == "__main__":

    import logging
    logger = logging.getLogger(__name__)

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler('./X_crawl.log')

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    splitter = SentenceSplitter(language='en')

    def custom(text) :
        return len(text)

    # "https://facts.net/world/", "https://facts.net/science/", "https://facts.net/history/", "https://facts.net/lifestyle/", "https://facts.net/nature/", "https://facts.net/general/"
    # page_url_list = ["https://facts.net/world/"]
    page_url_list = [
        "https://facts.net/world/cities/", "https://facts.net/world/countries/", "https://facts.net/world/landmarks/", "https://facts.net/world/us-states/",
        "https://facts.net/science/biology/", "https://facts.net/science/chemistry/", "https://facts.net/science/geography/", "https://facts.net/science/physics/", "https://facts.net/science/technology/", 
        "https://facts.net/history/culture/", "https://facts.net/history/historical-events/", "https://facts.net/history/people/", "https://facts.net/history/religion/",  
        "https://facts.net/lifestyle/entertainment/", "https://facts.net/lifestyle/food/", "https://facts.net/lifestyle/health/", "https://facts.net/lifestyle/sports/", 
        "https://facts.net/nature/animals/", "https://facts.net/nature/human-body/", "https://facts.net/nature/plants/", "https://facts.net/nature/universe/", 
        "https://facts.net/general/"
        ]

    for page_url in page_url_list:
        print('==================================================start   ', page_url, '==================================================')
        
        if 'general' not in page_url:
            sub_category = page_url.split('.net/')[1].split('/')[0]
            edge_category = page_url.split(sub_category)[1].replace('/','')
            basic_path = f"/home/local/anaconda3/envs/crawl/crawler/results_output/{sub_category + '/' + edge_category + '/'}"
        
        else:
            sub_category = page_url.split('.net/')[1].split('/')[0]
            basic_path = f"/home/local/anaconda3/envs/crawl/crawler/results_output/{sub_category + '/'}"

        url_list = page_url_crawl(page_url)
        print('url list crawled done!!!   ',url_list)      

        for content_url in url_list:
            try:
                sent_list, query = content_crawl(content_url)
                query = pd.DataFrame(query, columns=['query'])
                result = pd.DataFrame(sent_list,columns=['text'])

                result.dropna(how='all',inplace=True)
                result["length"] = result.apply(lambda x : custom(x["text"]) , axis = 1 )
                result = result.drop(result[result.length < 50].index)

                path = basic_path +  f"{content_url.split('facts.net')[1].replace('/','')}.csv"
                query_path = basic_path.replace('results_output','query_output') +  f"query_{content_url.split('facts.net')[1].replace('/','')}.csv"

                if result.shape[0] >= 50:
                    result.to_csv(path)
                    query.to_csv(query_path)

            except:
                print('='*50)
                print('\n', content_url, 'cannot crawl', '\n')
                print('='*50)
                logger.debug(f"cannot crawl dataset: {content_url}")
                logger.debug("="*100)
                continue
