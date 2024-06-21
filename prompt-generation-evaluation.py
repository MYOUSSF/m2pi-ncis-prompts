import serpapi.client
import yake
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
import random

import serpapi
from ncisKey import ncis_serp_key 
from bs4 import BeautifulSoup
from selenium import webdriver
from numpy import inf
import pickle
import spacy


def initialize_clusters(file_path : str,
                        lan : str,
                        max_ngram : int,
                        dedupLim : float,
                        num_keywords : int,
                        eps : float,
                        min_samples : int,
                        metric : str) -> pd.DataFrame:
    '''
    Extracts keywords from sample prompts.
    Initializes broad clusters of keywords from sample prompts using TF-IDF embeddings and K-Means clustering.
    Keywords are further clustered through DBSCAN to identify and remove noise. 

    file_path : Path to CSV of sample prompts.
    lan : Language to be used for keyword extraction.
    max_ngram : Maximum length of n-gram to be extracted by keyword extractor.
    dedupLim : Value between 0 and 1 used to control duplication of n-grams in keyword extraction. 
               Higher value allows for more duplication.
    num_keywords : Maximum number of keywords to be extracted from each prompt.
    eps : Epsilon to be used in DBSCAN.
    min_samples : Minimum number of neighbors the be considered a core point in DBSCAN.
    metric : Metric to be used in DBSCAN.

    Returns Pandas data frame containing keywords and clusters assigned through both K-Means and DBSCAN.
    '''

    prompts = pd.read_csv(file_path)
    extractor = yake.KeywordExtractor(lan=lan, n=max_ngram, dedupLim=dedupLim, top=num_keywords, features=None)

    all_keywords = []
    for prompt in prompts['Prompt']:
        keywords = [kw[0] for kw in extractor.extract_keywords(prompt)]
        all_keywords.extend(keywords)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_keywords)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(tfidf_matrix)    

    keyword_clusters_df = pd.DataFrame({'Keyword': all_keywords, 'Cluster': kmeans.labels_})

    model = SentenceTransformer('all-mpnet-base-v2')
    keyword_embeddings = model.encode(keyword_clusters_df['Keyword'].tolist())
    keyword_clusters_df['Embedding'] = list(keyword_embeddings)

    embeddings = keyword_clusters_df['Embedding'].tolist()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(embeddings)
    keyword_clusters_df['DBSCAN_Label'] = dbscan.labels_

    return keyword_clusters_df[keyword_clusters_df['DBSCAN_Label'] != -1]

def generate_prompts(n_prompts : int,
                     keywords : pd.DataFrame,
                     clusters : dict,
                     labels : dict) -> list:
    '''
    Randomly generates user specified number of prompts with the structure "Subject" + "Crime" + "Consequence" + "Specific"

    n_prompts : Number of prompts to be generated.
    keywords : Data frame of clustered keywords to be used in prompt generation.
    clusters : Dictionary mapping term in prompt to its K-Means cluster.
    labels : Dictionary mapping term in prompt to its DBSCAN clusters.

    Returns list of prompts as strings.
    '''

    subjects = keywords[(keywords['Cluster'].isin(clusters['subject'])) &
                        (keywords['DBSCAN_Label'].isin(labels['subject']))
                        ]['Keyword'].tolist()
    
    consequences = keywords[(keywords['Cluster'].isin(clusters['consequences'])) &
                        (keywords['DBSCAN_Label'].isin(labels['consequences']))
                        ]['Keyword'].tolist()
    
    crimes = keywords[(keywords['Cluster'].isin(clusters['crimes'])) &
                        (keywords['DBSCAN_Label'].isin(labels['crimes']))
                        ]['Keyword'].tolist()
    
    specifics = keywords[(keywords['Cluster'].isin(clusters['specifics'])) &
                        (keywords['DBSCAN_Label'].isin(labels['specifics']))
                        ]['Keyword'].tolist()
    
    formatted_prompts = []

    for _ in range(n_prompts):
        subject = random.choice(subjects)
        consequence = random.choice(consequences)
        crime = random.choice(crimes)
        specific = random.choice(specifics)

        prompt = f"{subject} {consequence} {crime} {specific}."
        formatted_prompts.append(prompt)

    return formatted_prompts

def get_url_text(url : str,
                 driver : webdriver.Chrome, 
                 save : str = None, 
                 file : str = None) -> str:
    '''
    Uses the Selenium WebDriver to scrape all text from the webpage associated to the provided url.

    url : URL address for webpage to be scraped.
    save : Optional argument for saving scraped text as a user specified file type.
    file : Optional argument for naming file with scraped text.

    Webpage text is returned as a string.
    '''

    driver.get(url)

    page_soup = BeautifulSoup(driver.page_source, 'html.parser')
    p_list = page_soup.find_all("p")

    text = ''

    for p in p_list:
        text += ' ' + p.get_text()

    if save:
        with open(f"{file}.txt", "w") as text_file:
            text_file.write(text)

    return text

def prompt_to_reports(prompt : str,
                      client : serpapi.client,
                      driver : webdriver.Chrome, 
                      num_results : int = inf, 
                      engine : str = 'google_news', 
                      hl : str = 'en', 
                      gl : str = 'us',
                      save : bool = False, 
                      file : str = 'scraped-results') -> list:
    '''
    Takes in a prompt and returns a list of strings containing the text of the first n search results.

    prompt : The prompt to be searched.
    client : SerpApi client to be used for retrieving results.
    engine : The search engine to use. Defaults to Google News.
    hl : Language to use for search. Defaults to English. For supported languages, see https://serpapi.com/google-languages
    gl : Country to use for search. Defaults to US. For countries supported, see https://serpapi.com/google-countries
    save : Boolean determining whether scraped text should be pickled for later use.
    file : Optional argument for naming pickle file with scraped text.

    Returns scraped text from search results as a list of strings.
    '''

    result_type = {'google' : 'organic_results', 
                   'google_news' : 'news_results'}
    
    results_json = client.search(
        q = prompt, 
        engine = engine,
        hl = hl,
        gl = gl
    )

    # print('Search Sucessful')

    results = results_json[result_type[engine]]
    to_scrape = []

    for i in range(min(num_results, len(results))):
        to_scrape.append(results[i]['link'])

    texts = []

    for url in to_scrape:
        texts.append(get_url_text(url, driver))

    if save:
        with open(f"{file}", "wb") as pickle_file:
            pickle.dump(texts, pickle_file)

    return texts



def get_similarity_score(prompt : str, 
                         article : str) -> float:
    '''
    Scores an article based on its similarity to the provided prompt using a transformer model.

    prompt : Prompt for article to be compared to.
    article : Article for comparison.

    Returns score between 0 and 1 as a float. Higher score indicates higher similarity.
    '''

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embedding_1 = model.encode(prompt, convert_to_tensor=True)
    embedding_2 = model.encode(article, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding_1, embedding_2).numpy()[0]
    return similarity.max()


def get_ner_score(article : str,
                  N: int) -> float:
    '''
    Calculates the proportion of desired entities in the provided article.

    article : Text to find entities within.
    N : Total number of entities of interest.

    Returns score between 0 and 1 as a float. A score of 1 indicates all relevant entities were found within the article.
    '''
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(article)

    vars = []
    num_vars = 0

    for ent in doc.ents:
        if ent.label_ not in vars:
            vars += [ent.label_]
            num_vars += 1

    return num_vars/N

def get_score(prompt : str, 
              article : str,
              N : int) -> float:
  '''
  Calculates weighted average of a similarity and NER score for a given article.

  prompt : Prompt to use for similarity score.
  article : Article to be scored.
  N : Total number of entities of interest for NER score.

  Returns a value between 0 and 1 as a float. Scores closer to 1 indicate higher quality articles.
  '''
  return 0.6*get_similarity_score(prompt, article) + 0.4*get_ner_score(article, N)


def score_prompt(prompt : str,
                 num_results : int,
                 num_ner : int,
                 client : serpapi.client,
                 driver : webdriver.Chrome) -> float:
    '''
    Scores a prompt by retrieving the top n search results and finding the average relevancy score for each results.

    prompt : Prompt to be evaluated.
    num_results : Number of results to be evaluated per prompt.

    Returns prompt score as a float value between 0 and 1 with higher scores indicating higher quality prompts.
    '''
    score = 0
    # print(f'Propmpt: {prompt}')

    articles = prompt_to_reports(prompt, client, driver, num_results=num_results + 5)
    # print('Reports Retrieved')

    scored  = 0
    iters = 0
    while scored < min(num_results, len(articles)):
        article = articles[iters]
        if article != '':
            score += get_score(prompt, article, num_ner)
            scored += 1
            # print(f'Scored: {scored}')
        
        iters += 1
        # print(f'Iteration {iters}')
        if iters > num_results + 5:
            break
    
    return score/num_results


def main(n_prompts : int,
         file_path : str = 'sample-prompts.csv',
         lan : str = 'en',
         max_ngram : int = 1,
         dedupLim : float = 0.9,
         num_keywords : int = 20,
         eps : float = 0.4,
         min_samples : int = 5,
         metric : str = 'cosine',
         n_results : int = 10,
         n_entities : int = 10,
         initialize : bool = True) -> pd.DataFrame:
    '''
    Main function for generating prompts, retrieving search engine results, and evaluating prompt performance.

    n_prompts : Number of prompts to generate.
    file_path : File path for initializing keyword clusters - either sample-prompts.csv or precomuputed clusters file.
    lan : Language for key word extraction. Defaults to English.
    max_ngram : Maximum n-gram length for keyword extraction. Defaults to 1.
    dedeupLim : Deduplication threshold for keyword extraction. Defaults to 0.9
    eps : Epsilon for use in DBSCAN. Defaults to 0.4.
    min_samples : Minimum number of neigbours for a core point in DBSCAN. Defaults to 5.
    metric : Metric for use in DBSCAN. Defaults to 'cosine'.
    n_results : Number of results to scrape for evaluation. Defaults to 10.
    n_entities : Number of entities to identify in NER scoring. Defaults to 10.
    initialize : Indicates whether keyword clusters should be initialized from sample prompts or loaded from file.

    Returns Pandas data frame containing prompts and scores.
    '''
    options = webdriver.ChromeOptions()
    options.add_argument("headless") 
    api_key = ncis_serp_key()
    client = serpapi.Client(api_key=api_key)

    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    driver = webdriver.Chrome(options=options)



    if initialize:
        keyword_clusters = initialize_clusters(file_path, lan, max_ngram, dedupLim, num_keywords, eps, min_samples, metric)
        
    else:
        print("Add in option to load keywords from file.")

    cluster_dict = {'subject' : [1],
                    'crimes' : [0],
                    'consequences' : [3], 
                    'specifics' : [0]}

    labels_dict = {'subject' : [0],
                    'crimes' : [2, 3, 9],
                    'consequences' : [1], 
                    'specifics' : [4, 7]}

    prompt_list = generate_prompts(n_prompts, keyword_clusters, cluster_dict, labels_dict)
    # print(prompt_list)

    prompt_data = []

    for prompt in prompt_list:
        prompt_dict = {'prompt' : prompt,
                       'score' : score_prompt(prompt, n_results, n_entities, client, driver)}
        
        prompt_data.append(prompt_dict)

    pd.DataFrame(prompt_data).to_csv('sample-scored-prompts.csv', index=False)

    # print(pd.DataFrame(prompt_data))

    return pd.DataFrame(prompt_data)
        

    

main(100, 'sample-prompts.csv', 'en', 1, 0.9, 20, 0.4, 5, 'cosine', 10, True)


