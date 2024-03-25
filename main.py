import gspread
from google.oauth2.service_account import Credentials
import json
import os
import re
import pandas as pd

# trafilatura for web scraping
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract

# urllib to get rid of url error
import urllib.request
import urllib.error

# for summarizing the merged articles
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import heapq

# to set time limit for a function
import multiprocessing
import requests

# to integrate with gsheet
import gspread
from google.oauth2.service_account import Credentials
from oauth2client.service_account import ServiceAccountCredentials

# 1. to check if the URL is accessible
def fetch_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        return response.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        # print(f"HTTPError: {e.code} - {e.reason}")
        return "ERROR"
    except Exception as e:
        # print(f"Error occurred: {str(e)}")
        return "ERROR"


# 2. to get all the URLs from the given website
def get_urls_from_sitemap(url):
    downloaded = fetch_url(url)
    if downloaded == "ERROR":
        return "ERROR"
    try:
        url_list = sitemap_search(url)
        return url_list
    except Exception as e:
        # print(f"Error occurred during extraction: {str(e)}")
        return "ERROR"


# 3. authenticate function to authenticate the api credentials
def authenticate():
    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    # Load credentials from file
    creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
    # Authorize the client
    client = gspread.authorize(creds)
    return client


# 4. get_keywords function to fetch keywords from the spreadsheet
def get_keywords():
    # Authenticate
    gc = authenticate()

    # Open the workbook by name
    workbook = gc.open('clients_info')
    try:
        # Get the desired worksheet by name
        sheet = workbook.worksheet('keywords')
        Include_keywords = []
        Exclude_keywords = []
        all_keywords = sheet.get_all_values()

        for row in all_keywords:
            if row[0]!="":
                Include_keywords.append(row[0])
            if row[1]!="":
                Exclude_keywords.append(row[1])

        Include_keywords = Include_keywords[1:]
        Exclude_keywords = Exclude_keywords[1:]

        if len(Include_keywords)==0 and len(Exclude_keywords)==0:
            Include_keywords = ['about', 'resources', 'innovate', 'case', 'study', 'services', 'industries', 'updates', 'vision', 'mission',
                      "what's new", 'company-overview', 'future-and-strategy', 'service', 'network', 'management', 'case-study',
                      'solutions', 'features', 'skills', 'business', 'brand', 'digital', 'products']

            Exclude_keywords = ['blog', 'news', 'post', 'event', 'thinking', 'press', 'releases', 'story', 'articles', 'fairs-events',
        'social-impact', 'guide', 'content-library', 'glossary', 'disclaimer', 'events', 'press', 'initiatives', 'media']
            return [Include_keywords,Exclude_keywords]

        elif len(Include_keywords)==0:
            Include_keywords = ['about', 'resources', 'innovate', 'case', 'study', 'services', 'industries', 'updates', 'vision', 'mission',
                      "what's new", 'company-overview', 'future-and-strategy', 'service', 'network', 'management', 'case-study',
                      'solutions', 'features', 'skills', 'business', 'brand', 'digital', 'products']
            return [Include_keywords,Exclude_keywords]

        else:
            return [Include_keywords,Exclude_keywords]

    except gspread.exceptions.APIError as e:
        Include_keywords = []
        Exclude_keywords = []
        return [Include_keywords,Exclude_keywords]


# 5. to filter out the relevant URLs from the list of all the URLs
def relevant_urls(url_list):
    if url_list=="ERROR":
        return "ERROR"
    else:
        keywords = get_keywords()
        # Convert the lists to sets for faster lookup
        include_set = set(keywords[0])
        exclude_set = set(keywords[1])
        # Convert the list of URLs to a pandas Series
        url_series = pd.Series(url_list)
        if len(url_series)==0:
            return "ERROR"
        # Check if any of the included keywords is present and none of the excluded keywords are present in the URL
        contains_included = url_series.str.contains('|'.join(include_set))
        contains_excluded = ~url_series.str.contains('|'.join(exclude_set))
        # Filter URLs that meet the conditions
        filtered_urls = url_series[contains_included & contains_excluded]
        return filtered_urls


# 6. to extract the article text from the relevant URLs
def extract_article(url):
    try:
        downloaded = fetch_url(url)
        if downloaded is None:
            return ""
        article = extract(downloaded, favor_precision=True)
        if article is None:
            return ""
        return article
    except Exception as e:
    # print(f"Error occurred: {str(e)}")
        return ""  # Return an empty string for any other error


# 7. to merge all the articles
def merge_text(updates_url_list):
    if isinstance(updates_url_list, pd.Series):
        articles = map(extract_article, updates_url_list)
        merged_text = ''.join(articles)
        return merged_text
    return "Empty list"


# 8. to clean the data if some sentences gets repeated, then the below code will get rid of the repeated sentences.
def data_cleaning(para):
    # Remove extra whitespace characters and newlines
    if para=="Empty list":
        return "Invalid text"
    else:

        cleaned_para = re.sub(r'\s+', ' ', para)

        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_para.strip())

        # Join the sentences back into a paragraph
        cleaned_para = ''.join(sentences)

        cleaned_text = ""
        for i in cleaned_para.split('.'):
            if i not in cleaned_text:
                cleaned_text+=i
                cleaned_text+='. '

        return cleaned_text

# 9. main function start that uses the above created functions to get the cleaned merged articles
def final_text(website):  # websites landing page url

    all_urls = get_urls_from_sitemap(website)       # list containing all the URLs

    if type(all_urls) is list:
        important_urls = relevant_urls(all_urls)        # list containing all the relevant URLs

        final_merged_text = merge_text(important_urls)  # final merged text

        cleaned_text = data_cleaning(final_merged_text) # final cleaned text

        return cleaned_text
    else:
        cleaned_text = 'ERROR'
        return cleaned_text

# 10. function for setting time limit for the landing page function
def landing_page_worker(website, queue):
    result = data_cleaning(extract_article(website))
    queue.put(result)

def landing_page_with_timeout(website, timeout=180):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=landing_page_worker, args=(website, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return "Time limit exceeded"
    else:
        if not queue.empty():
            return queue.get()
        else:
            return "No result"

# 11. function for setting the time limit for final_text function
def final_text_worker(website, queue):
    result = final_text(website)
    queue.put(result)

def final_text_with_timeout(website, timeout=180):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=final_text_worker, args=(website, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return "Time limit exceeded"
    else:
        if not queue.empty():
            return queue.get()
        else:
            return "No result"

# 12. for summarizing the merged article
nltk.download('punkt')
nltk.download('stopwords')
def calculate_sentence_scores(sentences, word_frequencies):
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]
    return sentence_scores

def generate_summary(text, percentage):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]

    # Calculate word frequencies
    word_frequencies = nltk.FreqDist(words)

    # Calculate sentence scores based on word frequencies
    sentence_scores = calculate_sentence_scores(sentences, word_frequencies)

    # Get number of sentences for the summary
    summary_length = int(len(sentences) * percentage)

    # Get top sentences based on scores
    summary_sentences = heapq.nlargest(summary_length, sentence_scores, key=sentence_scores.get)

    # Join the sentences to form the summary
    summary = ' '.join(summary_sentences)
    return summary

# 13. recursive function to summarize the article in less than 2000 characters
def recursive_summary(article, percentage, max_length=2000):
    # If the article length is already less than the max_length
    if len(article) < max_length:
        return article

    # Generate the summary
    summary = generate_summary(article, percentage)

    # If the length of the summary is less than 2000 characters, return the summary
    if len(summary) < max_length:
        return summary

    # Otherwise, recursively call the function with a smaller percentage
    new_percentage = percentage * 0.6  # You can adjust the reduction rate as needed
    return recursive_summary(article, new_percentage, max_length)


credentials_file = 'service_gsheet_cred.json'
# 3. authenticate function to authenticate the api credentials
def authenticate():
    # Define the scope
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    # Load credentials from file
    creds = Credentials.from_service_account_file(credentials_file, scopes=scope)
    # Authorize the client
    client = gspread.authorize(creds)
    return client


def main():
    gc = authenticate()
    # Open the workbook by name
    workbook = gc.open('clients_info')

    

    try:
        # Get the desired worksheet
        url_sheet = workbook.worksheet('test1')    # pass the sheet name containing URL's

        # prompt_sheet = workbook.worksheet('prompt')   # pass the sheet name containing the prompt

        urls = url_sheet.get_all_values()

        i = 1
        while (i < len(urls)) and (urls[i][0]!=''):
            print('a')
            url = urls[i][0].strip()
            print(url)
            article = final_text_with_timeout(url)
            print('article done')

            length_of_article = len(article)

            if length_of_article<100:
                print('entered landing page')
                landing_page_article = landing_page_with_timeout(url)
                print('landing page fatched')

                # if error encountered on the landing page of the url
                if len(landing_page_article)<100:
                    summarized_article = 'Access Denied'
                else:
                    summarized_article = recursive_summary(landing_page_article, 0.9, max_length=2000)

            else:
                summarized_article = recursive_summary(article, 0.9, max_length=2000)

            # print("Done")

            # # if the output of the above code is a summarized article then it will be sent to the openAI api using the below line of code
            # if len(summarized_article) > 100:
            #     prompt = prompt_sheet.get_all_values()[1][0]

            #     prompt = prompt + '\n' + summarized_article

            #     try:
            #         chat_completion = client.chat.completions.create(
            #                         messages = [{"role":"user", "content":prompt}],
            #                         model = "gpt-3.5-turbo")

            #         # updating the mail column using the generated mail
            #         url_sheet.update_cell(i+1,2,chat_completion.choices[0].message.content)

            #     except Exception as e:
            #         url_sheet.update_cell(i+1,2,"Tokens exhausted. Buy some.")
            #         return None

            # else:
            #     url_sheet.update_cell(i+1,2,"Can't generate mail")
            url_sheet.update_cell(i+1,2,summarized_article)
            i+=1

        # if the URLs are not in proper order then the below else statement will work.
        # else:
        #     if i==1:
        #         url_sheet.update_cell(i+1,2,"Your G-Sheet format is not readable: Keep the urls in the column 'A' and you will get the mail generated in the column 'B'")



    except gspread.exceptions.APIError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()






