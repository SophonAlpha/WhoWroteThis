'''
Created on 1 Jun 2017

@author: H155936
'''

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from click._compat import raw_input
import time
import re
import os
import json

wait_timeout = 60
scroll_down_wait = 1

def start_webdriver(browser_to_use):
    if browser_to_use == 'Firefox':
        browser = webdriver.Firefox(
             executable_path='C:\Program Files\eckodriver\geckodriver.exe')
    if browser_to_use == 'Chrome':
        browser = webdriver.Chrome(
            executable_path='C:\Program Files (x86)\Google\Chromedriver\chromedriver.exe')
    browser.get('http://www.linkedin.com/')
    return(browser)

def auto_login(browser):
    user_id_field = browser.find_element_by_xpath('//input[@id="login-email"]')
    pwd_field = browser.find_element_by_xpath('//input[@id="login-password"]')
    sign_in_button = browser.find_element_by_xpath('//input[@id="login-submit"]')
    config_file = os.getenv('WHOWROTETHIS_CFG')
    configuration = json.load(open(config_file))
    user_id_field.send_keys(configuration['userid'])
    pwd_field.send_keys(configuration['passwd'])
    sign_in_button.click()
    wait_for_element(browser, '//a[@data-control-name="identity_profile_photo"]')

def wait_for_element(browser, element_xpath):
    try:
        element = WebDriverWait(browser, wait_timeout).until(
            EC.presence_of_element_located((By.XPATH, element_xpath)))
    except TimeoutException:
        print("Loading took too much time!")
    return element

def get_author_articles_url(browser, author_url):
    browser.get(author_url)
    # Scroll down only a bit. If we go straight to the bottom of the page 
    # we won't reveal the 'Articles & Activity' section.
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight/6);")
    no_of_articles_element = wait_for_element(browser, '//a[@data-control-name="recent_activity_posts_all"]') 
    no_of_articles_text = no_of_articles_element.text
    m = re.search(r'(?P<number>[0-9]+)', no_of_articles_text)
    # The 'number' is in addition to the one article on the users profile page. 
    # Therefore + 1.
    no_of_articles_on_users_profile_page = 1 
    no_of_articles = int(m.group('number')) + no_of_articles_on_user_profile_page
    author_articles_url = no_of_articles_element.get_attribute('href')
    return {'no of articles': no_of_articles, 'author articles url': author_articles_url}

def get_article_urls(browser, author_articles_url, no_of_articles):
    browser.get(author_articles_url)
    article_list = build_complete_article_page(browser, no_of_articles)
    article_links = extract_article_urls(article_list)
    return article_links

def build_complete_article_page(browser, no_of_articles):
    while True:
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        article_list = browser.find_elements_by_tag_name('article')
        no_of_articles_on_page = len(article_list)
        time.sleep(scroll_down_wait)
        if no_of_articles_on_page == no_of_articles:
            break
    return article_list

def extract_article_urls(article_list):
    article_links = []
    for article in article_list:
        article_links.append(article.find_element_by_tag_name('a').get_attribute('href'))
    return article_links

if __name__ == "__main__":
    browser = start_webdriver('Firefox')
    auto_login(browser)
    a = get_author_articles_url(browser, 'https://www.linkedin.com/in/travisbradberry/')
    article_links = get_article_urls(browser, a['author articles url'], a['no of articles'])
    for link in article_links:
        print(link)
    print('%i articles found\n' % a['no of articles'])
    browser.quit()
