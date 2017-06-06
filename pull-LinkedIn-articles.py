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

class LinkedInArticleCollector:
    
    def __init__(self, browserType='Firefox'):
        self.wait_timeout = 60
        self.scroll_down_wait = 1
        self.start_webdriver(browserType)
        self.auto_login()

    def start_webdriver(self, browserType):
        if browserType == 'Firefox':
            self.browser = webdriver.Firefox(
                 executable_path='C:\Program Files\Geckodriver\geckodriver.exe')
        if browserType == 'Chrome':
            self.browser = webdriver.Chrome(
                executable_path='C:\Program Files (x86)\Google\Chromedriver\chromedriver.exe')
        self.browser.get('http://www.linkedin.com/')

    def auto_login(self):
        user_id_field = self.browser.find_element_by_xpath('//input[@id="login-email"]')
        pwd_field = self.browser.find_element_by_xpath('//input[@id="login-password"]')
        sign_in_button = self.browser.find_element_by_xpath('//input[@id="login-submit"]')
        config_file = os.getenv('WHOWROTETHIS_CFG')
        configuration = json.load(open(config_file))
        user_id_field.send_keys(configuration['userid'])
        pwd_field.send_keys(configuration['passwd'])
        sign_in_button.click()
        self.wait_for_element('//a[@data-control-name="identity_profile_photo"]')

    def wait_for_element(self, element_xpath):
        element = WebDriverWait(self.browser, wait_timeout).until(
            EC.presence_of_element_located((By.XPATH, element_xpath)))
        return element

    def get_author_articles_url(self, author_url):
        self.browser.get(author_url)
        # Scroll down only a bit. If we go straight to the bottom of the page 
        # we won't reveal the 'Articles & Activity' section.
        self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight/6);')
        no_of_articles_element = self.wait_for_element('//a[@data-control-name="recent_activity_posts_all"]') 
        no_of_articles_text = no_of_articles_element.text
        m = re.search(r'(?P<number>[0-9]+)', no_of_articles_text)
        # The 'number' is in addition to the one article on the users profile page. 
        # Therefore + 1.
        no_of_articles_on_user_profile_page = 1 
        no_of_articles = int(m.group('number')) + no_of_articles_on_user_profile_page
        author_articles_url = no_of_articles_element.get_attribute('href')
        return {'no of articles': no_of_articles, 'author articles url': author_articles_url}

    def get_article_urls(self, author_articles_url, no_of_articles):
        self.browser.get(author_articles_url)
        article_list = self.build_complete_article_page(no_of_articles)
        article_links = self.extract_article_urls(article_list)
        return article_links
    
    def build_complete_article_page(self, no_of_articles):
        while True:
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            article_list = self.browser.find_elements_by_tag_name('article')
            no_of_articles_on_page = len(article_list)
            time.sleep(scroll_down_wait)
            if no_of_articles_on_page == no_of_articles:
                break
        return article_list
    
    def extract_article_urls(self, article_list):
        article_links = []
        for article in article_list:
            article_links.append(article.find_element_by_tag_name('a').get_attribute('href'))
        return article_links
    
    def close(self):
        self.browser.quit()

if __name__ == "__main__":
    LinkedIn = LinkedInArticleCollector()
    a = LinkedIn.get_author_articles_url('https://www.linkedin.com/in/travisbradberry/')
    article_links = LinkedIn.get_article_urls(a['author articles url'], a['no of articles'])
    for link in article_links:
        print(link)
    print('%i articles found\n' % a['no of articles'])
    LinkedIn.close()
