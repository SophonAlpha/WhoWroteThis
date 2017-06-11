# -----------------------------------------------------------------------------
# Created on 1 Jun 2017
# 
# @author: H155936
# -----------------------------------------------------------------------------

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import queue
from threading import Thread
from threading import Event
import time
import re
import os
import json

# ------------------- Parallel Processing --------------------------------------

class Dispatcher:
    
    def __init__(self, num_of_workers=3):
        self.num_of_workers = num_of_workers
        self.Qs = {'jobQ': queue.Queue(), 'resultQ':queue.Queue()}
        self.setup_collator()
        self.setup_workers()
        self.start_collator()
        self.start_workers()
        pass
    
    def run(self):
        authors_file = 'list_of_authors_short.txt'
        with open(authors_file, 'r') as f:
            authors = f.readlines()
        for author_url in authors:
            self.Qs['jobQ'].put(author_url)
        self.wait_till_queues_empty(self.Qs)
        self.stop_workers()
        self.stop_collator()

    def setup_collator(self):
        self.collator = Collator(self.Qs)

    def start_collator(self):
        self.collator.start()
        
    def stop_collator(self):
        self.collator.stop()
        self.collator.join()
    
    def setup_workers(self):
        self.worker_pool = []
        for _ in range(self.num_of_workers):
            worker = Worker(self.Qs)
            self.worker_pool.append(worker)

    def start_workers(self):
        for worker in self.worker_pool:
            worker.start()

    def stop_workers(self):
        for worker in self.worker_pool:
            worker.stop()
            worker.join()

    def wait_till_queues_empty(self, waitQs):
        for waitQ in waitQs:
            waitQs[waitQ].join()

class Collator(Thread):
    
    def __init__(self, Qs):
        Thread.__init__(self)
        self.stop_signal = Event()
        self.Qs = Qs
        print('%s collator initialised' %self.getName())
    
    def setup(self):
        pass
    
    def run(self):
        while True:
            try:
                result = self.Qs['resultQ'].get(block=False)
            except queue.Empty:
                if self.is_stopped():
                    self.shutdown()
                    break
            else:
                print('received {0} articles from {1}'.format(len(result), result[0]['author']))
                with open('articles.json', 'a') as f:
                    json.dump(result, f)
                self.Qs['resultQ'].task_done()
            
    def stop(self):
        self.stop_signal.set()
        
    def is_stopped(self):
        return self.stop_signal.is_set()
    
    def shutdown(self):
        # any clean up tasks before we terminate the thread
        pass

class Worker(Thread):

    def __init__(self, Qs):
        Thread.__init__(self)
        self.stop_signal = Event()
        self.LinkedIn = LinkedInArticleCollector()
        self.Qs = Qs
        print('%s worker initialised' %self.getName())

    def setup(self):
        pass

    def run(self):
        self.LinkedIn.start_webdriver()
        self.LinkedIn.auto_login()
        while True:
            try:
                author_url = self.Qs['jobQ'].get(block=False)
            except queue.Empty:
                if self.is_stopped():
                    self.shutdown()
                    break
            else:
                articles = articles = self.LinkedIn.get_articles(author_url)
                self.Qs['resultQ'].put(articles)
                self.Qs['jobQ'].task_done()
            
    def stop(self):
        self.stop_signal.set()
        
    def is_stopped(self):
        return self.stop_signal.is_set()
    
    def shutdown(self):
        # any clean up tasks before we terminate the thread
        self.LinkedIn.close()
        pass

# ------------------- LinkedIn Article Collector -------------------------------

class LinkedInArticleCollector:
    
    def __init__(self, browserType='Firefox'):
        self.wait_timeout = 60
        self.scroll_down_wait = 1
        self.browserType = browserType

    def start_webdriver(self):
        if self.browserType == 'Firefox':
            self.browser = webdriver.Firefox(
                 executable_path='C:\Program Files\Geckodriver\geckodriver.exe')
        if self.browserType == 'Chrome':
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

    def get_articles(self, author_url):
        articles = []
        article_links = self.get_article_urls(author_url)
        for link in article_links:
            articles.append(self.get_article(link))
        return articles
    
    def get_article_urls(self, author_url):
        a = self.get_author_articles_url(author_url)
        self.browser.get(a['author articles url'])
        article_list = self.build_complete_article_page(a['no of articles'])
        self.article_links = self.extract_article_urls(article_list)
        return self.article_links
    
    def get_author_articles_url(self, author_url):
        self.browser.get(author_url)
        # Scroll down only a bit. If we go straight to the bottom of the page 
        # we won't reveal the 'Articles & Activity' section.
        self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight/6);')
        no_of_articles_element = self.wait_for_element('//a[@data-control-name="recent_activity_posts_all"]')
        no_of_articles_text = no_of_articles_element.text
        m = re.search(r'(?P<number>[0-9]+)', no_of_articles_text)
        no_of_articles = int(m.group('number'))
        author_articles_url = no_of_articles_element.get_attribute('href')
        return {'no of articles': no_of_articles, 'author articles url': author_articles_url}

    def build_complete_article_page(self, no_of_articles):
        while True:
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            article_list = self.browser.find_elements_by_tag_name('article')
            no_of_articles_on_page = len(article_list)
            time.sleep(self.scroll_down_wait)
            if no_of_articles_on_page >= no_of_articles:
                break
        return article_list
    
    def extract_article_urls(self, article_list):
        self.article_links = []
        for article in article_list:
            self.article_links.append(article.find_element_by_tag_name('a').get_attribute('href'))
        return self.article_links
    
    def get_article(self, article_url):
        self.browser.get(article_url)
        article = {'author': self.browser.find_element_by_xpath('//span[@itemprop="name"]').text,
                   'headline': self.browser.find_element_by_xpath('//h1[@itemprop="headline"]').text,
                   'body': self.browser.find_element_by_xpath('//div[@itemprop="articleBody"]').text}
        return article

    def wait_for_element(self, element_xpath):
        element = WebDriverWait(self.browser, self.wait_timeout).until(
            EC.presence_of_element_located((By.XPATH, element_xpath)))
        return element

    def close(self):
        self.browser.quit()

# ------------------- Main -----------------------------------------------------

if __name__ == "__main__":
    d = Dispatcher(num_of_workers=1)
    d.run()
