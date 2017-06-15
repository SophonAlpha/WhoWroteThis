# -----------------------------------------------------------------------------
# Created on 1 Jun 2017
# 
# @author: H155936
# -----------------------------------------------------------------------------

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import queue
from threading import Thread
from threading import Event
import time
import re
import os
import json
from pathlib import Path

# AUTHORS_FILE = 'list_of_authors_short.txt'
AUTHORS_FILE = 'list_of_authors.txt'
ARTICLES_URLS_FILE = 'article_urls.json'
ARTICLES_FILE = 'articles.json'

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
        with open(AUTHORS_FILE, 'r') as f:
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
        open(ARTICLES_FILE, 'w').close()
        print('%s collator initialised' %self.getName())
    
    def setup(self):
        # any setup tasks
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
                with open(ARTICLES_FILE, 'a') as f:
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
        # any setup tasks
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
                articles = self.LinkedIn.get_articles(author_url)
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

    def save_article_urls(self):
        current_article_urls = self.load_current_article_urls()
        author_urls = self.load_authors()
        author_urls = self.remove_authors_already_covered(author_urls, current_article_urls)
        self.write_article_urls_to_file(author_urls, current_article_urls)

    def load_current_article_urls(self):
        if Path(ARTICLES_URLS_FILE).exists():
            with open(ARTICLES_URLS_FILE, 'r') as f:
                article_addresses = json.load(f)
        else:
            article_addresses = []
        return article_addresses

    def load_authors(self):
        with open(AUTHORS_FILE, 'r') as f:
            authors = f.readlines()
        return authors

    def remove_authors_already_covered(self, author_urls, current_article_urls):
        reduced_list = []
        author_urls_already_covered = list(set([article_url['author_url'] for article_url in current_article_urls]))
        for author_url in author_urls:
            if author_url in author_urls_already_covered:
                pass
            else:
                reduced_list.append(author_url)
        return reduced_list

    def write_article_urls_to_file(self, author_urls, current_article_urls):
        for author_url in author_urls:
            article_urls = self.get_author_articles_urls(author_url)
            for article_url in article_urls:
                current_article_urls.append({'author_url': author_url,
                                          'article_url': article_url})
            with open(ARTICLES_URLS_FILE, 'w') as f:
                json.dump(current_article_urls, f)

    def get_articles(self, author_url):
        ''' 
        Download all articles for a particular author. The author's LinkedIn
        url needs to be given as argument.
        Return list of articles.
        '''
        articles = []
        article_links = self.get_author_articles_urls(author_url)
        for link in article_links:
            articles.append(self.get_article(link))
        return articles
    
    def get_author_articles_urls(self, author_url):
        '''
        Navigate to the authors list of articles.
        Return list of all URLs to articles for the author.
        '''
        articles_page = self.get_author_articles_page(author_url)
        try:
            self.browser.get(articles_page['author articles page'])
        except WebDriverException as err:
            if err.msg.find('Reached error page:') >= 0:
                # Occasionally elements of the LinkedIn page are not loaded and articles_page
                # "server side reset" exception is thrown by the WebDriver.
                # This exception handler catches and ignores this.
                print('encountered WebDriverException trying to continue. {0}'.format(err.msg))
            else:
                raise err
        article_list = self.build_complete_articles_list_page(articles_page['no of articles'])
        self.article_links = self.extract_articles_urls(article_list)
        return self.article_links
    
    def get_author_articles_page(self, author_url):
        '''
        Navigate to the author's LinkedIn page.
        Return number of articles and the link to the author's article page.
        '''
        try:
            self.browser.get(author_url)
        except WebDriverException as err:
            if err.msg.find('Reached error page:') >= 0:
                # Occasionally elements of the LinkedIn page are not loaded and a
                # "server side reset" exception is thrown by the WebDriver.
                # This exception handler catches and ignores this.
                print('encountered WebDriverException trying to continue. {0}'.format(err.msg))
            else:
                raise err
        # Scroll down only a bit. If we go straight to the bottom of the page 
        # we won't reveal the 'Articles & Activity' section.
        self.browser.execute_script('window.scrollTo(0, document.body.scrollHeight/6);')
        no_of_articles_element = self.wait_for_element('//a[@data-control-name="recent_activity_posts_all"]')
        no_of_articles_text = no_of_articles_element.text
        m = re.search(r'(?P<number>[0-9]+)', no_of_articles_text)
        no_of_articles = int(m.group('number'))
        author_articles_page = no_of_articles_element.get_attribute('href')
        return {'no of articles': no_of_articles, 'author articles page': author_articles_page}

    def build_complete_articles_list_page(self, no_of_articles):
        '''
        Build complete list of articles by scrolling down until list is complete.
        '''
        while True:
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            article_list = self.browser.find_elements_by_tag_name('article')
            no_of_articles_on_page = len(article_list)
            time.sleep(self.scroll_down_wait)
            if no_of_articles_on_page >= no_of_articles:
                break
        return article_list
    
    def extract_articles_urls(self, article_url_list):
        '''
        Extract the articles URLs.
        '''
        self.article_links = []
        for article_url in article_url_list:
            self.article_links.append(article_url.find_element_by_tag_name('a').get_attribute('href'))
        return self.article_links
    
    def get_article(self, article_url):
        '''
        Download one article.
        '''
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
    startTime = time.time()
    print('Start')
    if False:
        # Run this part to build the list of article URLs.
        LinkedIn = LinkedInArticleCollector()
        LinkedIn.start_webdriver()
        LinkedIn.auto_login()        
        LinkedIn.save_article_urls()
        LinkedIn.close()
    if False:
        # Run this part when downloading the articles. This requires the list
        # of article URLs to be build beforehand.
        d = Dispatcher(num_of_workers=1)
        d.run()
    elapsedTime = time.time() - startTime
    print('Done! Runtime: {0} seconds'.format(elapsedTime))
