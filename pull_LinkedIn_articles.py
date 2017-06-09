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
import threading
import time
import re
import os
import json

# ------------------- Parallel Processing --------------------------------------

class Dispatcher:
    
    def __init__(self, Job_Descriptor, num_of_workers=3):
        self.num_of_workers = num_of_workers
        self.job_description = Job_Descriptor()
        self.Qs = {'jobQ': queue.Queue(), 'resultQ':queue.Queue()}
        self.setup_collator(Job_Descriptor)
        self.setup_workers(Job_Descriptor)
        self.start_collator()
        self.start_workers()
        pass
    
    def run(self):
        job = self.job_description.get_job()
        while True:
            try:
                author_url = job.__next__()
            except StopIteration:
#                self.stop_workers()
#                self.stop_collator()
                break
            else:
                self.Qs['jobQ'].put(author_url)

    def setup_collator(self, Job_Descriptor):
        self.collator = Collator(self.Qs, Job_Descriptor)

    def start_collator(self):
        self.collator.get_thread().start()
        
    def stop_collator(self):
        self.collator.get_thread().stop()
    
    def setup_workers(self, Job_Descriptor):
        self.worker_pool = []
        for _ in range(self.num_of_workers):
            worker = Worker(self.Qs, Job_Descriptor)
            self.worker_pool.append(worker)

    def start_workers(self):
        for worker in self.worker_pool:
            worker.get_thread().start()

    def stop_workers(self):
        for worker in self.worker_pool:
            worker.stop_worker()
            worker.get_thread().join()

class Collator:
    
    def __init__(self, Qs, Job_Descriptor):
        self.jd = Job_Descriptor()
        self.jd.setup_collator()
        self.Qs = Qs
        self.thread = threading.Thread(target=self.run)
        self.thread_name = self.thread.getName()
        print('%s collator initialised' %self.thread_name)
    
    def run(self):
        while True:
            result = self.Qs['resultQ'].get()
            self.jd.process_result(result)

    def get_thread(self):
        return self.thread    

class Worker:

    def __init__(self, Qs, Job_Descriptor):
        self.jd = Job_Descriptor()
        self.jd.setup_worker()
        self.Qs = Qs
        self.thread = threading.Thread(target=self.run)
        self.thread_name = self.thread.getName()
        print('%s worker initialised' %self.thread_name)

    def run(self):
        while True:
            author_url = self.Qs['jobQ'].get()
            articles = self.jd.process_job(author_url)
            self.Qs['resultQ'].put(articles)
            
    def stop_worker(self):
        self.jd.stop_worker()
            
    def get_thread(self):
        return self.thread

# ------------------- Job Descriptor -------------------------------------------

class JobDescriptor():

    def __init__(self):
        pass

    def setup_collator(self):
        # initialisation tasks that the collator needs to perform
        pass

    def stop_collator(self):
        # tear down tasks that the collator needs to perform
        pass
    
    def setup_worker(self):
        # initialisation tasks that every worker needs to perform
        self.LinkedIn = LinkedInArticleCollector()
        
    def stop_worker(self):
        # tear down tasks that every worker needs to perform
        self.LinkedIn.close()
        
    def get_job(self):
        # function used by Dispatcher to get jobs for workers
#        authors_file = 'list_of_authors_short.txt'
        authors_file = 'list_of_authors.txt'
        with open(authors_file, 'r') as f:
            authors = f.readlines()
        for author in authors:
            yield author 
    
    def process_job(self, author_url):
        # entry function to be called by Worker with the job details
        articles = self.LinkedIn.get_articles(author_url)
        return articles
        
    def process_result(self, articles):
        # entry function to be called by the collator. 
        # Describes what to do with the results from the workers
        print('received %i articles' %len(articles))
        with open('articles.json', 'a') as f:
            json.dump(articles, f)

# ------------------- LinkedIn Article Collector--------------------------------

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
#        # The 'number' is in addition to the one article on the users profile page. 
#        # Therefore + 1.
#        no_of_articles_on_user_profile_page = 1 
#        no_of_articles = int(m.group('number')) + no_of_articles_on_user_profile_page
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

if __name__ == "__main__":
    d = Dispatcher(Job_Descriptor=JobDescriptor, num_of_workers=3)
    d.run()
