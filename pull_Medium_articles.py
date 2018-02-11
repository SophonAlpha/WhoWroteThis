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
import json
from pathlib import Path

# AUTHORS_FILE = 'list_of_authors_short.txt'
AUTHORS_FILE = 'list_of_top_Medium_authors.txt'
ARTICLE_URLS_FILE = 'Medium_article_urls.json'
ARTICLES_FILE = 'Medium_articles.json'

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
        with open(ARTICLE_URLS_FILE, 'r') as f:
            articles_urls = json.load(f)
        for article_url in articles_urls:
            self.Qs['jobQ'].put(article_url['article_url'])
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
                print('received article "{0}" from {1}'.format(result['headline'], result['author']))
                with open(ARTICLES_FILE, 'a') as f:
                    out = json.dumps(result)
                    f.write(out + '\n')
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
        self.LinkedIn = MediumArticleCollector()
        self.Qs = Qs
        self.existing_articles = self.load_existing_articles(ARTICLES_FILE)
        print('%s worker initialised' %self.getName())

    def load_existing_articles(self, articles_file):
        existing_articles = []
        with open(articles_file, 'r') as f:
            ea = f.readlines()
        for article in ea:
            existing_articles.append(json.loads(article)['url'])
        return existing_articles

    def setup(self):
        # any setup tasks
        pass

    def run(self):
        self.LinkedIn.start_webdriver()
        self.LinkedIn.login()
        while True:
            try:
                article_url = self.Qs['jobQ'].get(block=False)
            except queue.Empty:
                if self.is_stopped():
                    self.shutdown()
                    break
            else:
                if not article_url in self.existing_articles:
                    article = self.LinkedIn.get_article(article_url)
                    self.Qs['resultQ'].put(article)
                self.Qs['jobQ'].task_done()
            
    def stop(self):
        self.stop_signal.set()
        
    def is_stopped(self):
        return self.stop_signal.is_set()
    
    def shutdown(self):
        # any clean up tasks before we terminate the thread
        self.LinkedIn.close()
        pass

# ------------------- Medium Article Collector ---------------------------------

class MediumArticleCollector:
    
    def __init__(self, browserType='Firefox'):
        self.wait_timeout = 60
        self.scroll_down_wait = 1
        self.browserType = browserType
        self.saved_article_URLs = []

    # --------------- methods to collect article URLs --------------------------

    def start_webdriver(self):
        if self.browserType == 'Firefox':
            self.browser = webdriver.Firefox(
                 executable_path='C:\Program Files\Geckodriver\geckodriver.exe')
        if self.browserType == 'Chrome':
            self.browser = webdriver.Chrome(
                executable_path='C:\Program Files (x86)\Google\Chromedriver\chromedriver.exe')
        self.browser.get('http://www.medium.com/')

    def login(self):
        """
        Medium.com login is not automated. The login needs to be performed by 
        the user. Medium offers multiple options for login. The user needs to 
        make sure he logs in the same tab that was opened by this script.
        Not ideal. Maybe I build an automated solution later.
        """
        input('Please login to medium.com and press Enter to continue.')
        self.wait_for_element('//img[@class="avatar-image avatar-image--icon"]')

    def save_article_URLs(self):
        self.saved_article_URLs = self.load_saved_article_URLs()
        author_URLs = self.load_authors()
#        author_URLs = self.remove_authors_already_covered(author_URLs)
        self.write_article_URLs_to_file(author_URLs)

    def load_saved_article_URLs(self):
        if Path(ARTICLE_URLS_FILE).exists():
            with open(ARTICLE_URLS_FILE, 'r') as f:
                article_addresses = json.load(f)
        else:
            article_addresses = []
        return article_addresses

    def load_authors(self):
        with open(AUTHORS_FILE, 'r') as f:
            authors = f.readlines()
        return authors

    def remove_authors_already_covered(self, author_urls):
        reduced_list = []
        author_urls_already_covered = list(set([article_url['author_URL'] 
                                                for article_url in self.saved_article_URLs]))
        for author_url in author_urls:
            if author_url in author_urls_already_covered:
                pass
            else:
                reduced_list.append(author_url)
        return reduced_list

    def write_article_URLs_to_file(self, author_URLs):
        for author_URL in author_URLs:
            article_URLs = self.get_author_article_URLs(author_URL)
            print('author: {}, articles: {}'.format(author_URL.rstrip(), len(article_URLs)))
            for article_URL in article_URLs:
                if not(self.article_URL_in_list(article_URL)):
                    self.saved_article_URLs.append({'author_URL': author_URL,
                                                    'article_URL': article_URL})
            with open(ARTICLE_URLS_FILE, 'w') as f:
                json.dump(self.saved_article_URLs, f)
            with open('saved_Medium_authors.csv', 'w') as f:
                f.write('{};{}'.format(author_URL.rstrip(), len(article_URLs)))

    def article_URL_in_list(self, article_URL):
        found = False
        for a in self.saved_article_URLs:
            if a['article_URL'] == article_URL:
                found = True
                break
        return found

    def get_author_article_URLs(self, author_url):
        '''
        Navigate to the authors list of articles.
        Return list of all URLs to articles for the author.
        '''
        try:
            self.browser.get(author_url)
        except WebDriverException as err:
            if err.msg.find('Reached error page:') >= 0:
                # Occasionally elements of the Medium page are not loaded and articles_page
                # "server side reset" exception is thrown by the WebDriver.
                # This exception handler catches and ignores this.
                print('encountered WebDriverException trying to continue. {0}'.format(err.msg))
            else:
                raise err
        self.build_article_links_page()
        article_URLs = self.extract_article_URLs()
        return article_URLs

    def build_article_links_page(self):
        '''
        Build complete list of articles by scrolling down until list is complete.
        '''
        while True:
            height_before_scroll = self.browser.execute_script('return document.body.scrollHeight')
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            if self.reached_end_of_page(height_before_scroll):
                break
        return
    
    def reached_end_of_page(self, height_before_scroll):
        """
        Do an escalating wait to ensure page has been updated after the scroll
        down command and we really have reached the end of the page. Wait time
        is escalated to avoid unnecessary long wait time with faster page loads.
        """
        max_wait_cycles = 5
        reached_end_of_page = False
        for factor in range(1, max_wait_cycles+1):
            height_after_scroll = self.browser.execute_script('return document.body.scrollHeight')
            if height_after_scroll == height_before_scroll:
                reached_end_of_page = True
                wait_time = self.scroll_down_wait*factor
                time.sleep(wait_time)
                self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            else:
                reached_end_of_page = False
                break
        return reached_end_of_page
    
    def extract_article_URLs(self):
        '''
        Extract the article URLs.
        '''
        article_URLs = self.browser.find_elements_by_xpath("//a[@class='button button--smaller button--chromeless u-baseColor--buttonNormal']")
        article_URLs = [article.get_attribute('href') for article in article_URLs]
        return article_URLs

    # --------------- methods to download the articles -------------------------

    def get_articles(self, author_URL):
        # TODO: this method needs to be modified to read the the article URLs
        # from the file that has been created already.
        ''' 
        Download all articles for a particular author. The author's Medium
        URL needs to be given as argument.
        Return list of articles.
        '''
        articles = []
        article_URLs = self.get_author_article_URLs(author_URL)
        for link in article_URLs:
            articles.append(self.get_article(link))
        return articles
    
    def get_article(self, article_url):
        '''
        Download one article.
        '''
        self.browser.get(article_url)
        article = {'url': article_url,
                   'author': self.browser.find_element_by_xpath('//span[@itemprop="name"]').text,
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
    if True:
        # Run this part to build the list of article URLs.
        Medium = MediumArticleCollector()
        Medium.start_webdriver()
        Medium.login()        
        Medium.save_article_URLs()
        Medium.close()
    if False:
        # Run this part when downloading the articles. This requires the list
        # of article URLs to be build beforehand.
        d = Dispatcher(num_of_workers=3)
        d.run()
    elapsedTime = time.time() - startTime
    print('Done! Runtime: {0} hours.'.format(elapsedTime/3600))
