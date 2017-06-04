'''
Created on 1 Jun 2017

@author: H155936
'''

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from click._compat import raw_input
import time
import re

browser = webdriver.Chrome('C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
browser.get('http://www.linkedin.com/')
i = raw_input('\n Please login to LinkedIn and press enter to continue.')

browser.get('https://www.linkedin.com/in/travisbradberry/')
browser.execute_script("window.scrollTo(0, document.body.scrollHeight/6);") # scroll down only a bit,
                                                                            # if we go straight to the bottom of the page we won't reveal the 'Articles & Activity" section
time.sleep(2)

# wait_timeout = 10
# try:
#     WebDriverWait(browser, wait_timeout).until(EC.presence_of_element_located(browser.find_elements_by_xpath('//a[@data-control-name="recent_activity_posts_all"]')))
# except TimeoutException:
#     print("Loading took too much time!")

no_of_articles_element = browser.find_element_by_xpath('//a[@data-control-name="recent_activity_posts_all"]')
no_of_articles_text = no_of_articles_element.text
m = re.search(r'(?P<number>[0-9]+)', no_of_articles_text)
no_of_articles = int(m.group('number')) + 1 # the 'number' is in addition to the one article
                                       # on the users profile page, therefore + 1
link_to_articles = no_of_articles_element.get_attribute('href')

browser.get(link_to_articles)

standard_wait = 0.5
# last_height = 0
while True:
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    article_list = browser.find_elements_by_tag_name('article')
    no_of_articles_on_page = len(article_list)
    print(no_of_articles_on_page)
    if no_of_articles_on_page == no_of_articles:
        break
    
#    time.sleep(max_wait)
#    new_height = browser.execute_script("return document.body.scrollHeight")
#    if new_height == last_height:
#        break
#    last_height = new_height
    
for article in article_list:
    article_link = article.find_element_by_tag_name('a').get_attribute('href')
    print(article_link)

print('%i articles found\n' % no_of_articles)
i = raw_input('\n Please press enter to finish script.')
browser.quit()
