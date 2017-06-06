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

# ------ start webdriver ------

browser_to_use = 'Firefox'
wait_timeout = 30

if browser_to_use == 'Firefox':
    print('Starting Geckodriver ...')
    browser = webdriver.Firefox(executable_path='C:\Program Files\Geckodriver\geckodriver.exe')
if browser_to_use == 'Chrome':
    print('Starting Chromedriver ...')
    browser = webdriver.Chrome('C:\Program Files (x86)\Google\Chromedriver\chromedriver.exe')
browser.get('http://www.linkedin.com/')

# ------ auto login -------

user_id_field = browser.find_element_by_xpath('//input[@id="login-email"]')
pwd_field = browser.find_element_by_xpath('//input[@id="login-password"]')
sign_in_button = browser.find_element_by_xpath('//input[@id="login-submit"]')
config_file = os.getenv('WHOWROTETHIS_CFG')
configuration = json.load(open(config_file))
user_id_field.send_keys(configuration['userid'])
pwd_field.send_keys(configuration['passwd'])
sign_in_button.click()
# i = raw_input('\n Please login to LinkedIn and press enter to continue.')
WebDriverWait(browser, wait_timeout).until(
        EC.presence_of_element_located((By.XPATH, '//a[@data-control-name="identity_profile_photo"]')))

# ------ move to authors profile page and retrieve number of articles  -------

browser.get('https://www.linkedin.com/in/travisbradberry/')
# scroll down only a bit, if we go straight to the bottom of the page we won't reveal the 'Articles & Activity" section
browser.execute_script("window.scrollTo(0, document.body.scrollHeight/6);") 
# time.sleep(2)
try:
#    WebDriverWait(browser, wait_timeout).until(
#        EC.presence_of_element_located(browser.find_elements_by_xpath('//a[@data-tracking-control-name="pp_post_see_more"]')))
#    WebDriverWait(browser, wait_timeout).until(EC.presence_of_element_located(browser.find_element_by_xpath('//a[@data-control-name="recent_activity_posts_all"]')))
    no_of_articles_element = WebDriverWait(browser, wait_timeout).until(
        EC.presence_of_element_located((By.XPATH, '//a[@data-control-name="recent_activity_posts_all"]')))
    # list_of_a = browser.find_elements_by_xpath('//a[@data-tracking-control-name="pp_post_see_more"]')
    print('element loaded')
except TimeoutException:
    print("Loading took too much time!")
no_of_articles_text = no_of_articles_element.text
m = re.search(r'(?P<number>[0-9]+)', no_of_articles_text)
# the 'number' is in addition to the one article on the users profile page, therefore + 1
no_of_articles = int(m.group('number')) + 1

link_to_articles = no_of_articles_element.get_attribute('href')

# ------ move to list of articles and collect article url's ------ 

browser.get(link_to_articles)

scroll_down_wait = 1
while True:
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    article_list = browser.find_elements_by_tag_name('article')
    no_of_articles_on_page = len(article_list)
    time.sleep(scroll_down_wait)
    print(no_of_articles_on_page)
    if no_of_articles_on_page == no_of_articles:
        break

for article in article_list:
    article_link = article.find_element_by_tag_name('a').get_attribute('href')
    print(article_link)

# ------ summary ------ 

print('%i articles found\n' % no_of_articles)
i = raw_input('\n Please press enter to finish script.')
browser.quit()
