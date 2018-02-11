'''
Created on 10 Feb 2018

@author: H155936
'''
from selenium import webdriver

browser = webdriver.Firefox(
    executable_path='C:\Program Files\Geckodriver\geckodriver.exe')
browser.get('https://topauthors.xyz/')
input('Please scroll to end of page and press Enter to continue.')
#table = browser.find_element_by_xpath("//tbody[@class='list endless-pagination']")
# author_Medium_URLs = table.find_elements_by_xpath("//a[@target='_blank']")
author_Medium_URLs = browser.find_elements_by_xpath("//tbody[@class='list endless-pagination']/tr/td[2]//a[@target='_blank']")
author_Medium_URLs = [author_Medium_URL.get_attribute('href') for author_Medium_URL in author_Medium_URLs]
with open('list_of_top_Medium_authors.txt', 'w') as f:
    for item in author_Medium_URLs:
        f.write("%s" % item)
    