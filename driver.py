from selenium import webdriver
import time
from datetime import date
from selenium.webdriver.common.keys import Keys
from scrape_table_all import scrape_table
from return_dates import return_dates
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

#Open the link
browser = webdriver.Edge()
browser.maximize_window()
browser.get("https://www.sharesansar.com/today-share-price")
#Select the type of data to scrape

#Select Commercial Bank
searchBar=browser.find_element(By.ID,'sector')
#searchBar.send_keys('Commercial Banks')
click_select=Select(searchBar)
click_select.select_by_visible_text('Hydropower')
sdate = date(2021, 1,1 )
edate = date(2024,3, 31)
dates = return_dates(sdate,edate)

for day in dates:
    #Enter the date

    date_box = browser.find_element(By.ID,'fromdate')
    date_box.clear()
    date_box.send_keys(day)
    #Click Search
    searchBar=browser.find_element(By.ID,'btn_todayshareprice_submit')
    searchBar.click()
    time.sleep(3) #Needed for this sites
    searchBar.send_keys(Keys.ENTER)
    time.sleep(8) #Wait for data to show up longer wait time ensures data has loaded before scraping begins
    #Scrape the table
    html = browser.page_source
    scrape_table(data=html,date=day)

browser.close()
