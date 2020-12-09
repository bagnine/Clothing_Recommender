import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
import urllib
import numpy as np

def make_dataframe(results):
    ''''''

    Name = []
    Designer = []
    Price = []
    NewPrice = []
    OldPrice = []
    Size = []
    Time = []
    LastBump = []
    Link = []

    for i, result in enumerate(results):

        Designer.append(result.find_element_by_class_name("listing-designer").text)
        Name.append(result.find_element_by_class_name("listing-title").text)
        
        try:
            Price.append(result.find_element_by_xpath('.//p[@class="sub-title original-price"]').text)
            NewPrice.append(np.nan)
            OldPrice.append(np.nan)
        except NoSuchElementException:
            NewPrice.append(result.find_element_by_xpath('.//p[@class="sub-title new-price"]').text)
            OldPrice.append(result.find_element_by_xpath('.//p[@class="sub-title original-price strike-through"]').text)
            Price.append(np.nan)

        Size.append(result.find_element_by_xpath('.//p[@class="listing-size sub-title"]').text)

        Time.append(result.find_element_by_xpath(".//span[@class='date-ago']").text)

        try:
            LastBump.append(result.find_element_by_xpath(".//span[@class='strike-through']").text)
        except NoSuchElementException:
            LastBump.append(np.nan)

        Link.append(result.find_element_by_xpath('./a').get_attribute("href"))

        grailed_dict = {'Name': Name, 
                    'Designer': Designer, 
                    'Price': Price, 
                    'NewPrice': NewPrice, 
                    'OldPrice': OldPrice, 
                    'Size': Size, 
                    'Time': Time, 
                    'LastBump': LastBump, 
                    'Link': Link}
    return pd.DataFrame(grailed_dict)