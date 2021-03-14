from bs4 import BeautifulSoup
from web_scrapers import SourceClient
from selenium import webdriver  # https://www.selenium.dev/selenium/docs/api/py/api.html
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from functools import wraps
from datetime import datetime
import pandas as pd
import pickle
import time
import os
from .local_drivers import chrome_88_path


class LinkedinPublic(SourceClient):
    login_required = False
    concurrency = 3
    sec_per_page = 1.1
    job_detail_cols = ['uid', 'seniority', 'employment_type', 'job_function', 'industries']
    job_summary_cols = ['uid', 'job_title', 'company', 'location', 'posting_date', 'keyword', 'city', 'state',
                        'search_date']

    def __init__(self):
        super().__init__()
        self.path = 'https://www.linkedin.com/'
        self.driver = None
        # chromedriver.exe is needed.  Current package supports chrome version 88.
        self.chrome_driver_path = chrome_88_path

    def load_client(self, name=None, pwd=None):
        try:
            self.driver = webdriver.Chrome(executable_path=self.chrome_driver_path)
            self.driver.get(self.path)
            self.security_check()
            # maximize to help linkedin maintain the expected architecture
            self.driver.maximize_window()
        except exceptions.SessionNotCreatedException:
            self.driver.quit()

    def get_detailed_data(self, uid, directory='linkedin/data/raw/', reload=True):
        if ((uid + '.txt') not in os.listdir(directory)) or reload:
            self.view_job_posting(uid)
            ul_list, article = None, None
            ul_list, article = self._get_details()
            if ul_list and article:
                salary = self._get_salary()
                sal_low, sal_high = self.parse_salary_range(salary)
                soup = BeautifulSoup(ul_list)
                columns = ['uid'] + [li.h3.text for li in soup.find_all('li')] + ['salary_low', 'salary_high']
                data = [', '.join([el.text for el in soup.contents[i].contents[1:]]) for i in
                        range(len(soup.find_all('li')))]
                df = pd.DataFrame([[uid] + data + [sal_low, sal_high]], columns=columns)
                with open(directory + uid + '.txt', 'w') as file:
                    try:
                        file.write(article)
                    except UnicodeEncodeError:
                        pass
                return df

    def quit_client(self):
        self.driver.quit()

    def view_job_posting(self, uid):
        path = f'{self.path}/jobs/view/{uid}'
        self.driver.get(path)
        self.security_check()

    def job_search(self, keyword, location, start=0, max_sal=True):
        if max_sal:
            path = f"{self.path}jobs/search/?keywords={keyword.replace(' ', '%20')}&location=" \
                   f"{location.replace(' ', '%20').replace(',', '%2C')}&f_SB2=5&start={str(start)}"
        else:
            path = f"{self.path}jobs/search/?keywords={keyword.replace(' ', '%20')}&location=" \
                   f"{location.replace(' ', '%20').replace(',', '%2C')}&start={str(start)}"
        self.driver.get(path)
        self.security_check()

    def oneline_summary(self, keyword, city=None, state=None):
        """
        :return: a dataframe with a one-line summary of search results
        """
        location = city + ', ' + state if (city and state) else (city if city else state)
        self.job_search(keyword, location)
        number, text = self.search_summary()
        return pd.DataFrame([[number, text, city, state, str(datetime.today().date())]],
                            columns=['results', 'summary_text', 'city', 'state', 'search_date'])

    def search_summary(self):
        res = self.no_results()
        if res:
            number, summary_text = 0, res
        else:
            header = self.driver.find_element(By.CLASS_NAME, 'results-context-header__context')
            summary_count = header.find_element(By.CLASS_NAME, 'results-context-header__job-count').text
            summary_text = header.find_element(By.CLASS_NAME, 'results-context-header__query-search').text
            # new = header.find_element(By.CLASS_NAME, 'results-context-header__new-jobs').text
            number = self.parse_int(summary_count)
        return number, summary_text

    def no_results(self):
        try:
            txt = self.driver.find_element(By.XPATH,
                                           "//div[starts-with(@class, 'results__container')]/section/div/h2").text
            return txt[17:]
        except exceptions.NoSuchElementException:
            return False

    def get_summary_data(self, keyword, city=None, state=None):
        location = city + ', ' + state if (city and state) else (city if city else state)
        self.job_search(keyword, location)
        df = pd.DataFrame(columns=self.job_summary_cols)
        res_count, search = self.search_summary()
        job_list = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]")
        for i in range(0, res_count, 10):
            job_list[-1].location_once_scrolled_into_view  # scrolls through items to ensure all data is loaded
            job_list = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]")
            time.sleep(self.sec_per_page)
            try:
                self.driver.find_element(By.XPATH, "//button[starts-with(@class, 'infinite-scroller')]").click()
            except (exceptions.NoSuchElementException, exceptions.ElementNotInteractableException):
                pass
            try:
                test = self.driver.find_element(By.XPATH, "//div/p[starts-with(@class, 'inline-notification')]").text
                if "all jobs" in test:
                    break
            except (exceptions.NoSuchElementException, exceptions.ElementNotInteractableException):
                continue
        cnt = len(job_list)
        for i in range(0, cnt):
            job = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]")[i]
            uid = job.get_attribute('data-id')
            title = self.driver.find_element(By.XPATH, f"//li[starts-with(@data-id, '{uid}')]/div/h3").text
            try:
                company = self.driver.find_element(By.XPATH, f"//li[starts-with(@data-id, '{uid}')]/div/h4/a").text
            except exceptions.NoSuchElementException:
                company = None
            loc = self.driver.find_element(By.XPATH, f"//li[starts-with(@data-id, '{uid}')]/div/div/span").text
            try:
                date = self.driver.find_element(By.XPATH,
                                                f"//li[starts-with(@data-id, '{uid}')]/div/div/time").get_attribute(
                    'datetime')
            except exceptions.NoSuchElementException:
                date = None
            data = [uid, title, company, loc, date, keyword, city, state, str(datetime.today().date())]
            df = df.append(pd.DataFrame([data], columns=self.job_summary_cols))
        return df

    def security_check(self, msg=None):
        if self.driver.current_url[:46] == 'https://www.linkedin.com/checkpoint/challenge/':
            if msg:
                wait = input(msg)
            else:
                wait = input('Please check security checkpoint:')

    def try_try_ask(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except:
                try:
                    time.sleep(2)
                    return func(*args, **kwargs)
                except Exception as ex:
                    # print(*args)
                    # test = None
                    # while test not in ['y','n']:
                    #     test = input('\n' + str(ex) + '\nPlease check current status.  Retry?  y/n ...\n')
                    # if test == 'y':
                    #     return func(*args, **kwargs)
                    # else:
                    return None, None

        return wrapper

    @try_try_ask
    def _get_details(self):
        ul_list = self.driver.find_element(By.XPATH, "//ul[starts-with(@class, 'job-criteria__list')]"). \
            get_attribute('innerHTML')
        article = self.driver.find_element(By.XPATH, "//div[starts-with(@class, 'show-more-less-html')]"). \
            get_attribute('innerHTML')
        return ul_list, article

    @try_try_ask
    def _get_salary(self):
        salary = self.driver.find_element(By.XPATH, "//div[starts-with(@class, 'salary compensation_')]").text
        return salary

    @staticmethod
    def parse_int(text: str):
        number = ''
        for i in range(len(text)):
            number += (text[i] if text[i].isnumeric() else '')
        return int(number)

    @staticmethod
    def parse_salary_range(salary_range: str):
        salary_low, salary_high = None, None
        if (('$' in salary_range) and ('K' in salary_range)):
            salary_low = LinkedinPublic.parse_int(salary_range[:(salary_range.find(' '))]) * 1000
            salary_high = LinkedinPublic.parse_int(salary_range[(salary_range.find(' ')):]) * 1000
        elif (('$' in salary_range) and ('.00/yr' in salary_range) and ('K' not in salary_range)):
            salary_low = LinkedinPublic.parse_int(salary_range[:(salary_range.find('.00/yr'))])
            salary_high = LinkedinPublic.parse_int(salary_range[(salary_range.find('.00/yr') + 6):-6])
        return salary_low, salary_high
