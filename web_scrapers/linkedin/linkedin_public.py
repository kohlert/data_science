from web_scrapers import SourceClient
from selenium import webdriver  # https://www.selenium.dev/selenium/docs/api/py/api.html
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from datetime import datetime
import pandas as pd
import time
from .local_drivers import chrome_88_path


class LinkedinPublic(SourceClient):
    login_required = False
    concurrency = 3
    sec_per_page = 1.0
    job_detail_cols = ['uid', 'seniority', 'employment_type', 'job_function', 'industries']
    job_summary_cols = ['uid', 'job_title', 'company', 'location', 'posting_date', 'search_date']

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
            # maximize to help linkedin maintain the expected architecture
            self.driver.maximize_window()
        except exceptions.SessionNotCreatedException:
            self.driver.quit()

    def get_detailed_data(self, uid):
        self.view_job_posting(uid)
        data = [el.text for el in
                self.driver.find_elements(By.XPATH, "//li[starts-with(@class, 'job-criteria')]")]
        seniority = data[0][(data[0].find('\n') + 1):]
        emp_type = data[1][(data[1].find('\n') + 1):]
        func = data[2][(data[2].find('\n') + 1):]
        industry = data[3][(data[3].find('\n') + 1):]
        df = pd.DataFrame([[uid, seniority, emp_type, func, industry]], columns=self.job_detail_cols)
        return df

    def quit_client(self):
        self.driver.quit()

    def view_job_posting(self, uid):
        path = f'{self.path}/jobs/view/{uid}'
        self.driver.get(path)

    def job_search(self, keyword, location, start=0):
        path = f"{self.path}jobs/search/?keywords={keyword.replace(' ', '%20')}&location=" \
               f"{location.replace(' ', '%20').replace(',', '%2C')}&start={str(start)}"
        self.driver.get(path)

    def search_summary(self):
        header = self.driver.find_element(By.CLASS_NAME, 'results-context-header__context')
        summary_count = header.find_element(By.CLASS_NAME, 'results-context-header__job-count').text
        summary_text = header.find_element(By.CLASS_NAME, 'results-context-header__query-search').text
        # new = header.find_element(By.CLASS_NAME, 'results-context-header__new-jobs').text
        number = self.parse_int(summary_count)
        return number, summary_text

    def get_summary_data(self, keyword, location):
        self.job_search(keyword, location)
        df = pd.DataFrame(columns=self.job_summary_cols)
        job_list = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]")
        extra = 0
        cnt = len(job_list)
        while len(self.driver.find_elements(By.XPATH,
                                            "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]/div/div")) \
                != len(self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]")):
            for i in range(0, cnt - extra, 3):
                job_list[i].location_once_scrolled_into_view  # scrolls through items to ensure all data is loaded
            cnt = len(self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]"))
        for i in range(0, cnt - extra):
            job = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search')]/li[@data-id]")[i]
            uid = job.get_attribute('data-id')
            title = self.driver.find_element(By.XPATH, f"//li[starts-with(@data-id, '{uid}')]/div/h3").text
            company = self.driver.find_element(By.XPATH, f"//li[starts-with(@data-id, '{uid}')]/div/h4/a").text
            loc = self.driver.find_element(By.XPATH, f"//li[starts-with(@data-id, '{uid}')]/div/div/span").text
            try:
                date = self.driver.find_element(By.XPATH,
                                                f"//li[starts-with(@data-id, '{uid}')]/div/div/time").get_attribute(
                    'datetime')
            except exceptions.NoSuchElementException:
                date = None
            data = [uid, title, company, loc, date, str(datetime.today().date())]
            df = df.append(pd.DataFrame([data], columns=self.job_summary_cols))
        return df

    @staticmethod
    def parse_int(text: str):
        number = ''
        for i in range(len(text)):
            number += (text[i] if text[i].isnumeric() else '')
        return int(number)
