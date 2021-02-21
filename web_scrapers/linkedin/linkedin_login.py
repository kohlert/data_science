from .linkedin_public import LinkedinPublic
from selenium import webdriver  # https://www.selenium.dev/selenium/docs/api/py/api.html
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from datetime import datetime
import pandas as pd
import os
import json
import time
from .local_drivers import chrome_88_path


class LinkedinLogin(LinkedinPublic):
    login_required = True
    concurrency = 1
    sec_per_page = 1.0

    def __init__(self):
        super().__init__()

    def load_client(self, name=None, pwd=None):
        super().load_client()
        self._login(name, pwd)

    def _login(self, name, pwd):
        name_field = self.driver.find_element(By.NAME, 'session_key')
        name_field.send_keys(name)
        time.sleep(2.5)
        password = self.driver.find_element(By.NAME, 'session_password')
        password.send_keys(pwd)
        time.sleep(0.8)
        self.driver.find_element(By.CLASS_NAME, 'sign-in-form__submit-button').click()

    def job_search(self, keyword, location, http=True, start=0):
        if http:
            path = 'https://www.linkedin.com/jobs/search/?keywords=' + keyword.replace(' ', '%20') + \
                   '&location=' + location.replace(' ', '%20').replace(',', '%2C') + \
                   '&start=' + str(start)
            self.driver.get(path)
        else:
            self.driver.get('https://www.linkedin.com/jobs/')
            if self.login:
                kw_input, loc_input = self.find_private_search_bar()
            else:
                kw_input, loc_input = self.find_public_search_bar(initial=(start == 0))
            loc_input.clear()
            kw_input.clear()
            loc_input.send_keys(location)
            kw_input.send_keys(keyword)
            time.sleep(0.5)
            kw_input.send_keys(Keys.ENTER)

    def find_private_search_bar(self):
        inpts = self.driver.find_elements(By.CSS_SELECTOR, 'input')
        kw_input, loc_input = None, None
        for inpt in inpts:
            try:
                if 'jobs-search-box-keyword-id-ember' in inpt.get_attribute('id'):
                    kw_input = inpt
                elif 'jobs-search-box-location-id-ember' in inpt.get_attribute('id'):
                    loc_input = inpt
            except:
                continue
        return kw_input, loc_input

    def find_public_search_bar(self, initial=False):
        forms = self.driver.find_elements(By.CLASS_NAME, 'base-search-bar__form')
        for form in forms:
            try:
                if form.get_attribute('id') == (
                        'JOBS' if initial else 'public_jobs_jobs-search-bar_base-search-bar-form'):
                    break
            except:
                continue
        kw_input = form.find_element(By.NAME, 'keywords')
        loc_input = form.find_element(By.NAME, 'location')
        return kw_input, loc_input

    def search_filter(salary=120):
        # elements = self.driver.find_elements(By.)
        pass

    def search_summary(self):
        if self.login:
            header = self.driver.find_element(By.CLASS_NAME, 'jobs-search-results-list__title-heading')
            summary_text = header.find_element(By.CSS_SELECTOR, 'h1').text
            summary_count = header.find_element(By.CSS_SELECTOR, 'small').text
        else:
            header = self.driver.find_element(By.CLASS_NAME, 'results-context-header__context')
            summary_count = header.find_element(By.CLASS_NAME, 'results-context-header__job-count').text
            summary_text = header.find_element(By.CLASS_NAME, 'results-context-header__query-search').text
            new = header.find_element(By.CLASS_NAME, 'results-context-header__new-jobs').text
        number = self.parse_int(summary_count)
        return number, summary_text

    def jobs_summary(self):
        # left_rail = self.driver.find_element(By.CLASS_NAME, 'jobs-search__left-rail')
        # self.driver.find_element(By.CLASS_NAME, 'jobs-list-feedback').location_once_scrolled_into_view
        cols = ['uid', 'job_title', 'company', 'location', 'posting_date', 'applicants', 'salary_low', 'salary_high',
                'search_date']
        df = pd.DataFrame(columns=cols)
        time.sleep(2.0)
        job_list = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]")
        extra = len(
            self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]/div/h3"))
        cnt = len(job_list)
        while len(self.driver.find_elements(By.XPATH,
                                            "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]/div/div")) \
                != cnt - extra:
            for i in range(0, cnt - extra, 3):
                job_list[i].location_once_scrolled_into_view
            time.sleep(0.5)
        for i in range(0, cnt - extra):
            job = \
                self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]/div/div")[
                    i]
            UID = job.get_attribute('data-job-id')
            posting = [entry.text for entry in job.find_elements(By.CSS_SELECTOR, 'a')]
            add_data = [entry.text for entry in job.find_elements(By.CSS_SELECTOR, 'li')]
            title = posting[1] if len(posting) >= 2 else None
            company = posting[2] if len(posting) >= 3 else None
            loc = add_data[0] if len(add_data) >= 1 else None
            applicants = linkedin.parse_int(add_data[-1]) if ('applicant' in add_data[-1]) else None
            salary_low, salary_high = None, None
            if len(add_data) >= 2:
                if (('$' in add_data[1]) and ('K' in add_data[1])):
                    salary_low = self.parse_int(add_data[1][:(add_data[1].find(' '))])
                    salary_high = self.parse_int(add_data[1][(add_data[1].find(' ')):])
            try:
                date = self.driver.find_element(By.XPATH,
                                                f"//div[starts-with(@data-job-id, '{UID}')]/ul/li/time").get_attribute(
                    'datetime')
            except:
                date = None
            data = [UID, title, company, loc, date, applicants, salary_low, salary_high,
                    str(datetime.today().date())]
            df = df.append(pd.DataFrame([data], columns=cols))
        return df

    def collect_all_summary_data(self, keyword, location, start=0, append_df=None, save_path=None):
        self.job_search(keyword, location, start=start)
        j, _ = self.search_summary()
        j = min(j, 1000)
        if append_df:
            df = append_df.append(self.jobs_summary()).drop_duplicates()
        else:
            df = self.jobs_summary()
        k = 25
        while k < j:
            self.job_search(keyword, location, start=k + start)
            df = df.append(self.jobs_summary())
            df.drop_duplicates(inplace=True)
            if save_path:
                df.to_csv(save_path)
            time.sleep(1.5)
            k += 25
        return df

    @staticmethod
    def parse_int(text: str):
        number = ''
        for i in range(len(text)):
            number += (text[i] if text[i].isnumeric() else '')
        return int(number)
