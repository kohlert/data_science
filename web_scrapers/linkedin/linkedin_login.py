from bs4 import BeautifulSoup
from .linkedin_public import LinkedinPublic
from selenium.common import exceptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from datetime import datetime
import pandas as pd
import pickle
import time


class LinkedinLogin(LinkedinPublic):
    login_required = True
    concurrency = 1
    sec_per_page = 2.0
    job_summary_cols = ['uid', 'job_title', 'company', 'location', 'posting_date', 'applicants', 'salary_low',
                        'salary_high', 'search_date']

    def __init__(self):
        super().__init__()

    def load_client(self, name=None, pwd=None):
        super().load_client()
        self._login(name, pwd)

    def _login(self, name, pwd):
        name_field = self.driver.find_element(By.NAME, 'session_key')
        name_field.send_keys(name)
        password = self.driver.find_element(By.NAME, 'session_password')
        password.send_keys(pwd)
        time.sleep(self.sec_per_page)
        self.driver.find_element(By.CLASS_NAME, 'sign-in-form__submit-button').click()

    def search_summary(self):
        header = self.driver.find_element(By.CLASS_NAME, 'jobs-search-results-list__title-heading')
        summary_text = header.find_element(By.CSS_SELECTOR, 'h1').text
        summary_count = header.find_element(By.CSS_SELECTOR, 'small').text
        number = self.parse_int(summary_count)
        return number, summary_text

    def jobs_summary(self):
        df = pd.DataFrame(columns=self.job_summary_cols)
        cnt = self._populate_search_list()
        for i in range(0, cnt):
            job = self._get_list_populated()[i]
            uid = job.get_attribute('data-job-id')
            posting = [entry.text for entry in job.find_elements(By.CSS_SELECTOR, 'a')]
            add_data = [entry.text for entry in job.find_elements(By.CSS_SELECTOR, 'li')]
            title = posting[1] if len(posting) >= 2 else None
            company = posting[2] if len(posting) >= 3 else None
            loc = add_data[0] if len(add_data) >= 1 else None
            applicants = self.parse_int(add_data[-1]) if ('applicant' in add_data[-1]) else None
            salary_low, salary_high = None, None
            if len(add_data) >= 2:
                if (('$' in add_data[1]) and ('K' in add_data[1])):
                    salary_low = self.parse_int(add_data[1][:(add_data[1].find(' '))])
                    salary_high = self.parse_int(add_data[1][(add_data[1].find(' ')):])
            try:
                date = self.driver.find_element(By.XPATH,
                                                f"//div[starts-with(@data-job-id, '{uid}')]/ul/li/time").get_attribute(
                    'datetime')
            except:
                date = None
            data = [uid, title, company, loc, date, applicants, salary_low, salary_high,
                    str(datetime.today().date())]
            df = df.append(pd.DataFrame([data], columns=self.job_summary_cols))
        return df

    def get_summary_data(self, keyword, location, start=0, append_df=pd.DataFrame(job_summary_cols), save_path=None):
        self.job_search(keyword, location, start=start)
        j, _ = self.search_summary()
        j = min(j, 1000)
        df = append_df.append(self.jobs_summary()).drop_duplicates()
        k = 25
        time.sleep(self.sec_per_page)
        while k < j:
            self.job_search(keyword, location, start=k + start)
            df = df.append(self.jobs_summary())
            df.drop_duplicates(inplace=True)
            if save_path:
                df.to_csv(save_path)
            time.sleep(self.sec_per_page)
            k += 25
        return df

    def get_detailed_data(self, uid, directory='linkedin/data/'):
        self.view_job_posting(uid)
        ul_list = self.driver.find_element(By.XPATH, "//div[starts-with(@class, 'jobs-description-details')]")
        article = self.driver.find_element(By.XPATH, "//div[starts-with(@class, 'jobs-box__html-content')]"). \
            get_attribute('innerHTML')
        soup = BeautifulSoup(ul_list.get_attribute('innerHTML'))
        columns = ['uid'] + [div.h3.text.strip() for div in soup.find_all('div')] + ['salary_low', 'salary_high']
        body = BeautifulSoup(article)
        with open(directory + uid + '.pkl', 'wb+') as file:
            pickle.dump(body, file)
        salary_low, salary_high = None, None
        try:
            salary = self.driver.find_element(By.XPATH, "//div[starts-with(@class, 'mh5 mt4')]/p").text
            if (('$' in salary) and ('yr' in salary)):
                salary_low = self.parse_int(salary[:(salary.find(' '))])
                salary_high = self.parse_int(salary[(salary.find(' ')):])
        except exceptions.NoSuchElementException:
            pass
        data = [', '.join([str for str in h3.parent.stripped_strings][1:]) for h3 in soup.find_all('h3')]
        df = pd.DataFrame([[uid] + data + [salary_low, salary_high]], columns=columns)
        return df

    def _get_list_skeleton(self):
        """
        :return: a list of elements identified as likely placeholders for job data cards (potentially not yet populated)
        """
        job_slots = self.driver.find_elements(By.XPATH, "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]")
        return job_slots

    def _get_list_populated(self):
        """
        :return: a list of elements which are populated with data of the expected structure
        """
        job_data = self.driver.find_elements(By.XPATH,
                                             "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]/div/div")
        return job_data

    def _get_count_correction(self):
        """
        :return: count of elements associated with _get_list_skeleton, but with wrong structure for _get_list_populated
        """
        other_slots = self.driver.find_elements(By.XPATH,
                                                "//ul[starts-with(@class, 'jobs-search-results')]/li[@id]/div/h3")
        return len(other_slots)

    def _populate_search_list(self):
        """
        Scrolls through search list to allow page results to fully populate.
        :return: # of populated job cards in available list
        """
        job_slots = self._get_list_skeleton()
        cnt = len(job_slots) - self._get_count_correction()
        while len(self._get_list_populated()) != cnt:
            for i in range(0, cnt, 3):
                job_slots[i].location_once_scrolled_into_view
            time.sleep(self.sec_per_page)
        return cnt

    def search_filter(self, salary=100, distance=25):
        pass
