from abc import ABC, abstractmethod


class SourceClient(ABC):
    login_required = False
    concurrency = 1
    sec_per_page = 0.0  # minimum time to take per page to avoid lockout while iterating

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load_client(self, *args):
        """
        Initializes a session with the data source of interest.
        """
        pass

    @abstractmethod
    def get_summary_data(self, *args):
        """
        Returns a table of summary information which can be iterated over to collect desired details.
        Table must have a unique id column 'uid'
        """
        pass

    @abstractmethod
    def get_detailed_data(self, uid):
        """
        Returns details for one unique id, taken from the 'uid' column of the summary table.
        """
        pass

    @abstractmethod
    def quit_client(self):
        """
        quit selenium client
        """
        pass
