import numpy as np
import pandas as pd

class DateTimeTool(object):

    @staticmethod
    def getNpDateTime(pdDate):
        return np.datetime64(pdDate) #rowSeries['Date']

    @staticmethod
    def getPdDatetimeIndexByRange(start, end, flagBusinessDayOnly = False):
        """
        getRangeOfDates(start='1/1/2018', end='1/08/2018')
        :return: date list
        """
        if flagBusinessDayOnly:
            return pd.bdate_range(start, end)
        else:
            return pd.date_range(start, end)

    @classmethod
    def getTimestampListByRange(cls, start, end, flagBusinessDayOnly=False):
        return cls.getPdDatetimeIndexByRange(start, end, flagBusinessDayOnly).tolist()

    @classmethod
    def getDatetimeListByRange(cls, start, end, flagBusinessDayOnly=False):
        return cls.getPdDatetimeIndexByRange(start, end, flagBusinessDayOnly).to_pydatetime().tolist()