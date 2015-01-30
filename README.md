# ts_analytics
===============

This module provides methods to fetch data from [OpenTSDB](http://opentsdb.net/) HTTP interface and convert them into [Python's](http://www.python.org/) [Pandas](http://pandas.pydata.org/) [Timeseries](http://pandas.pydata.org/pandas-docs/stable/timeseries.html) object. Basic structure is based mostly on [opentsdbr](https://github.com/holstius/opentsdbr/) library.

Ref. to help for information on specific feature

import ts_analytics as tsa
help(tsa)  
help(tsa.random_walk)
help(tsa.dtw)