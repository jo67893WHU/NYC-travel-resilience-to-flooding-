
The raw data is huge but open access, here we list the links of them for your downloading.

The multiple transportation data used are all open access. Specifically:
1. The taxi data and FHV data used are openly available in NYC Taxi & Limousine Commission at https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
2. The subway data can be linked by NY Open Data at https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-202/wujg-7c2s/about_data
3. The Citibike trip data is at https://citibikenyc.com/system-data

*Note Data cleaning is necessary to all types of dataset. Including:
* keep data with datetime within selected time range for study
* remove trips with nan O/D infomation
* remove trips with abnormal ridering time, passanger count, speed, fare fee, wait time, etc.

*Particularly, subway records may charged a lot due to the reboot of device, which should be taken into consideration when calculating riderships.
