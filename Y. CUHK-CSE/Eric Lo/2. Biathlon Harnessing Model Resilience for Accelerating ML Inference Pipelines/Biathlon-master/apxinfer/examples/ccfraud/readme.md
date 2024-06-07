

------------------------------------------------------------
all files in this dataset
``` text
-rw-rw-r--. 1 ckchang ckchang 2.2G Oct 14  2021 credit_card_transactions-ibm_v2.csv
-rw-rw-r--. 1 ckchang ckchang 476K Oct 14  2021 sd254_cards.csv
-rw-rw-r--. 1 ckchang ckchang 220K Oct 14  2021 sd254_users.csv
-rw-rw-r--. 1 ckchang ckchang 1.9M Oct 14  2021 User0_credit_card_transactions.csv
```

------------------------------------------------------------
``` SQL
select * from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv') limit 1 FORMAT Vertical
```
Row 1:
──────
User:           0
Card:           0
Year:           2002
Month:          9
Day:            1
Time:           06:21
Amount:         $134.09
Use Chip:       Swipe Transaction
Merchant Name:  3527213246127876953
Merchant City:  La Verne
Merchant State: CA
Zip:            91750
MCC:            5300
Errors?:        ᴺᵁᴸᴸ
Is Fraud?:      No

1 row in set. Elapsed: 0.003 sec. 


------------------------------------------------------------
``` SQL
select count(), min(Month), max(Month), min(Day), max(Day), min(Time), max(Time) from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv')
```
┌──count()─┬─min(Month)─┬─max(Month)─┬─min(Day)─┬─max(Day)─┬─min(Time)─┬─max(Time)─┐
│ 24386900 │          1 │         12 │        1 │       31 │ 00:00     │ 23:59     │
└──────────┴────────────┴────────────┴──────────┴──────────┴───────────┴───────────┘

1 row in set. Elapsed: 4.523 sec. Processed 24.39 million rows, 4.45 GB (5.39 million rows/s., 983.93 MB/s.)

------------------------------------------------------------
``` SQL
select uniqExact(`User`), uniqExact(`Card`), uniqExact(`Use Chip`), uniqExact(`Merchant Name`), uniqExact(`Merchant City`), uniqExact(`Merchant State`), uniqExact(`Zip`), uniqExact(MCC), uniqExact(`Errors?`), uniqExact(`Is Fraud?`), countIf(`Is Fraud?`='No'), countIf(`Is Fraud?`='Yes') from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv') FORMAT Vertical
```
Row 1:
──────
uniqExact(User):                   2000
uniqExact(Card):                   9
uniqExact(Use Chip):               3
uniqExact(Merchant Name):          100343
uniqExact(Merchant City):          13429
uniqExact(Merchant State):         223
uniqExact(Zip):                    27321
uniqExact(MCC):                    109
uniqExact(Errors?):                23
uniqExact(Is Fraud?):              2
countIf(equals(Is Fraud?, 'No')):  24357143
countIf(equals(Is Fraud?, 'Yes')): 29757

1 row in set. Elapsed: 4.853 sec. Processed 24.39 million rows, 4.45 GB (5.02 million rows/s., 916.96 MB/s.)

------------------------------------------------------------
``` SQL
select distinct `Errors?` from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv') order by  `Errors?`
```
┌─Errors?─────────────────────────────────────────────┐
│ Bad CVV                                             │
│ Bad CVV,Insufficient Balance                        │
│ Bad CVV,Technical Glitch                            │
│ Bad Card Number                                     │
│ Bad Card Number,Bad CVV                             │
│ Bad Card Number,Bad Expiration                      │
│ Bad Card Number,Bad Expiration,Insufficient Balance │
│ Bad Card Number,Bad Expiration,Technical Glitch     │
│ Bad Card Number,Insufficient Balance                │
│ Bad Card Number,Technical Glitch                    │
│ Bad Expiration                                      │
│ Bad Expiration,Bad CVV                              │
│ Bad Expiration,Insufficient Balance                 │
│ Bad Expiration,Technical Glitch                     │
│ Bad PIN                                             │
│ Bad PIN,Insufficient Balance                        │
│ Bad PIN,Technical Glitch                            │
│ Bad Zipcode                                         │
│ Bad Zipcode,Insufficient Balance                    │
│ Bad Zipcode,Technical Glitch                        │
│ Insufficient Balance                                │
│ Insufficient Balance,Technical Glitch               │
│ Technical Glitch                                    │
│ ᴺᵁᴸᴸ                                                │
└─────────────────────────────────────────────────────┘

24 rows in set. Elapsed: 4.428 sec. Processed 24.39 million rows, 4.45 GB (5.51 million rows/s., 1.01 GB/s.)

------------------------------------------------------------
``` SQL
select distinct `Use Chip` from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv') order by  `Use Chip`
```
┌─Use Chip───────────┐
│ Chip Transaction   │
│ Online Transaction │
│ Swipe Transaction  │
└────────────────────┘

3 rows in set. Elapsed: 4.578 sec. Processed 24.39 million rows, 4.45 GB (5.33 million rows/s., 972.15 MB/s.)

------------------------------------------------------------
``` SQL
select User, countIf(`Is Fraud?` == 'Yes') as fraud_cnt from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv') group by User order by fraud_cnt DESC INTO OUTFILE 'user_fraud_cnt.csv'
```

``` SQL
with parseDateTimeBestEffort(concat(toString(Year), '-', toString(Month), '-', toString(Day), ' ', toString(Time))) as trans_datetime
select User, count() as trans_cnt, countIf(`Is Fraud?` == 'Yes') as fraud_cnt, min(trans_datetime) as first_trans, max(trans_datetime) as last_trans, minIf(trans_datetime, `Is Fraud?` == 'Yes') as first_fraud, maxIf(trans_datetime, `Is Fraud?` == 'Yes') as last_fraud
from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv')
group by User 
order by fraud_cnt DESC 
INTO OUTFILE 'user_fraud_cnt.csv' FORMAT CSVWithNames
```

``` SQL
with parseDateTimeBestEffort(concat(toString(Year), '-', toString(Month), '-', toString(Day), ' ', toString(Time))) as trans_datetime
select User, count() as trans_cnt, countIf(`Is Fraud?` == 'Yes') as fraud_cnt, min(trans_datetime) as first_trans, max(trans_datetime) as last_trans, minIf(trans_datetime, `Is Fraud?` == 'Yes') as first_fraud, maxIf(trans_datetime, `Is Fraud?` == 'Yes') as last_fraud
from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/credit_card_transactions-ibm_v2.csv')
group by User 
order by fraud_cnt DESC 
LIMIT 3
```
┌─User─┬─trans_cnt─┬─fraud_cnt─┬─────────first_trans─┬──────────last_trans─┬─────────first_fraud─┬──────────last_fraud─┐
│ 1064 │     43835 │       113 │ 1997-05-24 20:54:00 │ 2020-02-28 20:23:00 │ 1998-12-07 18:05:00 │ 2019-10-25 18:09:00 │
│ 1487 │     18923 │       110 │ 2001-05-01 09:03:00 │ 2020-02-28 11:34:00 │ 2004-02-16 02:24:00 │ 2019-08-16 11:54:00 │
│ 1425 │     22862 │       100 │ 1996-06-01 14:47:00 │ 2020-02-28 09:19:00 │ 1997-06-17 18:18:00 │ 2019-07-23 14:18:00 │
└──────┴───────────┴───────────┴─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────┘

3 rows in set. Elapsed: 5.344 sec. Processed 24.39 million rows, 4.45 GB (4.56 million rows/s., 832.69 MB/s.)


------------------------------------------------------------
``` SQL
select * from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/sd254_cards.csv') limit 1 FORMAT Vertical
```
Row 1:
──────
User:                  0
CARD INDEX:            0
Card Brand:            Visa
Card Type:             Debit
Card Number:           4344676511950444
Expires:               12/2022
CVV:                   623
Has Chip:              YES
Cards Issued:          2
Credit Limit:          $24295
Acct Open Date:        09/2002
Year PIN last Changed: 2008
Card on Dark Web:      No

1 row in set. Elapsed: 0.021 sec. 

------------------------------------------------------------
``` SQL
select * from file('/mnt/hddraid/clickhouse-data/user_files/credit-card-transactions/sd254_users.csv') limit 1 FORMAT Vertical
```
Row 1:
──────
Person:                      Hazel Robinson
Current Age:                 53
Retirement Age:              66
Birth Year:                  1966
Birth Month:                 11
Gender:                      Female
Address:                     462 Rose Lane
Apartment:                   ᴺᵁᴸᴸ
City:                        La Verne
State:                       CA
Zipcode:                     91750
Latitude:                    34.15
Longitude:                   -117.76
Per Capita Income - Zipcode: $29278
Yearly Income - Person:      $59696
Total Debt:                  $127613
FICO Score:                  787
Num Credit Cards:            5

1 row in set. Elapsed: 0.021 sec. 