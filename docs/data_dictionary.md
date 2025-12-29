# Data Dictionary
## Healthcare Resource Optimization Analytics Platform

---

## NHAMCS Dataset Variables

### Demographic Variables

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `AGE` | Integer | Patient age in years | 0-120 |
| `age_group` | Categorical | Derived age group | 0-17, 18-44, 45-64, 65+ |
| `SEX` | Categorical | Patient sex | 1=Male, 2=Female |
| `ETHNIC` | Categorical | Ethnicity | 1=Hispanic, 2=Not Hispanic |
| `RACEUN` | Categorical | Race/ethnicity combined | Multiple categories |

### Visit Characteristics

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `VDATE` | Date | Visit date | MMDDYYYY format |
| `visit_date` | DateTime | Cleaned visit date | Standardized datetime |
| `VMONTH` | Integer | Visit month | 1-12 |
| `VDAYR` | Integer | Visit day of week | 1=Sunday...7=Saturday |
| `day_of_week` | Integer | Day of week (Python format) | 0=Monday...6=Sunday |
| `is_weekend` | Binary | Weekend indicator | 0=Weekday, 1=Weekend |

### Temporal Features

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `ARRTIME` | Integer | Arrival time (24hr) | 0-2400 |
| `arrival_hour` | Integer | Arrival hour | 0-23 |
| `time_of_day` | Categorical | Time period | Night, Morning, Afternoon, Evening |
| `WAITTIME` | Integer | Wait time in minutes | 0-999 |
| `TIMEMD` | Integer | Time with physician (min) | 0-999 |
| `LOS` | Integer | Length of stay (min) | 0-9999 |

### Clinical Variables

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `IMMEDR` | Integer | Immediacy/triage level | 1=Immediate, 2=Emergent, 3=Urgent, 4=Semi-urgent, 5=Non-urgent |
| `high_acuity` | Binary | High acuity indicator | 0=Low acuity (3-5), 1=High acuity (1-2) |
| `DIAG1-3` | String | ICD-10 diagnosis codes | Alphanumeric codes |
| `RFV1-3` | String | Reason for visit codes | Alphanumeric codes |
| `CAUSE1-3` | String | Cause of injury codes | E-codes |

### Insurance & Payment

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `PAYTYPER` | Integer | Expected payment source | 1=Private, 2=Medicare, 3=Medicaid, 4=Workers comp, 5=Self-pay, 6=No charge/charity, 7=Other |
| `has_insurance` | Binary | Insurance indicator | 0=No insurance (5,6), 1=Has insurance |
| `PAYTYPE` | Integer | Primary payment type | Similar to PAYTYPER |

### Procedures & Services

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `IMAGING` | Binary | Any imaging performed | 0=No, 1=Yes |
| `XRAY` | Binary | X-ray performed | 0=No, 1=Yes |
| `CATSCAN` | Binary | CT scan performed | 0=No, 1=Yes |
| `MRI` | Binary | MRI performed | 0=No, 1=Yes |
| `BLOODCUL` | Binary | Blood culture taken | 0=No, 1=Yes |
| `URINE` | Binary | Urinalysis performed | 0=No, 1=Yes |

### Disposition

| Variable | Type | Description | Values |
|----------|------|-------------|--------|
| `DISCHAR` | Integer | Discharge disposition | 1=Admitted, 2=Transferred, 3=Discharged home, 4=Left without seen, 5=DOA, 6=Other |
| `ADMITHOS` | Binary | Admitted to hospital | 0=No, 1=Yes |
| `LEFTBTRT` | Binary | Left before treatment | 0=No, 1=Yes |

---

## Web Scraped Data Variables

### CDC News Data

| Variable | Type | Description |
|----------|------|-------------|
| `date` | DateTime | Article publication date |
| `title` | String | Article headline |
| `content` | String | Full article text (max 5000 chars) |
| `url` | String | Article URL |
| `keywords` | String | Extracted health keywords (comma-separated) |
| `category` | Categorical | Article category (Outbreak, Alert, Vaccination, etc.) |
| `source` | String | Data source identifier (CDC) |
| `scraped_at` | DateTime | Timestamp of scraping |

### Reddit Data

| Variable | Type | Description |
|----------|------|-------------|
| `date` | DateTime | Post creation date |
| `subreddit` | String | Subreddit name |
| `post_id` | String | Unique Reddit post ID |
| `title` | String | Post title |
| `text` | String | Post text content |
| `full_text_clean` | String | Cleaned combined title+text |
| `score` | Integer | Reddit score (upvotes - downvotes) |
| `num_comments` | Integer | Number of comments |
| `upvote_ratio` | Float | Upvote ratio (0-1) |
| `sentiment_polarity` | Float | Sentiment score (-1 to 1) |
| `sentiment_subjectivity` | Float | Subjectivity score (0 to 1) |
| `symptoms_mentioned` | String | Detected symptoms (comma-separated) |
| `is_health_related` | Binary | Health-related post indicator |
| `url` | String | Reddit post URL |

### Twitter Data

| Variable | Type | Description |
|----------|------|-------------|
| `date` | DateTime | Tweet creation date |
| `tweet_id` | String | Unique tweet ID |
| `username` | String | Twitter username |
| `text` | String | Original tweet text |
| `clean_text` | String | Cleaned tweet text |
| `likes` | Integer | Number of likes |
| `retweets` | Integer | Number of retweets |
| `replies` | Integer | Number of replies |
| `sentiment_polarity` | Float | Sentiment score (-1 to 1) |
| `sentiment_subjectivity` | Float | Subjectivity score (0 to 1) |
| `health_keywords` | String | Health keywords found |
| `query` | String | Search query used |

---

## Engineered Features

### Temporal Features

| Variable | Type | Description |
|----------|------|-------------|
| `year` | Integer | Year extracted from date |
| `month` | Integer | Month (1-12) |
| `quarter` | Integer | Quarter (1-4) |
| `week_of_year` | Integer | ISO week number |
| `is_monday` | Binary | Monday indicator |
| `is_friday` | Binary | Friday indicator |
| `is_holiday` | Binary | Federal holiday indicator |
| `is_flu_season` | Binary | Flu season months (Oct-Mar) |

### Cyclical Encodings

| Variable | Type | Description |
|----------|------|-------------|
| `day_of_week_sin` | Float | Sine encoding of day of week |
| `day_of_week_cos` | Float | Cosine encoding of day of week |
| `month_sin` | Float | Sine encoding of month |
| `month_cos` | Float | Cosine encoding of month |
| `hour_sin` | Float | Sine encoding of hour |
| `hour_cos` | Float | Cosine encoding of hour |

### Aggregated Features

| Variable | Type | Description |
|----------|------|-------------|
| `news_mentions` | Integer | CDC news articles count per day |
| `news_mentions_lag1` | Integer | News mentions 1 day prior |
| `news_mentions_lag3` | Integer | News mentions 3 days prior |
| `news_mentions_lag7` | Integer | News mentions 7 days prior |
| `reddit_posts` | Integer | Reddit health posts per day |
| `reddit_sentiment` | Float | Average Reddit sentiment per day |
| `reddit_sentiment_7d` | Float | 7-day rolling average sentiment |
| `tweet_count` | Integer | Health tweets per day |
| `twitter_sentiment` | Float | Average Twitter sentiment per day |

### Interaction Features

| Variable | Type | Description |
|----------|------|-------------|
| `weekend_evening` | Binary | Weekend + evening hours (6-11pm) |
| `senior_uninsured` | Binary | Age 65+ without insurance |
| `weekend_high_acuity` | Float | Weekend × high_acuity interaction |

---

## Model Output Variables

| Variable | Type | Description |
|----------|------|-------------|
| `predicted_acuity` | Binary | Predicted high acuity classification |
| `acuity_probability` | Float | Probability of high acuity (0-1) |
| `predicted_visits` | Float | Forecasted ER visit count |
| `forecast_lower` | Float | Lower bound of forecast (95% CI) |
| `forecast_upper` | Float | Upper bound of forecast (95% CI) |

---

## Notes

- All dates are standardized to datetime format
- Missing values are handled via median (numeric) or mode (categorical) imputation
- Categorical variables are label-encoded for modeling
- Text fields are limited to prevent memory issues
- Sentiment scores use TextBlob library (polarity: -1=negative, +1=positive)
