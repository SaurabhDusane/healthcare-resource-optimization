# Web Scraping Methodology
## Healthcare Resource Optimization Project

---

## Overview

This document outlines the ethical considerations, technical approach, and best practices for web scraping healthcare data from CDC, Reddit, and Twitter.

---

## Ethical Considerations

### 1. Robots.txt Compliance

**CDC Scraping:**
- Reviewed CDC's robots.txt file
- Respected crawl-delay directives (2 seconds between requests)
- Only scraped publicly accessible news releases
- No login-protected content accessed

**Reddit & Twitter:**
- Used official APIs where possible
- Followed rate limiting guidelines
- Respected user privacy settings

### 2. Rate Limiting

All scrapers implement aggressive rate limiting to avoid server overload:

```python
RATE_LIMITS = {
    'cdc': 2 seconds between requests,
    'reddit': 1 second between requests (PRAW handles internally),
    'twitter': 1 second between requests
}
```

### 3. User-Agent Declaration

Transparent identification in all requests:

```python
User-Agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
```

For APIs:
```python
User-Agent: 'HealthTrendAnalyzer/1.0'
```

### 4. Data Privacy

- **No Personal Identifiable Information (PII)** is collected
- Reddit/Twitter usernames are collected but anonymized in analysis
- Focus on aggregate trends, not individual identification
- Data used solely for research/educational purposes

### 5. Terms of Service Compliance

**Reddit API Terms:**
- Used official PRAW library
- Followed rate limits (60 requests/minute)
- Data not used for commercial purposes
- Attribution provided where applicable

**Twitter/X:**
- Evaluated snscrape for academic use
- Alternative: Official Twitter API v2 (with rate limits)
- Compliance with Developer Agreement

**CDC:**
- Public domain content
- Proper attribution provided
- Educational Fair Use applies

---

## Technical Implementation

### Architecture

```
BaseScraper (Abstract Class)
    ├── Error handling
    ├── Rate limiting
    ├── Logging
    ├── Data validation
    └── CSV export
    
    ├── CDCScraper
    ├── RedditScraper
    └── TwitterScraper
```

### CDC Scraper Implementation

**Target URL:** https://www.cdc.gov/media/releases/

**Method:** BeautifulSoup4 + Requests

**Process:**
1. Fetch main news listing page
2. Extract article cards (limited to 100 most recent)
3. For each article:
   - Extract title, date, URL
   - Visit article page
   - Scrape full content
   - Extract health keywords
   - Categorize article
4. Validate data
5. Save to CSV with timestamp

**Key Features:**
- Fallback content selectors for HTML changes
- Keyword extraction (flu, outbreak, vaccine, etc.)
- Automatic categorization (Outbreak, Alert, Vaccination, etc.)
- ISO datetime parsing

**Sample Output:**
```csv
date,title,content,url,keywords,category,source,scraped_at
2024-01-15,CDC Issues Flu Alert,...,https://...,flu,influenza,Alert,CDC,2024-01-20
```

### Reddit Scraper Implementation

**Target Subreddits:** r/AskDocs, r/HealthAnxiety, r/phoenix, r/arizona

**Method:** PRAW (Python Reddit API Wrapper)

**Authentication:**
```python
reddit = praw.Reddit(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)
```

**Process:**
1. For each subreddit:
   - Fetch last 500 new posts
   - Filter posts from last 30 days
   - Extract title, text, metadata
2. Perform sentiment analysis (TextBlob)
3. Detect symptom keywords
4. Classify as health-related or not
5. Save aggregated data

**Symptom Keywords:**
```python
symptoms = [
    'fever', 'cough', 'shortness of breath', 'chest pain',
    'nausea', 'vomiting', 'headache', 'fatigue', 'sore throat',
    'body aches', 'chills', 'dizziness', 'congestion'
]
```

**Sample Output:**
```csv
date,subreddit,post_id,title,score,sentiment_polarity,symptoms_mentioned,is_health_related
2024-01-15,AskDocs,abc123,Feeling sick...,42,0.3,"fever,cough",True
```

### Twitter Scraper Implementation

**Target Queries:**
- Hashtags: #fluseason, #ERwait, #sicktoday
- Keywords: "emergency room", "urgent care", "feel sick"

**Method:** snscrape (scraping) or Twitter API v2 (if available)

**Process:**
1. For each query:
   - Search tweets from last 7 days
   - Limit to 500 tweets per query
   - Extract text, engagement metrics, timestamp
2. Clean tweet text (remove URLs, mentions, hashtags)
3. Perform sentiment analysis
4. Extract health keywords
5. Remove duplicates by tweet_id
6. Save results

**Text Cleaning:**
```python
def clean_tweet_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbols (keep word)
    text = re.sub(r'#(\w+)', r'\1', text)
    return text.strip()
```

**Sample Output:**
```csv
date,tweet_id,text,likes,retweets,sentiment_polarity,health_keywords
2024-01-15,12345,ER wait times terrible,23,5,-0.6,"emergency,hospital"
```

---

## Data Quality Assurance

### 1. Validation Checks

All scrapers implement validation before saving:

```python
required_columns = ['date', 'title', 'content']
if not validate_data(df, required_columns):
    raise ValidationError("Missing required columns")
```

### 2. Error Handling

**Network Errors:**
- Retry logic with exponential backoff
- Maximum 3 retry attempts
- Fallback to empty dataset on persistent failure

**Parsing Errors:**
- Log warning but continue processing
- Use try-except blocks around HTML parsing
- Default values for failed extractions

**Rate Limit Errors:**
- Respect 429 responses
- Implement exponential backoff
- Log rate limit events

### 3. Logging

Comprehensive logging at INFO level:

```python
logger.info(f"Scraping CDC news from {self.news_url}")
logger.info(f"Found {len(article_cards)} articles to process")
logger.warning(f"Error processing article {idx}: {str(e)}")
logger.error(f"Scraping failed: {str(e)}", exc_info=True)
```

Logs saved to: `logs/{scraper_name}.log`

---

## Automation & Scheduling

### Daily Automated Scraping

**Scheduler:** Python `schedule` library

**Configuration:**
```python
# Run daily at 6:00 AM
schedule.every().day.at("06:00").do(run_all_scrapers)
```

**Process:**
1. CDC scraper (5-10 minutes)
2. Reddit scraper (10-15 minutes)
3. Twitter scraper (5-10 minutes)
4. Generate summary report
5. Email notification on errors (optional)

**Manual Override:**
```bash
python src/scrapers/scheduler.py --now
```

---

## Data Storage

### File Naming Convention

```
data/raw/{source_name}/{source_name}_{timestamp}.csv

Examples:
- data/raw/cdc_news/cdc_news_20240120_060500.csv
- data/raw/reddit_health/reddit_health_20240120_061200.csv
- data/raw/twitter_health/twitter_health_20240120_062000.csv
```

### Retention Policy

- **Raw scraped data:** Retained indefinitely for reproducibility
- **Processed/aggregated data:** Overwritten on updates
- **Logs:** Rotated monthly (kept for 6 months)

---

## Limitations & Challenges

### 1. Website Structure Changes

**Challenge:** CDC website HTML structure may change

**Mitigation:**
- Multiple CSS selectors as fallbacks
- Regular testing of scrapers
- Error logging for failed extractions

### 2. API Rate Limits

**Challenge:** Reddit API limits (60 req/min), Twitter limits vary

**Mitigation:**
- Aggressive rate limiting in code
- Spread requests over time
- Implement caching for repeated queries

### 3. Data Volume

**Challenge:** Large text fields consume memory

**Mitigation:**
- Truncate content to 1000-5000 characters
- Process in batches
- Save incrementally

### 4. Sentiment Analysis Accuracy

**Challenge:** TextBlob may misinterpret medical language

**Mitigation:**
- Use as relative indicator, not absolute truth
- Validate with sample manual annotations
- Consider domain-specific sentiment models (future enhancement)

---

## Best Practices Implemented

✅ **Respectful scraping:** Rate limits, robots.txt compliance  
✅ **Error resilience:** Try-except blocks, logging, retries  
✅ **Data validation:** Schema checks, null handling  
✅ **Modularity:** Base class + specific implementations  
✅ **Documentation:** Inline comments, docstrings  
✅ **Reproducibility:** Timestamped outputs, version control  
✅ **Privacy:** No PII collection, aggregated analysis  
✅ **Transparency:** Clear user-agent, academic purpose  

---

## Future Enhancements

1. **Selenium integration** for JavaScript-heavy sites
2. **Proxy rotation** for larger-scale scraping
3. **MongoDB** for NoSQL storage of unstructured text
4. **Apache Airflow** for enterprise-grade scheduling
5. **spaCy NER** for better entity extraction
6. **Scrapy framework** for more robust scraping

---

## References

- [CDC Terms of Use](https://www.cdc.gov/other/disclaimer.html)
- [Reddit API Documentation](https://www.reddit.com/dev/api/)
- [Twitter Developer Agreement](https://developer.twitter.com/en/developer-terms/agreement)
- [Web Scraping Best Practices](https://www.scrapehero.com/web-scraping-best-practices/)
- [robots.txt Specification](https://www.robotstxt.org/)

---

**Last Updated:** December 28, 2024
