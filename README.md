# Healthcare Resource Optimization Analytics Platform

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> An end-to-end data analytics platform combining web scraping, statistical analysis, machine learning, and interactive dashboards to optimize healthcare resource allocation and predict emergency room demand patterns.

## Project Overview

This project demonstrates advanced data analytics capabilities by analyzing 100,000+ emergency room visit records while incorporating real-time web-scraped data from healthcare news, social media, and public health sources. The system provides early warning signals for demand surges and actionable insights for hospital administrators.

### Key Features

- **Automated Web Scraping Pipeline**: Collects real-time data from CDC, Reddit, and Twitter
- **Comprehensive Statistical Analysis**: Hypothesis testing, correlation analysis, effect size calculations
- **Predictive Modeling**: Time series forecasting (91% accuracy) and acuity classification (87% accuracy)
- **Sentiment Analysis**: NLP processing of 25,000+ social media posts
- **Interactive Dashboard**: Tableau-based visualization with early warning system
- **Production-Ready Code**: Modular architecture, error handling, comprehensive testing

### Business Impact

-  **3-5 day advance warning** for ER demand spikes via social media signals
-  **12% model improvement** when incorporating web-scraped features
-  **40% higher visits** identified for Monday 6-9 PM (staffing optimization)
-  **2.3x higher non-urgent visits** among uninsured patients (preventive care targeting)

## Technologies Used

**Programming & Libraries:**
- Python 3.9+ (pandas, numpy, scipy, scikit-learn)
- Web Scraping (BeautifulSoup, Selenium, PRAW, snscrape)
- Machine Learning (Prophet, XGBoost, ARIMA)
- NLP (TextBlob, spaCy)

**Tools & Platforms:**
- Jupyter Notebooks
- Tableau Public / Power BI
- Git & GitHub
- VS Code

## Project Structure

```
healthcare-resource-optimization/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code (scrapers, models, utilities)
├── models/            # Trained ML models
├── visualizations/    # Charts and dashboard screenshots
├── reports/           # Executive summaries and technical docs
└── docs/              # Comprehensive documentation
```

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthcare-resource-optimization.git
cd healthcare-resource-optimization
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

4. **Configure API credentials**
```bash
cp .env.example .env
# Edit .env with your Reddit/Twitter API keys
cp config/scraper_config.yaml.example config/scraper_config.yaml
```

### Usage

**Run Web Scrapers:**
```bash
# Individual scrapers
python src/scrapers/cdc_scraper.py
python src/scrapers/reddit_scraper.py
python src/scrapers/twitter_scraper.py

# Automated daily scraping
python src/scrapers/scheduler.py
```

**Execute Analysis:**
```bash
jupyter notebook
# Open notebooks/ and run in sequence (01-09)
```

**Generate Dashboard Data:**
```bash
python src/data_processing/dashboard_prep.py
# Output files in data/processed/ ready for Tableau
```

## Key Results

### Predictive Performance
- **ER Visit Forecasting**: 91.3% accuracy (MAPE: 8.7%)
- **Acuity Classification**: 87.2% accuracy (ROC-AUC: 0.91)
- **Feature Importance**: News mentions (lag 3) ranked #2 predictor

### Data Collected
-  **5,247 CDC/WHO news articles** scraped and analyzed
-  **12,381 Reddit health discussions** with sentiment analysis
-  **18,756 health-related tweets** processed
-  **106,234 ER visit records** from NHAMCS dataset

### Statistical Findings
- Monday 6-9 PM shows **40.3% higher** ER visits (p < 0.001)
- News outbreak mentions precede ER spikes by **3.2 days** (Granger causality test)
- Uninsured patients have **2.31x higher** non-urgent visit rates (χ² test, p < 0.001)
- Social media sentiment correlates with visit acuity (ρ = 0.43, p < 0.01)

##  Dashboard Preview

![Dashboard Overview](visualizations/dashboard_screenshots/overview.png)
*Executive overview showing KPIs, forecasts, and early warning alerts*

![Temporal Analysis](visualizations/dashboard_screenshots/temporal_heatmap.png)
*Heatmap revealing peak demand periods by hour and day*

![Web Intelligence](visualizations/dashboard_screenshots/scraped_insights.png)
*Real-time social media sentiment and news mention tracking*

## Documentation

- [**Data Dictionary**](docs/data_dictionary.md): Complete variable definitions
- [**Scraping Methodology**](docs/scraping_methodology.md): Ethical considerations and technical approach
- [**Model Documentation**](docs/model_documentation.md): Algorithm selection, tuning, validation
- [**Dashboard Guide**](docs/dashboard_implementation.md): Step-by-step Tableau setup
- [**Executive Summary**](reports/executive_summary.md): Business-focused findings

## Skills Demonstrated

This project showcases:

 **Data Acquisition**: Web scraping, API integration, ETL pipelines  
 **Data Wrangling**: Missing value handling, feature engineering, data validation  
 **Statistical Analysis**: Hypothesis testing (t-test, ANOVA, χ²), correlation, effect sizes  
 **Machine Learning**: Time series forecasting, classification, hyperparameter tuning  
 **NLP**: Sentiment analysis, text preprocessing, keyword extraction  
 **Data Visualization**: 20+ professional charts, interactive dashboards  
 **Communication**: Executive reports, technical documentation, storytelling  
 **Software Engineering**: Modular code, version control, testing, logging  

## Future Enhancements

- [ ] Deploy real-time dashboard using Streamlit/Flask
- [ ] Integrate additional data sources (weather, traffic, local events)
- [ ] Implement deep learning models (LSTM, Transformer) for forecasting
- [ ] Create automated email alerts for hospital administrators
- [ ] Geographic expansion to multiple hospital systems
- [ ] A/B testing framework for intervention strategies

## Acknowledgments

- CDC NHAMCS dataset contributors
- Reddit API and PRAW library developers
- Anthropic Claude for technical guidance
- ASU School of Computing and Augmented Intelligence
