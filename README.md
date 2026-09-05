# Amazon Marketplace Product Intelligence Platform

An AI-powered product intelligence platform that collects fresh Amazon marketplace product and review data, processes and analyzes the data, and provides four analytical capabilities through a unified Streamlit application.

The project was developed as an AI Engineer training project with emphasis on **data collection, data preprocessing, machine learning, model persistence, software architecture, testing, and application integration**.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Key Features](#key-features)
* [System Architecture](#system-architecture)
* [Data Pipeline](#data-pipeline)
* [Machine Learning Features](#machine-learning-features)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Technology Stack](#technology-stack)
* [Installation](#installation)
* [Configuration](#configuration)
* [Running the Application](#running-the-application)
* [Running the Data Collection Pipeline](#running-the-data-collection-pipeline)
* [Model Development](#model-development)
* [Testing](#testing)
* [Important Design Decisions](#important-design-decisions)
* [Engineering Considerations](#engineering-considerations)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Author](#author)

---

## Project Overview

The Amazon Marketplace Product Intelligence Platform provides a centralized interface for exploring Amazon marketplace products and generating useful product-level intelligence.

Instead of requiring users to manually inspect individual product pages and reviews, the platform combines collected product information with machine-learning-based analytical features.

The application provides:

1. **Similar Product Recommendation**
2. **Review Sentiment Analysis**
3. **Thumbnail-Based Visual Grouping**
4. **Price Tier Classification**

The final application is implemented using **Streamlit**, while the underlying system is organized into data collection, processing, feature/model, application-service, and UI layers.

---

## Objectives

The main objectives of the project are:

* Collect fresh product information from public Amazon pages.
* Collect publicly visible product reviews.
* Maintain structured and reusable product data.
* Clean and preprocess the collected dataset.
* Develop independent ML/analytical approaches for four product-intelligence features.
* Compare candidate approaches during model development.
* Persist the final models and inference artifacts.
* Integrate all features into one application.
* Provide deterministic and reusable analytical results.
* Build a maintainable Python project rather than a notebook-only solution.

---

# Key Features

## Feature A — Similar Product Recommendation

A content-based recommendation system that finds products similar to a selected product.

### Approach

Product textual information is transformed into a TF-IDF representation.

Similarity is then calculated using cosine similarity.

```text
Product Information
        ↓
Text Representation
        ↓
TF-IDF
        ↓
Product Vector Matrix
        ↓
Cosine Similarity
        ↓
Similarity Ranking
        ↓
Top-K Similar Products
```

The recommendation engine uses persisted artifacts including:

* TF-IDF vectorizer
* Product TF-IDF matrix
* Product metadata

The ranking is deterministic:

1. Similarity score — descending
2. ASIN — ascending for ties

This ensures that the same input produces consistent recommendations when the underlying artifacts remain unchanged.

---

## Feature B — Review Sentiment Analysis

The sentiment system classifies customer reviews into:

* **Positive**
* **Neutral**
* **Negative**

The project evaluates multiple approaches including traditional machine-learning baselines and transformer-based modeling.

The final sentiment system uses a persisted **weighted DistilBERT** model.

### Sentiment Pipeline

```text
Review Text
    ↓
Tokenization
    ↓
DistilBERT
    ↓
Class Probabilities
    ↓
Sentiment Prediction
    ↓
Product-Level Aggregation
```

The system can generate:

* Individual review sentiment
* Sentiment probabilities
* Positive/Neutral/Negative counts
* Sentiment percentages
* Overall product sentiment
* Product sentiment score

### Dataset

The sentiment dataset contains **7,494 reviews**.

The original labeled distribution was approximately:

| Sentiment | Reviews | Percentage |
| --------- | ------: | ---------: |
| Positive  |   6,524 |     87.06% |
| Neutral   |     526 |      7.02% |
| Negative  |     444 |      5.92% |

Because the dataset is imbalanced, class-weighted training and macro-level evaluation were considered.

### Final Weighted DistilBERT Test Performance

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 91.19% |
| Macro Precision | 75.20% |
| Macro Recall    | 73.54% |
| Macro F1        | 73.89% |
| Weighted F1     | 91.55% |

Accuracy should not be interpreted as human-level labeling accuracy. It represents performance on the project's held-out test set.

---

## Feature C — Thumbnail-Based Visual Grouping

Feature C groups product thumbnails based on visual similarity without using Amazon category labels.

### Pipeline

```text
Product Image
      ↓
Image Preprocessing
      ↓
CLIP ViT-B/32
      ↓
512-D Image Embedding
      ↓
L2 Normalization
      ↓
PCA
      ↓
K-Means
      ↓
Visual Group
```

The system uses:

* CLIP ViT-B/32
* 512-dimensional image embeddings
* L2 normalization
* Persisted PCA transformation
* Persisted K-Means clustering model

The current project contains **25 visual groups**.

These groups are model-generated visual clusters and should not be interpreted as official Amazon product categories.

### Persisted Artifacts

```text
models/thumbnail_grouping/
├── clip_vit_b32_pca.joblib
└── kmeans_clip_vit_b32_thumbnail_grouping.joblib
```

The application can also process a new image through the same inference pipeline.

---

## Feature D — Price Tier Classification

Feature D predicts a product's price tier:

* **Budget**
* **Mid-range**
* **Premium**

A key design requirement was to avoid target leakage.

The actual product price is used to construct the target label during dataset preparation, but **price itself is excluded from the model's predictive features**.

### Predictive Information

The classifier uses information such as:

* Product title
* Product description
* Brand
* Review count
* Average rating
* Number of collected reviews
* Title length
* Description length

### Inference Pipeline

```text
Product
   ↓
Feature Construction
   ↓
Text + Numeric + Brand Features
   ↓
Persisted ML Model
   ↓
Predicted Tier
   ↓
Class Probabilities
   ↓
Confidence
```

The classifier returns:

* Predicted price tier
* Confidence
* Class probabilities

The production classifier also reconstructs the same feature schema used during model development.

---

# System Architecture

The project follows a layered architecture.

```text
                    ┌─────────────────────────┐
                    │     Streamlit UI        │
                    │  Dashboard / Pages      │
                    └────────────┬────────────┘
                                 │
                                 ↓
                    ┌─────────────────────────┐
                    │  Application Services   │
                    │                         │
                    │ RecommendationService   │
                    │ SentimentService        │
                    │ ProductRepository       │
                    └────────────┬────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             ↓                   ↓                   ↓
       Recommendation       Sentiment          Other Features
          Engine             Model             C / D Services
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Persisted ML Artifacts  │
                    │        models/           │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │   Processed Dataset     │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │    Data Collection      │
                    │ Amazon / Playwright     │
                    └─────────────────────────┘
```

The Streamlit entry point initializes the repository and the feature services and routes users to the different application pages.

---

# Data Pipeline

The data flow is designed as a multi-stage pipeline.

```text
test_keywords.xlsx
        ↓
KeywordReader
        ↓
Amazon Search
        ↓
Product References
        ↓
product_references.json
        ↓
Full Product Collection
        ↓
products.json
        ↓
Raw Dataset
        ↓
Data Cleaning / Preparation
        ↓
cleaned_products.csv
        ↓
Feature-Specific Processing
        ↓
ML Models / Analytical Artifacts
        ↓
products_with_sentiment.json
        ↓
ProductRepository
        ↓
Streamlit Application
```

The collection pipeline separates product-reference collection from full product collection and saves results incrementally.

This makes the collection process more resilient to individual product failures and allows previously collected products to be skipped using their ASIN.

---

# Incremental Data Collection

The collection pipeline is designed to be recoverable.

For each keyword:

```text
Search keyword
      ↓
Collect product references
      ↓
Immediately persist results
```

For each product:

```text
Product reference
      ↓
Check existing ASINs
      ↓
Already collected?
   ↙          ↘
 Yes           No
 ↓              ↓
Skip        Collect product
               ↓
          Save immediately
```

The pipeline handles keyword-level and product-level exceptions so that one failed operation does not necessarily terminate the complete collection process.

---

# Dataset

The final cleaned product dataset contains:

* **946 products**
* **28 search keywords**
* Product information
* Product descriptions
* Brand
* Price
* Average rating
* Review count
* Product images
* Video URLs
* Collected reviews

The cleaned dataset contains 17 columns, including product metadata, review information, and derived text-length fields.

The review sentiment dataset contains:

* **7,494 reviews**
* Reviewer information
* Star rating
* Review title
* Review text
* Sentiment label

---

# Project Structure

```text
Amazon-Marketplace-Product-Intelligence-Platform/
│
├── data/
│   ├── input/
│   │   └── test_keywords.xlsx
│   │
│   ├── raw/
│   │   └── amazon_products_raw.csv
│   │
│   ├── processed/
│   │   ├── cleaned_products.csv
│   │   ├── feature_a/
│   │   ├── feature_b/
│   │   ├── feature_c/
│   │   └── feature_d_price_tier_dataset.csv
│   │
│   └── output/
│       ├── product_references.json
│       ├── products.json
│       └── products_with_sentiment.json
│
├── models/
│   ├── recommendation/
│   ├── sentiment/
│   │   └── final_sentiment_model/
│   ├── thumbnail_grouping/
│   └── price_tier/
│
├── notebooks/
│   ├── feature_a_product_recommendation/
│   ├── feature_b_review_sentiment/
│   ├── feature_c_thumbnail_grouping/
│   └── feature_d_price_tier/
│
├── scripts/
│   ├── data processing scripts
│   ├── sentiment processing scripts
│   └── analysis scripts
│
├── src/
│   ├── application/
│   │   ├── product_repository.py
│   │   ├── recommendation_service.py
│   │   └── sentiment_service.py
│   │
│   ├── common/
│   │
│   ├── data_collection/
│   │   ├── browser/
│   │   ├── collectors/
│   │   ├── models/
│   │   ├── pipeline/
│   │   └── storage/
│   │
│   ├── data_processing/
│   │
│   └── features/
│       ├── product_recommendation/
│       ├── review_sentiment/
│       ├── thumbnail_grouping/
│       └── price_tier/
│
├── tests/
│
├── ui/
│   ├── components/
│   └── pages/
│
├── streamlit_app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Technology Stack

## Programming

* Python

## Data Collection

* Playwright
* BeautifulSoup
* OpenPyXL

## Data Processing

* pandas
* NumPy

## Machine Learning

* scikit-learn
* PyTorch
* Hugging Face Transformers
* CLIP
* joblib

## Visualization / Application

* Streamlit
* Matplotlib
* Altair

## Development

* Jupyter Notebook
* pytest
* Git
* GitHub

The repository pins its Python dependencies in `requirements.txt`.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/mirtasrif9-ai/Amazon-Marketplace-Product-Intelligence-Platform.git
cd Amazon-Marketplace-Product-Intelligence-Platform
```

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## 4. Install Playwright browser

```powershell
playwright install
```

---

# Configuration

Create a `.env` file from the provided template:

```powershell
copy .env.example .env
```

Configure the required environment variables according to the project environment.

Do not commit secret API keys or credentials to GitHub.

---

# Running the Application

Start the Streamlit application with:

```powershell
streamlit run streamlit_app.py
```

The application provides the following navigation options:

```text
Dashboard
Product Explorer
Feature A — Recommendations
Feature B — Sentiment
Feature C — Thumbnail Groups
Feature D — Price Tier
```

The application loads the persisted product dataset and feature artifacts rather than retraining models every time the application starts.

---

# Product Explorer

The Product Explorer provides a centralized view of an individual product.

A selected product can expose:

* Product information
* Similar products
* Review sentiment
* Visual grouping
* Price tier prediction
* Amazon product link
* Product description
* Reviews and analytical information

This allows the user to inspect the product and its associated intelligence from one interface.

---

# Model Development

Each feature has its own model-development workflow.

## Feature A

```text
Data Preparation
      ↓
EDA
      ↓
Text Feature Engineering
      ↓
TF-IDF
      ↓
Cosine Similarity
      ↓
Recommendation Evaluation
      ↓
Persist Artifacts
```

## Feature B

```text
Data Preparation
      ↓
Sentiment Labeling
      ↓
EDA
      ↓
Traditional ML Baselines
      ↓
DistilBERT
      ↓
Class-Weighted Training
      ↓
Threshold Analysis
      ↓
Final Model
```

## Feature C

```text
Image Validation
      ↓
Image Preprocessing
      ↓
CLIP Embeddings
      ↓
Normalization
      ↓
PCA
      ↓
Clustering Evaluation
      ↓
K-Means
      ↓
Persist Assignments + Models
```

## Feature D

```text
Data Preparation
      ↓
Target Tier Construction
      ↓
Feature Engineering
      ↓
Model Comparison
      ↓
Leakage Check
      ↓
Final Model
      ↓
Persist Model
```

---

# Testing

The project contains automated tests under:

```text
tests/
```

Run the complete test suite with:

```powershell
python -m pytest
```

The application/backend test suite has been validated during development.

Testing covers important components such as:

* Data models
* Collection components
* Application services
* Feature logic
* Sentiment prediction
* Integration behavior

---

# Important Design Decisions

## 1. ASIN as Product Identity

Amazon Standard Identification Number (ASIN) is used as the primary product identifier for deduplication and lookup.

---

## 2. Incremental Persistence

Product references and collected products are saved incrementally.

This protects already collected data if a later request fails or the process is interrupted.

---

## 3. Separation of Responsibilities

The project separates:

```text
Data Collection
       ↓
Data Processing
       ↓
Feature / ML Logic
       ↓
Application Services
       ↓
UI
```

This prevents the Streamlit layer from containing the complete business and ML logic.

---

## 4. Persisted Model Artifacts

Models and preprocessing artifacts are persisted and loaded during inference.

For example, the recommendation engine loads:

```text
tfidf_vectorizer.joblib
product_tfidf_matrix.joblib
product_metadata.csv
```

and validates that the artifacts are consistent before producing recommendations.

---

## 5. Deterministic Recommendation

Recommendation results are sorted by:

```text
similarity_score DESC
ASIN ASC
```

This provides deterministic tie-breaking.

---

## 6. Target Leakage Prevention

For Feature D, price is used to construct the target tier but is deliberately excluded from model input.

This prevents the model from receiving the answer it is supposed to predict.

---

## 7. Lazy Loading for Feature C

The Feature C service loads the heavy CLIP model only when real-time image inference is required.

Precomputed visual-group assignments can therefore be browsed without immediately loading CLIP.

---

# Engineering Considerations

The project is designed around several practical engineering principles:

* Separation of concerns
* Reusable services
* Centralized repository access
* Incremental data persistence
* Failure isolation
* Model persistence
* Input validation
* Deterministic ranking
* Lazy model loading
* Automated testing
* Version control

The goal is not only to demonstrate ML models but also to demonstrate how those models can be integrated into a maintainable application.

---

# Data Collection Constraints

The collection pipeline is intended to operate on publicly accessible Amazon pages.

The project does not attempt to bypass:

* CAPTCHA
* Authentication
* Access controls
* Other explicit access restrictions

Collection should use modest request rates and respect applicable website policies.

---

# Limitations

Some practical limitations remain.

### Amazon page structure

Amazon's HTML structure can change, which may require selector maintenance.

### Anti-bot behavior

Automated browsing may encounter temporary blocking or bot-detection responses.

### Sentiment imbalance

The sentiment dataset is dominated by positive reviews, making minority-class performance more difficult.

### Sentiment evaluation

Model test-set performance should not be interpreted as equivalent to human-level annotation accuracy without a dedicated human-validated benchmark.

### Visual grouping

Visual clusters represent learned similarity patterns rather than official Amazon categories.

### Recommendation evaluation

Content similarity does not necessarily equal user preference. More extensive human relevance evaluation would strengthen the recommendation system.

---

# Future Improvements

Potential future improvements include:

* Stronger automated collection monitoring
* More robust selector management
* More comprehensive request/response logging
* Human-validated sentiment benchmark
* Better minority-class sentiment evaluation
* Human evaluation of recommendation relevance
* Additional recommendation approaches
* Improved visual cluster interpretability
* More extensive integration testing
* Model monitoring and versioning
* Automated CI/CD pipeline
* Application deployment
* Database-backed product storage
* Search and filtering improvements
* User feedback collection for recommendations

---

# Project Development Philosophy

The project follows an incremental development workflow.

Each major stage is developed, tested, persisted, and integrated before moving to the next stage.

```text
Collect
   ↓
Validate
   ↓
Clean
   ↓
Analyze
   ↓
Model
   ↓
Evaluate
   ↓
Persist
   ↓
Integrate
   ↓
Test
   ↓
Deploy / Demonstrate
```

---

# Author

**Mir Tasrif Ahmed**

GitHub:

https://github.com/mirtasrif9-ai

Repository:

https://github.com/mirtasrif9-ai/Amazon-Marketplace-Product-Intelligence-Platform

---

## License

This project was developed as an AI Engineer training project at BJIT.


