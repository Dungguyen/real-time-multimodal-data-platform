(real-time-multimodal-data-platform) PS D:\real-time-multimodal-data-platform> git remote -v

(real-time-multimodal-data-platform) PS D:\real-time-multimodal-data-platform> git remote add origin https://github.com/Dungguyen/real-time-multimodal-data-platform.git
(real-time-multimodal-data-platform) PS D:\real-time-multimodal-data-platform> git remote -v

origin  https://github.com/Dungguyen/real-time-multimodal-data-platform.git (fetch)
origin  https://github.com/Dungguyen/real-time-multimodal-data-platform.git (push)
(real-time-multimodal-data-platform) PS D:\real-time-multimodal-data-platform> git push -u origin main --force
Enumerating objects: 31, done.
Counting objects: 100% (31/31), done.
Delta compression using up to 12 threads
Compressing objects: 100% (21/21), done.
Writing objects: 100% (31/31), 26.39 KiB | 26.39 MiB/s, done.
Total 31 (delta 7), reused 31 (delta 7), pack-reused 0 (from 0)
remote: Resolving deltas: 100% (7/7), done.
To https://github.com/Dungguyen/real-time-multimodal-data-platform.git
 * [new branch]      main -> main
branch 'main' set up to track 'origin/main'.

2. Project Goals

The main objectives of this project are:

Build a reproducible large-scale data processing pipeline.
Process raw JSON/GZIP datasets efficiently.
Normalize heterogeneous Amazon product and review data.
Convert raw data into columnar Parquet format.
Perform data profiling and quality validation.
Detect duplicate and invalid records.
Separate invalid records into a quarantine layer.
Maintain clear data contracts and schemas.
Keep large datasets outside Git history.
Provide a foundation for future streaming and distributed processing.
Prepare the architecture for Kafka, PySpark, and cloud object storage.
3. Dataset

This project uses the Amazon Product Data dataset.

The dataset contains product metadata and customer reviews from the Amazon
Electronics category.

The datasets are intentionally not stored in this GitHub repository because
the raw and processed files are several gigabytes in size.

Instead, users should download the original datasets and place them into the
expected directory structure.

3.1 Required datasets
Amazon Electronics Reviews

Filename:

Electronics_5.json.gz

This dataset contains customer review information such as:

Reviewer information
Product ASIN
Rating
Review text
Review timestamp
Helpful votes
Review summary

Download:

Amazon Electronics Reviews

https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/categoryFilesSmall/Electronics_5.json.gz

Amazon Electronics Product Metadata

Filename:

meta_Electronics.json.gz

This dataset contains product metadata such as:

Product ASIN
Product title
Product category
Price
Brand
Product description
Product images
Related products
Sales rank

Download:

Amazon Electronics Product Metadata

https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/metaFiles2/meta_Electronics.json.gz

4. Dataset Setup

After downloading the two datasets, place them in:

data/
└── source/
    └── amazon/
        ├── Electronics_5.json.gz
        └── meta_Electronics.json.gz

The repository contains .gitkeep files to preserve the directory structure.

The actual dataset files are ignored by Git.

5. Why Is the Dataset Not Stored in GitHub?

The raw and processed datasets can reach several gigabytes in size.

Git is designed primarily for versioning source code, configuration files,
documentation, and relatively small artifacts.

Large datasets should instead be stored in dedicated data storage systems such
as:

Google Cloud Storage
Amazon S3
Azure Blob Storage
MinIO
Hugging Face Datasets
Kaggle
Data Lake / Lakehouse storage

Therefore, this repository follows the principle:

GitHub
    │
    ├── Source code
    ├── Schemas
    ├── Documentation
    ├── Tests
    └── Configuration
         │
         │ dataset URL
         ▼
External Dataset

The .gitignore configuration prevents large data files from being committed.

Ignored formats include:

*.parquet
*.json.gz
*.csv
6. Data Architecture

The project uses a layered data architecture.

                     Raw Dataset
                         │
                         ▼
                ┌─────────────────┐
                │      Source     │
                │   JSON / GZIP   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │      Silver     │
                │ Normalized Data │
                │     Parquet     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Data Validation │
                │  & Profiling    │
                └───────┬─┬───────┘
                        │ │
             Valid      │ │      Invalid
                        │ │
                        ▼ ▼
              ┌──────────┐ ┌─────────────┐
              │ Canonical│ │ Quarantine  │
              │   Data   │ │   Records   │
              └─────┬────┘ └─────────────┘
                    │
                    ▼
             Analytics / ML / AI
7. Data Layers
7.1 Source Layer

Location:

data/source/

Contains the original downloaded datasets.

Example:

data/source/amazon/
├── Electronics_5.json.gz
└── meta_Electronics.json.gz

The source layer should remain as close as possible to the original dataset.

7.2 Silver Layer

Location:

data/silver/

The Silver layer contains normalized and structured datasets stored in Parquet.

Example:

data/silver/
├── products/
│   └── products.parquet
│
└── reviews/
    └── reviews.parquet

The Silver layer is designed for:

Efficient analytical processing
Columnar storage
Schema consistency
Downstream transformations
Data validation
7.3 Canonical Layer

Location:

data/canonical/

The Canonical layer contains cleaned and standardized datasets suitable for
downstream consumers.

Example:

data/canonical/
├── products/
│   └── products.parquet
│
└── reviews/
    └── reviews.parquet

The canonical layer is intended to provide a stable representation of the
business entities used by downstream systems.

7.4 Quarantine Layer

Location:

data/quarantine/

Invalid records are separated from the main processing pipeline instead of
being silently discarded.

Example:

data/quarantine/
├── products/
│   └── invalid_products.parquet
│
└── reviews/
    └── invalid_reviews.parquet

This enables:

Error investigation
Data quality monitoring
Reprocessing
Auditing
Debugging
8. Data Processing Pipeline

The current processing workflow is:

Download Dataset
       │
       ▼
Raw JSON/GZIP
       │
       ▼
Data Normalization
       │
       ▼
Silver Parquet
       │
       ▼
Data Profiling
       │
       ▼
Schema Validation
       │
       ├───────────────┐
       │               │
       ▼               ▼
Valid Records      Invalid Records
       │               │
       ▼               ▼
Canonical          Quarantine
       │
       ▼
Downstream Processing
9. Data Normalization

The normalization pipeline converts raw Amazon JSON/GZIP data into structured
Parquet datasets.

Main script:

scripts/normalize_amazon.py

Example:

python scripts/normalize_amazon.py --entity product

For reviews:

python scripts/normalize_amazon.py --entity review

The normalized data is written into the Silver layer.

10. Data Profiling

The project includes data profiling functionality to understand the structure
and quality of the source datasets.

Main script:

scripts/profile_amazon.py

The profiling process examines characteristics such as:

Number of records
Number of columns
Data types
Missing values
Unique values
Duplicate records
Field distributions
Basic dataset statistics

Example:

python scripts/profile_amazon.py

Profiling reports are stored under:

reports/
11. Data Quality

The project includes automated data quality checks.

Main script:

scripts/data_quality.py

Run all entity checks:

python scripts/data_quality.py --entity all

Run product validation:

python scripts/data_quality.py --entity product

Run review validation:

python scripts/data_quality.py --entity review

The validation pipeline checks issues such as:

Missing required fields
Invalid data types
Duplicate product IDs
Invalid review records
Schema violations
Invalid values
Null values
Data consistency
12. Data Quality Results

The current Electronics dataset contains approximately:

Products:
~786K records


Reviews:
~6.7M records

During profiling and validation, the pipeline identifies data quality issues
such as duplicate product records and missing values.

For example, the product metadata contains duplicate ASIN records that are
detected during profiling.

Invalid records are separated into the quarantine layer instead of being
removed permanently.

13. Data Contracts and Schemas

The project maintains explicit schemas for the processed datasets.

Schemas are stored under:

schemas/

Example:

schemas/
├── product_schema.json
└── review_schema.json

Data contracts are documented under:

docs/

Example:

docs/data_contract.md

The purpose of data contracts is to define:

Required fields
Data types
Nullable fields
Entity relationships
Validation rules
Expected output structure
14. Project Structure
real-time-multimodal-data-platform/
│
├── data/
│   ├── source/
│   │   ├── amazon/
│   │   │   ├── Electronics_5.json.gz
│   │   │   └── meta_Electronics.json.gz
│   │   └── .gitkeep
│   │
│   ├── silver/
│   │   ├── products/
│   │   │   └── products.parquet
│   │   ├── reviews/
│   │   │   └── reviews.parquet
│   │   └── .gitkeep
│   │
│   ├── canonical/
│   │   ├── products/
│   │   │   └── products.parquet
│   │   ├── reviews/
│   │   │   └── reviews.parquet
│   │   └── .gitkeep
│   │
│   └── quarantine/
│       ├── products/
│       │   └── invalid_products.parquet
│       ├── reviews/
│       │   └── invalid_reviews.parquet
│       └── .gitkeep
│
├── scripts/
│   ├── normalize_amazon.py
│   ├── profile_amazon.py
│   ├── data_quality.py
│   └── inspect_duplicates.py
│
├── schemas/
│   ├── product_schema.json
│   └── review_schema.json
│
├── reports/
│   ├── products_profile.json
│   ├── reviews_profile.json
│   └── data_quality_report.json
│
├── docs/
│   └── data_contract.md
│
├── README.md
├── requirements.txt
└── .gitignore

Note: Large data files shown in the structure above are local files and are
intentionally excluded from GitHub.

15. Installation

Clone the repository:

git clone https://github.com/Dungguyen/real-time-multimodal-data-platform.git
cd real-time-multimodal-data-platform

Create a virtual environment:

Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt
16. Download the Dataset

Download:

Electronics_5.json.gz
meta_Electronics.json.gz

Place them in:

data/source/amazon/

Verify:

data/source/amazon/
├── Electronics_5.json.gz
└── meta_Electronics.json.gz
17. Running the Pipeline
Step 1 — Normalize product data
python scripts/normalize_amazon.py --entity product
Step 2 — Normalize review data
python scripts/normalize_amazon.py --entity review
Step 3 — Run data profiling
python scripts/profile_amazon.py
Step 4 — Run data quality checks
python scripts/data_quality.py --entity all
Step 5 — Inspect duplicates
python scripts/inspect_duplicates.py
18. Output

After running the pipeline, the expected structure is:

data/
│
├── source/
│   └── amazon/
│
├── silver/
│   ├── products/
│   │   └── products.parquet
│   └── reviews/
│       └── reviews.parquet
│
├── canonical/
│   ├── products/
│   │   └── products.parquet
│   └── reviews/
│       └── reviews.parquet
│
└── quarantine/
    ├── products/
    │   └── invalid_products.parquet
    └── reviews/
        └── invalid_reviews.parquet

Reports are generated under:

reports/
19. Technology Stack
Current
Python
Pandas
PyArrow
Parquet
JSON / GZIP
Git
GitHub
Planned / Extensible
Apache Kafka
Apache Spark
PySpark
Spark Structured Streaming
MinIO / Amazon S3 / Google Cloud Storage
Apache Iceberg / Delta Lake
PostgreSQL
BigQuery
Airflow
Docker
Kubernetes
Prometheus
Grafana
20. Why Parquet?

The project converts raw JSON data into Parquet because Parquet provides:

Columnar storage
Efficient compression
Faster analytical queries
Predicate pushdown
Column pruning
Efficient integration with Spark
Compatibility with data lake and lakehouse architectures

The transition is:

JSON/GZIP
   │
   ▼
Normalization
   │
   ▼
Parquet
   │
   ├── Pandas
   ├── PyArrow
   ├── Spark
   ├── DuckDB
   └── BigQuery / Data Lake
21. Scalability Considerations

The current implementation is designed as a local batch-processing foundation.

For larger workloads, the architecture can be extended to distributed processing.

Current:

Local Dataset
     │
     ▼
Python / PyArrow
     │
     ▼
Parquet

Future:

                 ┌──────────────┐
                 │ Data Sources │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    Kafka     │
                 │ Event Stream │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    Spark     │
                 │  Streaming  │
                 └──────┬───────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        ┌─────────┐          ┌─────────┐
        │ Bronze  │          │  Kafka  │
        │  Data   │          │ Topics  │
        └────┬────┘          └─────────┘
             │
             ▼
        ┌─────────┐
        │ Silver  │
        │  Data   │
        └────┬────┘
             │
             ▼
        ┌─────────┐
        │  Gold   │
        │  Data   │
        └────┬────┘
             │
       ┌─────┴─────┐
       ▼           ▼
   Analytics       AI/ML
22. Future Improvements

The project can be extended in several directions.

Real-time ingestion

Introduce Apache Kafka for real-time event ingestion.

Producer
   │
   ▼
Kafka
   │
   ▼
Spark Structured Streaming
Distributed processing

Replace or complement local Pandas/PyArrow processing with PySpark.

This enables processing datasets that are significantly larger than the
available memory of a single machine.

Data Lake

Move data storage from the local filesystem to:

Amazon S3
Google Cloud Storage
Azure Blob Storage
MinIO
Lakehouse

Introduce an open table format such as:

Apache Iceberg
Delta Lake

This provides features such as:

Schema evolution
ACID transactions
Time travel
Partition management
Incremental processing
Orchestration

Use Apache Airflow to orchestrate:

Download
   ↓
Normalize
   ↓
Validate
   ↓
Transform
   ↓
Publish
Observability

Add:

Prometheus
Grafana
Structured logging
Pipeline metrics
Data quality metrics
Processing latency monitoring
AI / ML integration

The processed product and review datasets can be used for:

Recommendation systems
Product classification
Review sentiment analysis
Semantic search
RAG systems
Product embeddings
Multimodal search
23. Git and Data Management

Large datasets are intentionally excluded from Git.

The .gitignore configuration includes:

# Large data files
*.parquet
*.json.gz
*.csv

and data directories are protected:

data/source/*
data/silver/*
data/canonical/*
data/quarantine/*

Only .gitkeep files are tracked to preserve the directory structure.

Therefore:

git add .

is safe for normal source-code commits as long as large datasets remain under
the ignored directories.

Never force-add the datasets using:

git add -f data/
24. Reproducibility

A new developer can reproduce the pipeline using the following workflow:

1. Clone repository
        ↓
2. Install dependencies
        ↓
3. Download Amazon datasets
        ↓
4. Place datasets in data/source/amazon/
        ↓
5. Run normalization
        ↓
6. Run profiling
        ↓
7. Run data quality validation
        ↓
8. Generate Silver / Canonical / Quarantine data

This approach keeps the repository lightweight while preserving a reproducible
data processing workflow.

25. Data Lineage

The current lineage can be summarized as:

Electronics_5.json.gz
        │
        ▼
Review Normalization
        │
        ▼
data/silver/reviews/reviews.parquet
        │
        ▼
Data Quality Validation
        │
        ├──────────────► invalid_reviews.parquet
        │
        ▼
data/canonical/reviews/reviews.parquet

Product metadata follows a similar flow:

meta_Electronics.json.gz
        │
        ▼
Product Normalization
        │
        ▼
data/silver/products/products.parquet
        │
        ▼
Data Quality Validation
        │
        ├──────────────► invalid_products.parquet
        │
        ▼
data/canonical/products/products.parquet
26. Key Engineering Principles

This project follows several data engineering principles:

Separation of code and data
GitHub → Code + Documentation
Storage → Large Datasets
Layered data architecture
Source → Silver → Canonical
                  │
                  └── Quarantine
Data quality before consumption

Invalid records are isolated rather than silently discarded.

Schema-first design

Data contracts and schemas define the expected structure of processed data.

Reproducibility

The pipeline can be rebuilt from the original source datasets.

Scalability

The architecture is designed so that local batch processing can later be
replaced or complemented by distributed and streaming technologies.

27. Author

Dung Nguyen

Data Engineering / Data Platform Project

GitHub:

https://github.com/Dungguyen

28. License and Dataset Attribution

This repository contains source code and documentation.

The Amazon datasets are obtained from the original dataset provider and are
not redistributed through this repository.

Users are responsible for reviewing and complying with the licensing,
attribution, and usage requirements of the original dataset before using it
for commercial or other purposes.