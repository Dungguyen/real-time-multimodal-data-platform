# Amazon Data Contract

## 1. Purpose

This document defines the canonical data contract for product
and review data used by the real-time multimodal data platform.

## 2. Product Entity

### Required fields

- product_id
- asin

### Optional fields

- title
- brand
- category
- main_category
- description
- features
- price
- image_urls
- high_res_image_urls
- also_buy
- also_view
- similar_items
- product_date

## 3. Review Entity

### Required fields

- review_id
- product_id
- asin
- reviewer_id
- rating
- review_timestamp

### Optional fields

- reviewer_name
- verified
- review_text
- summary
- review_time
- vote
- image_urls
- style

## 4. Schema Version

Initial version:

v1

Schema evolution must be backward compatible whenever possible.

## 5. Data Quality Rules

### Product

- ASIN must not be null.
- product_id must not be null.
- Duplicate ASINs must be detected.
- Invalid records must be quarantined.

### Review

- review_id must not be null.
- ASIN must not be null.
- reviewer_id must not be null.
- rating must be between 0 and 5.
- review_timestamp must be valid.
- Reviews referencing unknown products must be flagged.

## 6. Data Layers

Source
→ Bronze
→ Silver
→ Gold

### Source

Original Amazon dataset.

### Bronze

Raw records preserved with minimal transformation.

### Silver

Validated, normalized and deduplicated canonical entities.

### Gold

Business-ready and ML/AI-ready datasets.