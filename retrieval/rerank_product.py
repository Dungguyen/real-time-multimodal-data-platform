# rerank_product_v2.py
from pathlib import Path
import argparse
import json
import math
import re
import html
from collections import Counter

import pyarrow.parquet as pq


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PRODUCT_SUMMARY = PROJECT_ROOT / "lakehouse" / "gold" / "product_summary" / "product_summary.parquet"
REVIEW_STATS = PROJECT_ROOT / "lakehouse" / "gold" / "product_review_stats" / "product_review_stats.parquet"


# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_TOP_K = 10
METADATA_BATCH_SIZE = 10_000

SEMANTIC_WEIGHT = 0.36
TITLE_WEIGHT = 0.25
CATEGORY_WEIGHT = 0.07
BRAND_WEIGHT = 0.06
MULTIMODAL_WEIGHT = 0.10
QUALITY_WEIGHT = 0.07
POPULARITY_WEIGHT = 0.04
VERIFIED_WEIGHT = 0.05
PRICE_WEIGHT = 0.025

MIN_REVIEWS_FOR_CONFIDENCE = 5

MISSING_METADATA_PENALTY = 0.015
SINGLE_MODALITY_PENALTY = 0.010
WEAK_RELEVANCE_THRESHOLD = 0.20
WEAK_RELEVANCE_PENALTY = 0.015

EXACT_PHRASE_BOOST = 0.035
EXACT_BRAND_BOOST = 0.045
EXACT_CATEGORY_BOOST = 0.025


INTERACTION_WEIGHT = 0.05

MAX_DIVERSITY_PENALTY = 0.08

MAX_SINGLE_MODAL_PENALTY = 0.10

MAX_METADATA_PENALTY = 0.08

MIN_RELEVANCE_FOR_BUSINESS_SIGNALS = 0.35

QUERY_STOPWORDS = {
    "a", "an", "and", "the", "for", "with", "of", "to", "in", "on",
    "or", "by", "from", "is", "are", "this", "that", "at", "please",
    "show", "me", "find", "looking", "look", "want", "need", "get",
}


# ============================================================================
# BASIC HELPERS
# ============================================================================

def safe_float(value, default=0.0):
    if value is None:
        return default
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def clamp(value, minimum=0.0, maximum=1.0):
    value = safe_float(value)
    return max(minimum, min(value, maximum))


def normalize_rating(value):
    rating = safe_float(value)
    return 0.0 if rating <= 0 else clamp(rating / 5.0)


def normalize_popularity(review_count):
    count = safe_float(review_count)
    if count <= 0:
        return 0.0
    return clamp(math.log1p(count) / math.log1p(10000))


def normalize_verified_ratio(value):
    ratio = safe_float(value)
    if ratio > 1:
        ratio /= 100.0
    return clamp(ratio)


def find_column(table, candidates):
    columns = set(table.column_names)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def normalize_string(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(value)).strip().lower())


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize_text(text):
    if text is None:
        return []

    text = normalize_string(text)
    tokens = re.findall(r"[a-z0-9]+", text)

    return [token for token in tokens if token not in QUERY_STOPWORDS]


def unique_tokens(tokens):
    seen = set()
    result = []

    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)

    return result


# ============================================================================
# QUERY / LEXICAL SIGNALS
# ============================================================================

def detect_query_intent(query):
    tokens = unique_tokens(tokenize_text(query))
    normalized = normalize_string(query)

    return {
        "token_count": len(tokens),
        "has_query": bool(tokens),
        "is_specific": len(tokens) >= 2,
        "has_price_intent": bool(
            re.search(
                r"(under|below|less than|up to|at most|over|above|more than|minimum|max|\$|usd|dollar)",
                normalized,
            )
        ),
    }


def calculate_title_relevance(query, title):
    if not query or not title:
        return 0.0

    query_tokens = unique_tokens(tokenize_text(query))
    title_tokens = tokenize_text(title)

    if not query_tokens or not title_tokens:
        return 0.0

    title_set = set(title_tokens)
    normalized_title = " ".join(title_tokens)
    normalized_query = " ".join(query_tokens)

    token_coverage = sum(
        token in title_set for token in query_tokens
    ) / len(query_tokens)

    substring_score = sum(
        token not in title_set and token in normalized_title
        for token in query_tokens
    ) / len(query_tokens)

    phrase_score = 1.0 if normalized_query in normalized_title else 0.0

    matched_ordered = 0
    position = 0

    for token in query_tokens:
        for index in range(position, len(title_tokens)):
            if title_tokens[index] == token:
                matched_ordered += 1
                position = index + 1
                break

    ordered_score = matched_ordered / len(query_tokens)

    positions = [
        title_tokens.index(token)
        for token in query_tokens
        if token in title_set
    ]

    position_score = 0.0
    if positions:
        average_position = sum(positions) / len(positions)
        position_score = clamp(
            1.0 - average_position / max(len(title_tokens), 1)
        )

    return clamp(
        0.50 * token_coverage
        + 0.20 * phrase_score
        + 0.15 * ordered_score
        + 0.10 * substring_score
        + 0.05 * position_score
    )


def field_relevance(query, value):
    query_tokens = unique_tokens(tokenize_text(query))

    if not query_tokens or value is None:
        return 0.0

    value_text = (
        " ".join(str(x) for x in value)
        if isinstance(value, list)
        else str(value)
    )

    field_tokens = set(tokenize_text(value_text))

    if not field_tokens:
        return 0.0

    score = sum(
        token in field_tokens for token in query_tokens
    ) / len(query_tokens)

    normalized_query = " ".join(query_tokens)
    normalized_field = " ".join(tokenize_text(value_text))

    if normalized_query and normalized_query in normalized_field:
        score = 1.0

    return clamp(score)


def calculate_category_relevance(query, category):
    return field_relevance(query, category)


def calculate_brand_relevance(query, brand):
    return field_relevance(query, brand)


def extract_price_constraint(query):
    if not query:
        return None

    text = normalize_string(query)

    patterns = [
        (
            "max",
            r"(?:under|below|less than|up to|at most)\s*\$?\s*(\d+(?:\.\d+)?)",
        ),
        (
            "min",
            r"(?:over|above|more than|at least)\s*\$?\s*(\d+(?:\.\d+)?)",
        ),
        (
            "exact",
            r"\$\s*(\d+(?:\.\d+)?)",
        ),
    ]

    for kind, pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return {
                "type": kind,
                "value": safe_float(match.group(1)),
            }

    return None


def calculate_price_relevance(query, price):
    constraint = extract_price_constraint(query)
    product_price = safe_float(price)

    if constraint is None or product_price <= 0:
        return 0.0

    target = constraint["value"]

    if target <= 0:
        return 0.0

    if constraint["type"] == "max":
        if product_price <= target:
            return 1.0
        overshoot = (product_price - target) / target
        return clamp(math.exp(-4.0 * overshoot))

    if constraint["type"] == "min":
        if product_price >= target:
            return 1.0
        undershoot = (target - product_price) / target
        return clamp(math.exp(-4.0 * undershoot))

    distance = abs(product_price - target) / target
    return clamp(math.exp(-5.0 * distance))


def calculate_query_boost(query, title, brand, category):
    if not query:
        return 0.0

    query_tokens = unique_tokens(tokenize_text(query))
    if not query_tokens:
        return 0.0

    q = " ".join(query_tokens)
    t = " ".join(tokenize_text(title))
    b = " ".join(tokenize_text(brand))
    c = " ".join(tokenize_text(category))

    boost = 0.0

    if q in t:
        boost += EXACT_PHRASE_BOOST
    if q in b:
        boost += EXACT_BRAND_BOOST
    if q in c:
        boost += EXACT_CATEGORY_BOOST

    return clamp(boost, 0.0, 0.10)


# ============================================================================
# MULTIMODAL SIGNALS
# ============================================================================

def is_both_modality(modality):
    return str(modality or "").lower().strip() in {
        "text+image", "text-image", "both"
    }


def calculate_multimodal_agreement(candidate):
    text_score = clamp(candidate.get("normalized_text_score", candidate.get("raw_text_score", 0.0)))
    image_score = clamp(candidate.get("normalized_image_score", candidate.get("raw_image_score", 0.0)))

    if not is_both_modality(candidate.get("modality", "")):
        return 0.0

    if text_score <= 0 or image_score <= 0:
        return 0.0

    return clamp(math.sqrt(text_score * image_score))


def calculate_modality_coverage(candidate):
    text_score = clamp(candidate.get("raw_text_score", 0.0))
    image_score = clamp(candidate.get("raw_image_score", 0.0))

    if text_score > 0 and image_score > 0:
        return 1.0
    if text_score > 0 or image_score > 0:
        return 0.5
    return 0.0


# ============================================================================
# SEMANTIC SCORE
# ============================================================================

def calculate_query_aware_semantic_score(candidate, text_query="", image_query=None):
    text_score = clamp(
        candidate.get("normalized_text_score", candidate.get("raw_text_score", 0.0))
    )
    image_score = clamp(
        candidate.get("normalized_image_score", candidate.get("raw_image_score", 0.0))
    )

    modality = str(candidate.get("modality", "")).lower().strip()

    has_text_query = bool(str(text_query or "").strip())
    has_image_query = bool(image_query)

    if is_both_modality(modality):
        if has_text_query and has_image_query:
            return clamp(0.60 * text_score + 0.40 * image_score)
        if has_text_query:
            return clamp(0.80 * text_score + 0.20 * image_score)
        if has_image_query:
            return clamp(0.30 * text_score + 0.70 * image_score)
        return clamp(0.50 * text_score + 0.50 * image_score)

    if modality == "text-only":
        return text_score if has_text_query else 0.35 * text_score

    if modality == "image-only":
        if has_image_query:
            return image_score
        if has_text_query:
            return 0.70 * image_score
        return image_score

    if has_text_query and has_image_query:
        if text_score > 0 and image_score > 0:
            return clamp(0.60 * text_score + 0.40 * image_score)
        return max(text_score, image_score)

    if has_text_query:
        return text_score if text_score > 0 else 0.70 * image_score

    if has_image_query:
        return image_score if image_score > 0 else 0.35 * text_score

    return max(text_score, image_score)


# ============================================================================
# QUALITY
# ============================================================================

def calculate_quality_score(rating, review_count):
    rating = safe_float(rating)
    review_count = max(int(safe_float(review_count)), 0)

    if review_count <= 0:
        return 0.5

    rating_score = clamp(rating / 5.0)

    confidence = 1.0 - math.exp(
        -review_count / MIN_REVIEWS_FOR_CONFIDENCE
    )

    return clamp(
        confidence * rating_score
        + (1.0 - confidence) * 0.5
    )


# ============================================================================
# RETRIEVAL SCORE NORMALIZATION
# ============================================================================

def normalize_candidate_scores(candidates):
    """
    Normalize text/image scores within the current candidate pool.

    This is useful when the vector DB score distribution changes between
    queries. Original raw scores are preserved for debugging.
    """

    for raw_key, normalized_key in (
        ("raw_text_score", "normalized_text_score"),
        ("raw_image_score", "normalized_image_score"),
    ):
        values = [
            clamp(candidate.get(raw_key, 0.0))
            for candidate in candidates
            if safe_float(candidate.get(raw_key, 0.0)) > 0
        ]

        if not values:
            for candidate in candidates:
                candidate[normalized_key] = 0.0
            continue

        minimum = min(values)
        maximum = max(values)
        spread = maximum - minimum

        for candidate in candidates:
            raw = clamp(candidate.get(raw_key, 0.0))

            if raw <= 0:
                normalized = 0.0
            elif spread < 1e-9:
                normalized = raw
            else:
                scaled = (raw - minimum) / spread
                normalized = clamp(0.15 + 0.85 * scaled)

            candidate[normalized_key] = normalized

    return candidates


# ============================================================================
# WEIGHTS
# ============================================================================

def calculate_dynamic_weights(
    text_query,
    title_score,
    category_score,
    brand_score,
    price_score=0.0,
    modality="",
):
    weights = {
        "semantic": SEMANTIC_WEIGHT,
        "title": TITLE_WEIGHT,
        "category": CATEGORY_WEIGHT,
        "brand": BRAND_WEIGHT,
        "multimodal": MULTIMODAL_WEIGHT,
        "quality": QUALITY_WEIGHT,
        "popularity": POPULARITY_WEIGHT,
        "verified": VERIFIED_WEIGHT,
        "price": PRICE_WEIGHT,
    }

    query_tokens = unique_tokens(tokenize_text(text_query))

    if not query_tokens:
        removed = (
            weights["title"]
            + weights["category"]
            + weights["brand"]
            + weights["price"]
        )

        weights["title"] = 0.0
        weights["category"] = 0.0
        weights["brand"] = 0.0
        weights["price"] = 0.0
        weights["semantic"] += removed

    else:
        if title_score >= 0.90:
            weights["title"] += 0.05
            weights["semantic"] -= 0.05
        elif title_score >= 0.75:
            weights["title"] += 0.03
            weights["semantic"] -= 0.03

        if category_score >= 0.80:
            weights["category"] += 0.015
            weights["semantic"] -= 0.015

        if brand_score >= 0.80:
            weights["brand"] += 0.015
            weights["semantic"] -= 0.015

        if price_score > 0:
            weights["price"] += 0.015
            weights["semantic"] -= 0.015

    if is_both_modality(modality):
        weights["multimodal"] += 0.02
        weights["semantic"] -= 0.02

    weights["semantic"] = max(0.0, weights["semantic"])

    total = sum(weights.values())

    if total > 0:
        for key in weights:
            weights[key] /= total

    return weights



# ============================================================================
# V3 QUERY INTENT
# ============================================================================

def detect_advanced_query_intent(query):
    """
    Lightweight deterministic query intent extraction.

    Detects:
        - budget / price constraints
        - cheap / premium language
        - brand/category-like queries
        - query specificity

    It intentionally does not call an LLM.
    """

    text = str(query or "").lower().strip()
    tokens = unique_tokens(tokenize_text(text))

    budget_match = re.search(
        r"(?:under|below|less than|max(?:imum)?|up to)\s*\$?\s*(\d+(?:\.\d+)?)",
        text,
    )
    over_match = re.search(
        r"(?:over|above|more than|at least)\s*\$?\s*(\d+(?:\.\d+)?)",
        text,
    )
    dollar_match = re.search(
        r"\$\s*(\d+(?:\.\d+)?)",
        text,
    )

    budget = None
    price_mode = None

    if budget_match:
        budget = safe_float(budget_match.group(1))
        price_mode = "under"
    elif over_match:
        budget = safe_float(over_match.group(1))
        price_mode = "over"
    elif dollar_match:
        budget = safe_float(dollar_match.group(1))
        price_mode = "target"

    cheap_terms = {"cheap", "affordable", "budget", "inexpensive", "lowcost"}
    premium_terms = {"premium", "luxury", "professional", "flagship", "highend"}

    has_cheap_intent = any(t in text.replace("-", " ").split() for t in cheap_terms)
    has_premium_intent = any(t in text.replace("-", " ").split() for t in premium_terms)

    return {
        "token_count": len(tokens),
        "specificity": clamp(min(len(tokens) / 5.0, 1.0)),
        "budget": budget,
        "price_mode": price_mode,
        "has_price_intent": budget is not None or has_cheap_intent or has_premium_intent,
        "has_cheap_intent": has_cheap_intent,
        "has_premium_intent": has_premium_intent,
    }


# ============================================================================
# V3 PRICE RELEVANCE
# ============================================================================

def calculate_price_relevance(price, query):
    """
    Query-aware price relevance.

    No price intent -> 0.0 so price does not distort ordinary searches.

    Under/below:
        products at or below budget receive strong scores.

    Over/above:
        products at or above target receive strong scores.

    Target:
        closer prices receive higher scores.

    Cheap/premium:
        uses a soft logarithmic signal; this is intentionally weak.
    """

    parsed = detect_advanced_query_intent(query)
    if not parsed["has_price_intent"]:
        return 0.0

    price = safe_float(price, 0.0)
    if price <= 0:
        return 0.0

    budget = parsed["budget"]
    mode = parsed["price_mode"]

    if budget is not None and budget > 0:
        if mode == "under":
            if price <= budget:
                return 1.0
            return clamp(budget / price)

        if mode == "over":
            if price >= budget:
                return 1.0
            return clamp(price / budget)

        if mode == "target":
            ratio = price / budget
            if ratio <= 1:
                return clamp(ratio)
            return clamp(1.0 / ratio)

    if parsed["has_cheap_intent"]:
        return clamp(1.0 / (1.0 + math.log1p(price)))

    if parsed["has_premium_intent"]:
        return clamp(math.log1p(price) / math.log1p(5000))

    return 0.0


# ============================================================================
# V3 MODALITY COVERAGE
# ============================================================================

def calculate_modality_coverage(candidate):
    """
    Measures how much independent retrieval evidence is available.

    0.0 = no useful modality evidence
    0.5 = one useful modality
    1.0 = both modalities have evidence
    """

    text_score = clamp(candidate.get("raw_text_score", 0.0))
    image_score = clamp(candidate.get("raw_image_score", 0.0))

    has_text = text_score > 0
    has_image = image_score > 0

    if has_text and has_image:
        return 1.0
    if has_text or has_image:
        return 0.5
    return 0.0


# ============================================================================
# V3 SCORE INTERACTIONS
# ============================================================================

def calculate_interaction_score(
    semantic_score,
    title_score,
    category_score,
    brand_score,
    multimodal_score,
):
    """
    Reward agreement between independent relevance signals.

    This prevents a product from winning solely because one signal is
    unusually high while the rest of the evidence is poor.
    """

    semantic_score = clamp(semantic_score)
    title_score = clamp(title_score)
    category_score = clamp(category_score)
    brand_score = clamp(brand_score)
    multimodal_score = clamp(multimodal_score)

    semantic_title = math.sqrt(semantic_score * title_score)

    lexical = (
        0.55 * title_score
        + 0.25 * category_score
        + 0.20 * brand_score
    )

    semantic_multimodal = math.sqrt(
        semantic_score * multimodal_score
    )

    return clamp(
        0.45 * semantic_title
        + 0.30 * lexical
        + 0.25 * semantic_multimodal
    )


# ============================================================================
# V3 CONFIDENCE
# ============================================================================

def calculate_ranking_confidence(
    semantic_score,
    title_score,
    category_score,
    brand_score,
    multimodal_score,
    quality_score,
    review_count,
    metadata,
    candidate,
):
    """
    Estimate confidence in the ranking evidence.

    This is not a probability. It is an explainable 0-1 confidence index.
    """

    relevance = (
        0.50 * clamp(semantic_score)
        + 0.25 * clamp(title_score)
        + 0.10 * clamp(category_score)
        + 0.15 * clamp(brand_score)
    )

    modality = calculate_modality_coverage(candidate)

    review_evidence = clamp(
        math.log1p(max(safe_float(review_count), 0.0))
        / math.log1p(1000)
    )

    metadata_fields = [
        metadata.get("title"),
        metadata.get("brand"),
        metadata.get("category"),
        metadata.get("price"),
    ]
    metadata_completeness = sum(
        1 for value in metadata_fields
        if value not in (None, "", [])
    ) / len(metadata_fields)

    return clamp(
        0.45 * relevance
        + 0.20 * modality
        + 0.15 * clamp(quality_score)
        + 0.10 * review_evidence
        + 0.10 * metadata_completeness
    )


# ============================================================================
# V3 METADATA COMPLETENESS
# ============================================================================

def calculate_metadata_completeness(metadata):
    fields = [
        metadata.get("title"),
        metadata.get("brand"),
        metadata.get("category"),
        metadata.get("price"),
    ]

    return clamp(
        sum(
            1
            for value in fields
            if value not in (None, "", [])
        ) / len(fields)
    )


# ============================================================================
# V3 BUSINESS SIGNAL GUARDRAIL
# ============================================================================

def apply_relevance_guardrail(
    semantic_score,
    title_score,
    category_score,
    brand_score,
    quality_score,
    popularity_score,
    verified_score,
):
    """
    Prevent business signals from overpowering relevance.

    A highly rated product should not outrank a clearly relevant product
    merely because it has many reviews.
    """

    relevance = (
        0.55 * clamp(semantic_score)
        + 0.25 * clamp(title_score)
        + 0.10 * clamp(category_score)
        + 0.10 * clamp(brand_score)
    )

    if relevance >= MIN_RELEVANCE_FOR_BUSINESS_SIGNALS:
        return 0.0

    business_strength = (
        0.45 * clamp(quality_score)
        + 0.35 * clamp(popularity_score)
        + 0.20 * clamp(verified_score)
    )

    gap = MIN_RELEVANCE_FOR_BUSINESS_SIGNALS - relevance

    return clamp(
        min(
            MAX_METADATA_PENALTY,
            gap * business_strength,
        )
    )


# ============================================================================
# V3 SINGLE-MODAL PENALTY
# ============================================================================

def calculate_single_modal_penalty(candidate, text_query, image_query):
    """
    Small penalty only when both text and image queries exist but the
    candidate has evidence from just one modality.
    """

    if not text_query or not image_query:
        return 0.0

    modality = str(candidate.get("modality", "")).lower().strip()

    if modality in {"text+image", "text-image", "both"}:
        return 0.0

    coverage = calculate_modality_coverage(candidate)

    if coverage == 0.5:
        return MAX_SINGLE_MODAL_PENALTY

    return 0.0


# ============================================================================
# V3 DIVERSITY
# ============================================================================

def calculate_similarity_for_diversity(current, previous):
    """
    Lightweight lexical similarity used only for final top-K diversity.

    This is intentionally cheap and deterministic; it is not a replacement
    for a semantic embedding model.
    """

    current_tokens = set(
        tokenize_text(current.get("title", ""))
    )
    previous_tokens = set(
        tokenize_text(previous.get("title", ""))
    )

    if not current_tokens or not previous_tokens:
        return 0.0

    intersection = len(current_tokens & previous_tokens)
    union = len(current_tokens | previous_tokens)

    return intersection / union if union else 0.0


def apply_diversity_reranking(results, top_k):
    """
    Greedy Maximal Marginal Relevance-like final selection.

    Keeps relevance dominant while avoiding a top-K filled with nearly
    identical titles/brands.
    """

    if len(results) <= 1:
        return results[:top_k]

    remaining = list(results)
    selected = []

    while remaining and len(selected) < top_k:
        best = None
        best_score = -float("inf")

        for candidate in remaining:
            relevance = safe_float(candidate.get("final_score", 0.0))

            if not selected:
                adjusted = relevance
            else:
                max_similarity = max(
                    calculate_similarity_for_diversity(
                        candidate,
                        previous,
                    )
                    for previous in selected
                )

                brand_similarity = 0.0
                candidate_brand = str(
                    candidate.get("brand", "")
                ).strip().lower()

                if candidate_brand:
                    for previous in selected:
                        previous_brand = str(
                            previous.get("brand", "")
                        ).strip().lower()
                        if previous_brand == candidate_brand:
                            brand_similarity = max(
                                brand_similarity,
                                0.25,
                            )

                penalty = min(
                    MAX_DIVERSITY_PENALTY,
                    0.06 * max_similarity + brand_similarity * 0.02,
                )

                adjusted = relevance - penalty

            if adjusted > best_score:
                best_score = adjusted
                best = candidate

        selected.append(best)
        remaining.remove(best)

    return selected


# ============================================================================
# V3 WEIGHTS
# ============================================================================

def calculate_v3_dynamic_weights(
    text_query,
    title_score,
    category_score,
    brand_score,
    modality="",
    image_query=None,
):
    weights = {
        "semantic": SEMANTIC_WEIGHT,
        "title": TITLE_WEIGHT,
        "category": CATEGORY_WEIGHT,
        "brand": BRAND_WEIGHT,
        "multimodal": MULTIMODAL_WEIGHT,
        "quality": QUALITY_WEIGHT,
        "popularity": POPULARITY_WEIGHT,
        "verified": VERIFIED_WEIGHT,
        "price": PRICE_WEIGHT,
        "interaction": INTERACTION_WEIGHT,
    }

    normalized_modality = str(modality or "").lower().strip()
    has_text = bool(str(text_query or "").strip())
    has_image = bool(image_query)

    if not has_text:
        shift = weights["title"] + weights["category"] + weights["brand"]
        weights["title"] = 0.0
        weights["category"] = 0.0
        weights["brand"] = 0.0
        weights["semantic"] += shift

    if has_text and title_score >= 0.85:
        shift = 0.035
        weights["title"] += shift
        weights["semantic"] -= shift

    if has_text and category_score >= 0.80:
        shift = 0.015
        weights["category"] += shift
        weights["semantic"] -= shift

    if has_text and brand_score >= 0.80:
        shift = 0.015
        weights["brand"] += shift
        weights["semantic"] -= shift

    if has_text and has_image and normalized_modality in {
        "text+image", "text-image", "both"
    }:
        shift = 0.025
        weights["multimodal"] += shift
        weights["semantic"] -= shift

    if not has_text and not has_image:
        weights["semantic"] = 1.0
        for key in weights:
            if key != "semantic":
                weights[key] = 0.0

    weights["semantic"] = max(0.0, weights["semantic"])

    total = sum(weights.values())

    if total <= 0:
        return weights

    return {
        key: value / total
        for key, value in weights.items()
    }


# ============================================================================
# V3 FINAL SCORE
# ============================================================================

def calculate_v3_final_score(
    semantic_score,
    title_score,
    category_score,
    brand_score,
    multimodal_score,
    quality_score,
    popularity_score,
    verified_score,
    price_score,
    interaction_score,
    confidence_score,
    weights,
    guardrail_penalty=0.0,
    single_modal_penalty=0.0,
    metadata_penalty=0.0,
):
    contributions = {
        "semantic": weights["semantic"] * semantic_score,
        "title": weights["title"] * title_score,
        "category": weights["category"] * category_score,
        "brand": weights["brand"] * brand_score,
        "multimodal": weights["multimodal"] * multimodal_score,
        "quality": weights["quality"] * quality_score,
        "popularity": weights["popularity"] * popularity_score,
        "verified": weights["verified"] * verified_score,
        "price": weights["price"] * price_score,
        "interaction": weights["interaction"] * interaction_score,
    }

    base_score = sum(contributions.values())

    final_score = clamp(
        base_score
        + 0.02 * clamp(confidence_score)
        - guardrail_penalty
        - single_modal_penalty
        - metadata_penalty
    )

    contributions["confidence_bonus"] = 0.02 * clamp(confidence_score)
    contributions["guardrail_penalty"] = -guardrail_penalty
    contributions["single_modal_penalty"] = -single_modal_penalty
    contributions["metadata_penalty"] = -metadata_penalty

    return final_score, contributions


# ============================================================================
# FINAL SCORE
# ============================================================================

def calculate_final_score(
    semantic_score,
    title_score,
    category_score,
    brand_score,
    multimodal_score,
    quality_score,
    popularity_score,
    verified_score,
    price_score,
    weights,
):
    contributions = {
        "semantic": weights["semantic"] * semantic_score,
        "title": weights["title"] * title_score,
        "category": weights["category"] * category_score,
        "brand": weights["brand"] * brand_score,
        "multimodal": weights["multimodal"] * multimodal_score,
        "quality": weights["quality"] * quality_score,
        "popularity": weights["popularity"] * popularity_score,
        "verified": weights["verified"] * verified_score,
        "price": weights["price"] * price_score,
    }

    return clamp(sum(contributions.values())), contributions


# ============================================================================
# CANDIDATE IDENTITY + DUPLICATE MERGING
# ============================================================================

def get_candidate_identity(candidate):
    product_id = candidate.get(
        "product_id",
        candidate.get("canonical_product_id"),
    )
    asin = candidate.get("asin")

    if product_id is not None:
        return ("product_id", str(product_id))
    if asin is not None:
        return ("asin", str(asin))
    return None


def get_candidate_keys(candidates):
    product_ids = set()
    asins = set()

    for candidate in candidates:
        product_id = candidate.get(
            "product_id",
            candidate.get("canonical_product_id"),
        )
        asin = candidate.get("asin")

        if product_id is not None:
            product_ids.add(str(product_id))
        if asin is not None:
            asins.add(str(asin))

    return product_ids, asins


def merge_duplicate_candidates(candidates):
    """
    IMPORTANT FIX:

    The old deduplication calculated semantic score without passing the query,
    which effectively returned 0 for every duplicate. Therefore, when the same
    product appeared in text and image retrieval, the first occurrence could
    win and the other modality evidence could be lost.

    This version merges the evidence instead:
        text score = max(text evidence)
        image score = max(image evidence)
        modality = derived from available evidence
    """

    merged = {}
    anonymous = []

    for candidate in candidates:
        identity = get_candidate_identity(candidate)

        if identity is None:
            anonymous.append(dict(candidate))
            continue

        current = merged.get(identity)

        if current is None:
            merged[identity] = dict(candidate)
            continue

        for score_key in ("raw_text_score", "raw_image_score"):
            current_value = safe_float(current.get(score_key, 0.0))
            incoming_value = safe_float(candidate.get(score_key, 0.0))

            if incoming_value > current_value:
                current[score_key] = candidate[score_key]

        # Preserve useful fields from either candidate.
        for key, value in candidate.items():
            if key not in current or current[key] in (None, "", []):
                current[key] = value

        text_score = safe_float(current.get("raw_text_score", 0.0))
        image_score = safe_float(current.get("raw_image_score", 0.0))

        if text_score > 0 and image_score > 0:
            current["modality"] = "text+image"
        elif text_score > 0:
            current["modality"] = "text-only"
        elif image_score > 0:
            current["modality"] = "image-only"

    return list(merged.values()) + anonymous


def deduplicate_candidates(candidates):
    return merge_duplicate_candidates(candidates)


# ============================================================================
# METADATA LOADING
# ============================================================================

def load_product_metadata_for_candidates(
    candidate_product_ids,
    candidate_asins,
    batch_size=METADATA_BATCH_SIZE,
):
    print("\nLoading product metadata...")

    if not PRODUCT_SUMMARY.exists():
        raise FileNotFoundError(
            f"Product summary not found:\n{PRODUCT_SUMMARY}"
        )

    parquet_file = pq.ParquetFile(PRODUCT_SUMMARY)
    total_rows = parquet_file.metadata.num_rows

    print(f"Product metadata rows: {total_rows:,}")

    product_lookup = {}
    asin_lookup = {}
    total_processed = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        product_id_column = find_column(
            batch, ["canonical_product_id", "product_id"]
        )
        asin_column = find_column(batch, ["asin"])
        title_column = find_column(batch, ["title", "product_title"])
        brand_column = find_column(batch, ["brand"])
        category_column = find_column(
            batch, ["main_category", "main_cat", "category"]
        )
        price_column = find_column(batch, ["price"])

        if product_id_column is None:
            raise ValueError(
                "Product summary must contain 'product_id' "
                "or 'canonical_product_id'."
            )

        row_count = batch.num_rows

        product_ids = batch[product_id_column].to_pylist()
        asins = batch[asin_column].to_pylist() if asin_column else [None] * row_count
        titles = batch[title_column].to_pylist() if title_column else [None] * row_count
        brands = batch[brand_column].to_pylist() if brand_column else [None] * row_count
        categories = batch[category_column].to_pylist() if category_column else [None] * row_count
        prices = batch[price_column].to_pylist() if price_column else [None] * row_count

        for product_id, asin, title, brand, category, price in zip(
            product_ids, asins, titles, brands, categories, prices
        ):
            if product_id is None:
                continue

            product_id_str = str(product_id)
            asin_str = str(asin) if asin is not None else None

            if (
                product_id_str not in candidate_product_ids
                and (
                    asin_str is None
                    or asin_str not in candidate_asins
                )
            ):
                continue

            metadata = {
                "product_id": product_id,
                "asin": asin,
                "title": title,
                "brand": brand,
                "category": category,
                "price": price,
            }

            product_lookup[("product_id", product_id_str)] = metadata

            if asin_str:
                asin_lookup[("asin", asin_str)] = metadata

        total_processed += row_count

        if total_processed % (batch_size * 10) == 0:
            print(
                f"Scanning metadata: "
                f"{min(total_processed, total_rows):,}/{total_rows:,}"
                f" | Product matches: {len(product_lookup):,}"
                f" | ASIN matches: {len(asin_lookup):,}"
            )

        if (
            candidate_product_ids
            and len(product_lookup) >= len(candidate_product_ids)
            and (
                not candidate_asins
                or len(asin_lookup) >= len(candidate_asins)
            )
        ):
            break

    print(
        f"Product metadata matched by ID: "
        f"{len(product_lookup):,}/{len(candidate_product_ids):,}"
    )
    print(
        f"Product metadata matched by ASIN: "
        f"{len(asin_lookup):,}/{len(candidate_asins):,}"
    )

    return {"product_id": product_lookup, "asin": asin_lookup}


def load_review_stats_for_candidates(
    candidate_product_ids,
    candidate_asins,
    batch_size=METADATA_BATCH_SIZE,
):
    print("\nLoading review statistics...")

    if not REVIEW_STATS.exists():
        print("[WARNING] Review statistics file not found.")
        return {}

    parquet_file = pq.ParquetFile(REVIEW_STATS)
    total_rows = parquet_file.metadata.num_rows

    print(f"Review statistics rows: {total_rows:,}")

    review_lookup = {}

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        product_id_column = find_column(
            batch, ["canonical_product_id", "product_id"]
        )
        asin_column = find_column(batch, ["asin"])

        review_count_column = find_column(
            batch,
            ["review_count", "num_reviews", "total_reviews", "reviews_count"],
        )
        rating_column = find_column(
            batch,
            ["average_rating", "avg_rating", "rating", "mean_rating"],
        )
        verified_column = find_column(
            batch,
            ["verified_ratio", "verified_review_ratio", "verified_ratio_pct"],
        )

        row_count = batch.num_rows

        product_ids = (
            batch[product_id_column].to_pylist()
            if product_id_column else [None] * row_count
        )
        asins = (
            batch[asin_column].to_pylist()
            if asin_column else [None] * row_count
        )
        review_counts = (
            batch[review_count_column].to_pylist()
            if review_count_column else [0] * row_count
        )
        ratings = (
            batch[rating_column].to_pylist()
            if rating_column else [0] * row_count
        )
        verified_ratios = (
            batch[verified_column].to_pylist()
            if verified_column else [0] * row_count
        )

        for product_id, asin, review_count, rating, verified_ratio in zip(
            product_ids, asins, review_counts, ratings, verified_ratios
        ):
            record = {
                "review_count": safe_float(review_count),
                "rating": safe_float(rating),
                "verified_ratio": safe_float(verified_ratio),
            }

            product_id_str = (
                str(product_id) if product_id is not None else None
            )
            asin_str = str(asin) if asin is not None else None

            if product_id_str and product_id_str in candidate_product_ids:
                review_lookup[("product_id", product_id_str)] = record

            if asin_str and asin_str in candidate_asins:
                review_lookup[("asin", asin_str)] = record

    print(f"Review statistics matched: {len(review_lookup):,}")
    return review_lookup


# ============================================================================
# RESOLUTION
# ============================================================================

def resolve_product_metadata(product_id, asin, product_lookup):
    if product_id:
        result = product_lookup.get("product_id", {}).get(
            ("product_id", str(product_id)), {}
        )
        if result:
            return result

    if asin:
        return product_lookup.get("asin", {}).get(
            ("asin", str(asin)), {}
        )

    return {}


def resolve_review_stats(product_id, asin, review_lookup):
    if not review_lookup:
        return {}

    if product_id:
        result = review_lookup.get(("product_id", str(product_id)))
        if result is not None:
            return result

    if asin:
        result = review_lookup.get(("asin", str(asin)))
        if result is not None:
            return result

    return {}


# ============================================================================
# RERANK
# ============================================================================


def rerank_products_v3(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
    text_query="",
    image_query=None,
    debug=False,
):
    """
    Full V3 business-aware multimodal reranker.

    Stages:
        1. Merge/deduplicate candidates
        2. Normalize retrieval evidence
        3. Extract lexical/business features
        4. Compute interactions and confidence
        5. Apply relevance guardrails
        6. Apply diversity-aware final selection
    """

    candidates = deduplicate_candidates(candidates)

    # Normalize retrieval scores once before calculating any V3 score.
    # This also guarantees normalized_text_score / normalized_image_score
    # are present in the final output and prevents KeyError in print_results.
    candidates = normalize_candidate_scores(candidates)

    results = []

    for candidate in candidates:
        product_id = candidate.get(
            "product_id",
            candidate.get("canonical_product_id", ""),
        )
        product_id = str(product_id) if product_id is not None else ""

        asin = candidate.get("asin")

        semantic_score = calculate_query_aware_semantic_score(
            candidate=candidate,
            text_query=text_query,
            image_query=image_query,
        )

        metadata = resolve_product_metadata(
            product_id=product_id,
            asin=asin,
            product_lookup=product_lookup,
        )

        if not metadata and debug:
            print(
                f"[WARNING] Metadata missing: "
                f"product_id={product_id}, asin={asin}"
            )

        if asin is None:
            asin = metadata.get("asin")

        title = metadata.get("title", "")
        brand = metadata.get("brand", "")
        category = metadata.get("category", "")
        price = metadata.get("price")

        title_score = calculate_title_relevance(text_query, title)
        category_score = calculate_category_relevance(text_query, category)
        brand_score = calculate_brand_relevance(text_query, brand)

        review_stats = resolve_review_stats(
            product_id=product_id,
            asin=asin,
            review_lookup=review_lookup,
        ) or {
            "review_count": 0,
            "rating": 0,
            "verified_ratio": 0,
        }

        review_count = safe_float(review_stats.get("review_count", 0))
        rating = safe_float(review_stats.get("rating", 0))
        verified_ratio = safe_float(review_stats.get("verified_ratio", 0))

        rating_score = normalize_rating(rating)
        quality_score = calculate_quality_score(rating, review_count)
        popularity_score = normalize_popularity(review_count)
        verified_score = normalize_verified_ratio(verified_ratio)

        multimodal_score = calculate_multimodal_agreement(candidate)
        modality_coverage = calculate_modality_coverage(candidate)

        price_score = calculate_price_relevance(text_query, price)

        interaction_score = calculate_interaction_score(
            semantic_score,
            title_score,
            category_score,
            brand_score,
            multimodal_score,
        )

        confidence_score = calculate_ranking_confidence(
            semantic_score=semantic_score,
            title_score=title_score,
            category_score=category_score,
            brand_score=brand_score,
            multimodal_score=multimodal_score,
            quality_score=quality_score,
            review_count=review_count,
            metadata=metadata,
            candidate=candidate,
        )

        metadata_completeness = calculate_metadata_completeness(metadata)

        guardrail_penalty = apply_relevance_guardrail(
            semantic_score=semantic_score,
            title_score=title_score,
            category_score=category_score,
            brand_score=brand_score,
            quality_score=quality_score,
            popularity_score=popularity_score,
            verified_score=verified_score,
        )

        single_modal_penalty = calculate_single_modal_penalty(
            candidate=candidate,
            text_query=text_query,
            image_query=image_query,
        )

        metadata_penalty = (
            (1.0 - metadata_completeness) * 0.03
            if semantic_score >= MIN_RELEVANCE_FOR_BUSINESS_SIGNALS
            else 0.0
        )

        weights = calculate_v3_dynamic_weights(
            text_query=text_query,
            title_score=title_score,
            category_score=category_score,
            brand_score=brand_score,
            modality=candidate.get("modality", ""),
            image_query=image_query,
        )

        final_score, contributions = calculate_v3_final_score(
            semantic_score=semantic_score,
            title_score=title_score,
            category_score=category_score,
            brand_score=brand_score,
            multimodal_score=multimodal_score,
            quality_score=quality_score,
            popularity_score=popularity_score,
            verified_score=verified_score,
            price_score=price_score,
            interaction_score=interaction_score,
            confidence_score=confidence_score,
            weights=weights,
            guardrail_penalty=guardrail_penalty,
            single_modal_penalty=single_modal_penalty,
            metadata_penalty=metadata_penalty,
        )

        result = {
            **candidate,
            "product_id": product_id,
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,

            "semantic_score": semantic_score,
            "title_relevance_score": title_score,
            "category_relevance_score": category_score,
            "brand_relevance_score": brand_score,
            "multimodal_score": multimodal_score,
            "modality_coverage": modality_coverage,

            "rating": rating,
            "review_count": review_count,
            "verified_ratio": verified_ratio,
            "rating_score": rating_score,
            "quality_score": quality_score,
            "popularity_score": popularity_score,
            "verified_score": verified_score,

            "price_relevance_score": price_score,
            "interaction_score": interaction_score,
            "ranking_confidence": confidence_score,
            "metadata_completeness": metadata_completeness,

            "guardrail_penalty": guardrail_penalty,
            "single_modal_penalty": single_modal_penalty,
            "metadata_missing_penalty": metadata_penalty,

            "semantic_weight": weights["semantic"],
            "title_weight": weights["title"],
            "category_weight": weights["category"],
            "brand_weight": weights["brand"],
            "multimodal_weight": weights["multimodal"],
            "quality_weight": weights["quality"],
            "popularity_weight": weights["popularity"],
            "verified_weight": weights["verified"],
            "price_weight": weights["price"],
            "interaction_weight": weights["interaction"],

            "semantic_contribution": contributions["semantic"],
            "title_relevance_contribution": contributions["title"],
            "category_relevance_contribution": contributions["category"],
            "brand_relevance_contribution": contributions["brand"],
            "multimodal_contribution": contributions["multimodal"],
            "quality_contribution": contributions["quality"],
            "popularity_contribution": contributions["popularity"],
            "verified_contribution": contributions["verified"],
            "price_contribution": contributions["price"],
            "interaction_contribution": contributions["interaction"],
            "confidence_bonus": contributions["confidence_bonus"],

            "final_score_before_diversity": final_score,
            "final_score": final_score,

            "ranking_modality": candidate.get("modality", ""),
            "raw_text_score": safe_float(candidate.get("raw_text_score", 0.0)),
            "raw_image_score": safe_float(candidate.get("raw_image_score", 0.0)),
            "normalized_text_score": safe_float(candidate.get("normalized_text_score", 0.0)),
            "normalized_image_score": safe_float(candidate.get("normalized_image_score", 0.0)),
        }

        # Human-readable explanation.
        reasons = []

        if semantic_score >= 0.80:
            reasons.append("strong semantic match")
        elif semantic_score >= 0.60:
            reasons.append("good semantic match")

        if title_score >= 0.80:
            reasons.append("strong title match")

        if multimodal_score >= 0.75:
            reasons.append("strong text-image agreement")

        if quality_score >= 0.80 and review_count >= 5:
            reasons.append("reliable product quality")

        if price_score >= 0.80:
            reasons.append("strong price fit")

        if confidence_score >= 0.75:
            reasons.append("high ranking confidence")

        if not reasons:
            reasons.append("combined retrieval and product signals")

        result["ranking_explanation"] = {
            "primary_reason": reasons[0],
            "reasons": reasons,
        }

        results.append(result)

    # Initial relevance ordering.
    results.sort(
        key=lambda item: (
            safe_float(item.get("final_score", 0.0)),
            safe_float(item.get("semantic_score", 0.0)),
            safe_float(item.get("title_relevance_score", 0.0)),
            safe_float(item.get("quality_score", 0.0)),
        ),
        reverse=True,
    )

    # Diversity-aware final top-K.
    selected = apply_diversity_reranking(
        results,
        top_k=top_k,
    )

    # Assign final rank and record diversity impact.
    for rank, result in enumerate(selected, start=1):
        result["rank"] = rank
        result["diversity_adjusted"] = (
            result["final_score"]
            != result["final_score_before_diversity"]
        )

    return selected


def rerank_products(
    candidates,
    product_lookup,
    review_lookup,
    top_k,
    text_query="",
    image_query=None,
    debug=False,
):
    # 1. Merge duplicate product evidence.
    candidates = merge_duplicate_candidates(candidates)

    # 2. Normalize vector retrieval score distributions.
    candidates = normalize_candidate_scores(candidates)

    results = []

    for input_order, candidate in enumerate(candidates):
        product_id = candidate.get(
            "product_id",
            candidate.get("canonical_product_id", ""),
        )
        product_id = str(product_id) if product_id is not None else ""

        asin = candidate.get("asin")

        metadata = resolve_product_metadata(
            product_id,
            asin,
            product_lookup,
        )

        if not metadata and debug:
            print(
                f"[WARNING] Metadata missing: "
                f"product_id={product_id}, asin={asin}"
            )

        if asin is None:
            asin = metadata.get("asin")

        title = metadata.get("title", "")
        brand = metadata.get("brand", "")
        category = metadata.get("category", "")
        price = metadata.get("price")

        semantic_score = calculate_query_aware_semantic_score(
            candidate,
            text_query,
            image_query,
        )

        title_score = calculate_title_relevance(
            text_query,
            title,
        )

        category_score = calculate_category_relevance(
            text_query,
            category,
        )

        brand_score = calculate_brand_relevance(
            text_query,
            brand,
        )

        price_score = calculate_price_relevance(
            text_query,
            price,
        )

        review_stats = resolve_review_stats(
            product_id,
            asin,
            review_lookup,
        ) or {
            "review_count": 0,
            "rating": 0,
            "verified_ratio": 0,
        }

        review_count = safe_float(review_stats.get("review_count", 0))
        rating = safe_float(review_stats.get("rating", 0))
        verified_ratio = safe_float(
            review_stats.get("verified_ratio", 0)
        )

        rating_score = normalize_rating(rating)
        quality_score = calculate_quality_score(
            rating,
            review_count,
        )
        popularity_score = normalize_popularity(review_count)
        verified_score = normalize_verified_ratio(
            verified_ratio
        )

        multimodal_score = calculate_multimodal_agreement(candidate)
        modality_coverage = calculate_modality_coverage(candidate)

        weights = calculate_dynamic_weights(
            text_query=text_query,
            title_score=title_score,
            category_score=category_score,
            brand_score=brand_score,
            price_score=price_score,
            modality=candidate.get("modality", ""),
        )

        final_score, contributions = calculate_final_score(
            semantic_score,
            title_score,
            category_score,
            brand_score,
            multimodal_score,
            quality_score,
            popularity_score,
            verified_score,
            price_score,
            weights,
        )

        query_boost = calculate_query_boost(
            text_query,
            title,
            brand,
            category,
        )
        final_score = clamp(final_score + query_boost)

        relevance_anchor = max(
            semantic_score,
            title_score,
            0.75 * category_score,
            0.75 * brand_score,
        )

        guardrail_penalty = 0.0
        if (
            text_query
            and relevance_anchor < WEAK_RELEVANCE_THRESHOLD
        ):
            guardrail_penalty = WEAK_RELEVANCE_PENALTY
            final_score = clamp(final_score - guardrail_penalty)

        metadata_penalty = (
            MISSING_METADATA_PENALTY
            if not metadata
            else 0.0
        )

        if metadata_penalty:
            final_score = clamp(final_score - metadata_penalty)

        modality_penalty = 0.0
        if (
            text_query
            and image_query
            and modality_coverage < 1.0
        ):
            modality_penalty = SINGLE_MODALITY_PENALTY
            final_score = clamp(final_score - modality_penalty)

        result = {
            **candidate,

            "rank": 0,
            "product_id": product_id,
            "asin": asin,
            "title": title,
            "brand": brand,
            "category": category,
            "price": price,

            "semantic_score": semantic_score,
            "title_relevance_score": title_score,
            "category_relevance_score": category_score,
            "brand_relevance_score": brand_score,
            "price_relevance_score": price_score,
            "multimodal_score": multimodal_score,
            "modality_coverage": modality_coverage,

            "rating": rating,
            "review_count": review_count,
            "verified_ratio": verified_ratio,

            "rating_score": rating_score,
            "quality_score": quality_score,
            "popularity_score": popularity_score,
            "verified_score": verified_score,

            "semantic_weight": weights["semantic"],
            "title_weight": weights["title"],
            "category_weight": weights["category"],
            "brand_weight": weights["brand"],
            "multimodal_weight": weights["multimodal"],
            "quality_weight": weights["quality"],
            "popularity_weight": weights["popularity"],
            "verified_weight": weights["verified"],
            "price_weight": weights["price"],

            "semantic_contribution": contributions["semantic"],
            "title_relevance_contribution": contributions["title"],
            "category_relevance_contribution": contributions["category"],
            "brand_relevance_contribution": contributions["brand"],
            "multimodal_contribution": contributions["multimodal"],
            "quality_contribution": contributions["quality"],
            "popularity_contribution": contributions["popularity"],
            "verified_contribution": contributions["verified"],
            "price_contribution": contributions["price"],

            "query_boost": query_boost,
            "guardrail_penalty": guardrail_penalty,
            "metadata_missing_penalty": metadata_penalty,
            "single_modality_penalty": modality_penalty,

            "final_score": final_score,

            "ranking_modality": candidate.get("modality", ""),

            # Keep original retrieval scores for auditability.
            "raw_text_score": safe_float(
                candidate.get("raw_text_score", 0.0)
            ),
            "raw_image_score": safe_float(
                candidate.get("raw_image_score", 0.0)
            ),
            "normalized_text_score": safe_float(
                candidate.get("normalized_text_score", 0.0)
            ),
            "normalized_image_score": safe_float(
                candidate.get("normalized_image_score", 0.0)
            ),

            "_input_order": input_order,
        }

        results.append(result)

    # Deterministic tie-breaking.
    results.sort(
        key=lambda item: (
            safe_float(item.get("final_score", 0.0)),
            safe_float(item.get("title_relevance_score", 0.0)),
            safe_float(item.get("semantic_score", 0.0)),
            safe_float(item.get("multimodal_score", 0.0)),
            safe_float(item.get("quality_score", 0.0)),
            safe_float(item.get("review_count", 0.0)),
            -safe_float(item.get("_input_order", 0)),
        ),
        reverse=True,
    )

    results = results[:top_k]

    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
        result.pop("_input_order", None)

    return results


# ============================================================================
# LOAD / SAVE
# ============================================================================

def load_candidates(candidate_path):
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{candidate_path}"
        )

    with open(candidate_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    text_query = ""
    image_query = None

    if isinstance(data, dict):
        candidates = data.get("candidates")
        query = data.get("query", {})

        if isinstance(query, dict):
            text_query = query.get("text", "")
            image_query = query.get("image")

    elif isinstance(data, list):
        candidates = data

    else:
        raise ValueError("Invalid candidates JSON format.")

    if not isinstance(candidates, list):
        raise ValueError("'candidates' must be a list.")

    return candidates, text_query, image_query


def print_results(results):
    print("\n" + "=" * 110)
    print("FINAL PRODUCT RESULTS")
    print("=" * 110)
    print(f"Displaying top {len(results)} results")

    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 110)
        print(f"#{index}")
        print(f"Product ID:        {result.get('product_id')}")
        print(f"ASIN:              {result.get('asin')}")
        print(f"Title:             {result.get('title')}")
        print(f"Brand:             {result.get('brand')}")
        print(f"Category:          {result.get('category')}")
        print(f"Price:             {result.get('price')}")
        print(f"Modality:          {result.get('modality')}")

        print("\nRANKING SCORES")
        for label, key in (
            ("Final score", "final_score"),
            ("Semantic score", "semantic_score"),
            ("Title relevance", "title_relevance_score"),
            ("Category relevance", "category_relevance_score"),
            ("Brand relevance", "brand_relevance_score"),
            ("Price relevance", "price_relevance_score"),
            ("Multimodal score", "multimodal_score"),
            ("Quality score", "quality_score"),
            ("Popularity score", "popularity_score"),
            ("Verified score", "verified_score"),
        ):
            print(
                f"{label + ':':20s} "
                f"{safe_float(result.get(key, 0.0)):.6f}"
            )

        print("\nPRODUCT QUALITY")

        print(
            f"Rating:            "
            f"{safe_float(result.get('rating', 0.0)):.2f}"
        )

        print(
            f"Reviews:           "
            f"{safe_float(result.get('review_count', 0.0)):,.0f}"
        )

        print(
            f"Verified ratio:    "
            f"{safe_float(result.get('verified_ratio', 0.0)):.2%}"
        )

        print("\nRETRIEVAL EVIDENCE")

        print(
            f"Raw text score:    "
            f"{safe_float(result.get('raw_text_score', 0.0)):.6f}"
        )

        print(
            f"Raw image score:   "
            f"{safe_float(result.get('raw_image_score', 0.0)):.6f}"
        )

        print(
            f"Normalized text:   "
            f"{safe_float(result.get('normalized_text_score', 0.0)):.6f}"
        )

        print(
            f"Normalized image:  "
            f"{safe_float(result.get('normalized_image_score', 0.0)):.6f}"
        )

        print(
            f"Modality coverage: "
            f"{safe_float(result.get('modality_coverage', 0.0)):.2f}"
        )

        print("\nSCORE ADJUSTMENTS")

        print(
            f"Query boost:       "
            f"{safe_float(result.get('query_boost', 0.0)):.6f}"
        )

        print(
            f"Guardrail penalty: "
            f"{safe_float(result.get('guardrail_penalty', 0.0)):.6f}"
        )

        print(
            f"Metadata penalty:  "
            f"{safe_float(result.get('metadata_missing_penalty', 0.0)):.6f}"
        )

        print(
            f"Modality penalty:  "
            f"{safe_float(result.get('single_modality_penalty', 0.0)):.6f}"
        )


def save_results(results, output_path, text_query, image_query):
    payload = {
        "query": {
            "text": text_query,
            "image": image_query,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved final results to:\n{output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Product-level multimodal business-aware reranking V3."
        )
    )

    parser.add_argument(
        "--candidates",
        default="candidates.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--output",
        default="product_results.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=METADATA_BATCH_SIZE,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    print("=" * 110)
    print("PRODUCT RERANKING V3")
    print("=" * 110)
    print(f"Candidates:  {args.candidates}")
    print(f"Top-K:       {args.top_k}")
    print(f"Output:      {args.output}")
    print(f"Batch size:  {args.batch_size}")

    candidate_path = Path(args.candidates)

    candidates, text_query, image_query = load_candidates(
        candidate_path
    )

    print(f"\nText query:  {text_query}")
    print(f"Image query: {image_query}")
    print(f"Candidates:  {len(candidates):,}")

    if not candidates:
        print("No candidates to rerank.")
        return

    candidate_product_ids, candidate_asins = get_candidate_keys(
        candidates
    )

    print(
        f"\nCandidate product IDs: {len(candidate_product_ids):,}"
    )
    print(
        f"Candidate ASINs:       {len(candidate_asins):,}"
    )

    intent = detect_query_intent(text_query)

    print("\nQUERY INTENT")
    print(f"Tokens:       {intent['token_count']}")
    print(f"Specific:     {intent['is_specific']}")
    print(f"Price intent: {intent['has_price_intent']}")

    price_constraint = extract_price_constraint(text_query)
    if price_constraint:
        print(
            f"Price rule:   "
            f"{price_constraint['type']} "
            f"{price_constraint['value']}"
        )

    product_lookup = load_product_metadata_for_candidates(
        candidate_product_ids,
        candidate_asins,
        batch_size=args.batch_size,
    )

    review_lookup = load_review_stats_for_candidates(
        candidate_product_ids,
        candidate_asins,
        batch_size=args.batch_size,
    )

    print("\nReranking products...")

    results = rerank_products_v3(
        candidates=candidates,
        product_lookup=product_lookup,
        review_lookup=review_lookup,
        top_k=args.top_k,
        text_query=text_query,
        image_query=image_query,
        debug=args.debug,
    )

    print_results(results)

    save_results(
        results=results,
        output_path=Path(args.output),
        text_query=text_query,
        image_query=image_query,
    )

    print("\n" + "=" * 110)
    print("PRODUCT RERANKING V3 COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
