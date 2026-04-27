"""
ID Embedding Pipeline for V3 Click Data
========================================

OVERVIEW
--------
This script processes user event sequences to create ID embeddings for training
recommendation models. It handles multiple event domains (ads, shopping, web 
browsing) and creates train/eval splits with proper sequence handling.

The pipeline is modular - each stage can be run independently if its input data
exists, allowing for flexible execution and debugging.


PIPELINE STAGES
---------------

STAGE 1: Prepare All Events
    Purpose:  Load raw events, parse JSON, filter invalid data, deduplicate
    Input:    INPUT_PATH/*.parquet (raw event data)
    Output:   OUTPUT_PATH/all_events (cleaned parquet)
    Actions:  - Parse events_parsed array
              - Extract title/url from texts
              - Filter events with time > 2010-01-01
              - Remove duplicate events
              - Handle NULL values with placeholders

STAGE 2: Create Domain Vocabularies
    Purpose:  Build vocabulary indices for each domain (ads, shopping, web)
    Input:    OUTPUT_PATH/all_events
    Outputs:  OUTPUT_PATH/ads_vocab (ads domain items with indices)
              OUTPUT_PATH/shopping_vocab (shopping domain items with indices)
              OUTPUT_PATH/web_browsing_vocab (web domain items with indices)
    Actions:  - Extract unique (title, url) pairs per domain
              - Assign sequential indices to each item
              - Configurable starting indices per domain
              - Outputs include index ranges and vocab sizes

STAGE 3: Join Events with IDs
    Purpose:  Map each event to its corresponding vocabulary ID and encode
    Input:    OUTPUT_PATH/all_events
              OUTPUT_PATH/ads_vocab
              OUTPUT_PATH/shopping_vocab
              OUTPUT_PATH/web_browsing_vocab
    Output:   OUTPUT_PATH/all_events_with_id
    Actions:  - Assign domain_id (0=ads, 1=web, 2=shopping) to each event
              - Join with respective vocabulary to get item_id
              - Calculate domain offsets dynamically based on vocab sizes
              - Create encoded_id = offset + item_id (unique across domains)
              - Reports vocab sizes and offset calculations

STAGE 4: Create Event Sequences
    Purpose:  Group events by user and create time-ordered sequences
    Input:    OUTPUT_PATH/all_events_with_id
    Output:   OUTPUT_PATH/events_seq
    Actions:  - Sort events by user_id and timestamp
              - Group into arrays per user (encoded_ids, titles, urls, etc.)
              - Create comprehensive user event sequences
              - Includes timestamps, types, markets, domain info

STAGE 5: Create Train/Eval Split
    Purpose:  Split sequences into training and evaluation datasets
    Input:    OUTPUT_PATH/events_seq
    Outputs:  OUTPUT_PATH/train (training sequences)
              OUTPUT_PATH/eval (evaluation sequences)
    Actions:  - Filter sequences with length >= MIN_SEQUENCE_LENGTH
              - Split based on ads-positive events (clicks/conversions)
              - Users with ads-positive: 95% train, 5% eval
              - Users without ads-positive: all go to train
              - Eval sequences truncated to last ads-positive event
              - Ensures eval has minimum sequence length after truncation


USAGE EXAMPLES
--------------

1. Run complete pipeline (all stages):
   $ python ids_creation_pipeline.py

2. Run only a specific stage:
   $ python ids_creation_pipeline.py --only-stage 1
   $ python ids_creation_pipeline.py --only-stage 3

3. Run from a specific stage to end:
   $ python ids_creation_pipeline.py --start-stage 3

4. Run a range of stages:
   $ python ids_creation_pipeline.py --start-stage 2 --end-stage 4

5. Force rerun even if outputs exist:
   $ python ids_creation_pipeline.py --only-stage 3 --force-rerun

6. Check what would run (dry-run not implemented, but outputs are checked):
   Stages automatically skip if output already exists unless --force-rerun


COMMAND LINE ARGUMENTS
----------------------
--start-stage N      Start from stage N (1-5)
--end-stage N        End at stage N (1-5), runs to completion if not specified
--only-stage N       Run only stage N (shorthand for --start-stage N --end-stage N)
--force-rerun        Rerun stages even if outputs already exist


CONFIGURATION
-------------
Edit these constants at the top of the file:

PREFIX_PATH                  (removed — replaced by INPUT_PATH and OUTPUT_PATH)
INPUT_PATH                   Base path for raw input *.parquet event files
OUTPUT_PATH                  Base path for all pipeline outputs (intermediates + finals)
                             Default: same as INPUT_PATH; override to write elsewhere
ADS_TYPES                    Event types for ads domain
WEB_BROWSING_TYPES          Event types for web domain
SHOPPING_TYPES              Event types for shopping domain
ADS_POSITIVE_TYPES          Positive events for ads (clicks, conversions)
DOMAIN_OFFSET_PATH          Path where the computed domain_offset is persisted
ADS_VOCAB_START_INDEX       Starting index for ads vocab (default: 20)
WEB_VOCAB_START_INDEX       Starting index for web vocab (default: 0)
SHOPPING_VOCAB_START_INDEX  Starting index for shopping vocab (default: 0)
MIN_SEQUENCE_LENGTH         Minimum events per sequence (default: 5)
TRAIN_EVAL_SPLIT_RATIO      Train/eval split ratio (default: 0.95)
RANDOM_SEED                 Random seed for reproducibility (default: 42)
PARTITIONS_*                Partition counts for various outputs


OUTPUT DATA LOCATIONS
---------------------
OUTPUT_PATH/all_events              Cleaned and deduplicated events
OUTPUT_PATH/ads_vocab               Ads domain vocabulary with indices
OUTPUT_PATH/shopping_vocab          Shopping domain vocabulary with indices
OUTPUT_PATH/web_browsing_vocab      Web domain vocabulary with indices
OUTPUT_PATH/all_events_with_id      Events joined with vocabulary IDs
OUTPUT_PATH/events_sorted_temp      Temporary sorted events (intermediate)
OUTPUT_PATH/events_seq              User event sequences (all events)
OUTPUT_PATH/train                   Training dataset
OUTPUT_PATH/eval                    Evaluation dataset


DOMAIN OFFSET CALCULATION
--------------------------
A single domain_offset is computed as the smallest power of 10 strictly greater
than the maximum item_id across all three domain vocabularies:

  domain_offset = 10 ** ceil(log10(max(ads_max, web_max, shop_max) + 1))

Encoded IDs:
  Ads:      encoded_id = item_id                      (0 to ads_max)
  Web:      encoded_id = domain_offset + item_id
  Shopping: encoded_id = domain_offset * 2 + item_id

The domain_offset is stored as a single-row parquet at:
  OUTPUT_PATH/domain_offset

To recover domain from encoded_id downstream:
  domain  = encoded_id // domain_offset   (0=ads, 1=web, 2=shopping)
  item_id = encoded_id  % domain_offset


REUSABILITY & CHECKPOINTING
---------------------------
The pipeline is designed for reusability:
- Each stage checks if its output exists before running
- If output exists, stage is skipped (unless --force-rerun is used)
- You can start from any stage if earlier stages are complete
- Failed runs can be resumed from the last successful stage
- Use --only-stage to inspect individual stage outputs


NOTES
-----
- Requires PySpark with Azure Data Lake access configured
- Spark session configured for production workloads (32G memory, 50 executors)
- Shuffle partitions and coalesce counts are tunable in configuration section
- All counts and statistics are printed during execution
- Vocabulary sizes and offset calculations are displayed in Stage 3
"""

import sys
from pyspark.sql import SparkSession, Window, Row
from pyspark.sql.functions import (
    from_json, col, explode, size, element_at, coalesce, lit,
    countDistinct, count, when, row_number, monotonically_increasing_id,
    min as spark_min, max as spark_max, unix_timestamp, struct, collect_list,
    array_intersect, array, slice, rand, aggregate, array_contains
)
from pyspark.sql.types import (
    ArrayType, StructType, StructField, TimestampType, StringType
)


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Input/Output Paths
# INPUT_PATH = "adl://bingads-algo-prod-networkprotection-c08.azuredatalakestore.net/shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/v3/rows_clicks_no_impressions_min_seqlen10/"
INPUT_PATH  = "adl://bingads-algo-prod-networkprotection-c08.azuredatalakestore.net/shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/v3/L1Score/parquet_data/"
# OUTPUT_PATH = INPUT_PATH  # Override to write outputs to a different location
OUTPUT_PATH = "adl://bingads-algo-prod-networkprotection-c08.azuredatalakestore.net/shares/bingads.hm/local/NativeAds/Relevance/Data/sequential/hstu/v3/L1Score/test_id_creation/"

# Event Type Classifications
ADS_TYPES = ['NativeImpression', 'NativeClick', 'NativeConversion', 'SearchImpression', 'SearchClick']
WEB_BROWSING_TYPES = ['ChromePageTitle', 'EdgePageTitle', 'EdgeSearchQuery', 'OrganicSearchQuery', 'MSN']
SHOPPING_TYPES = ['UET', 'UETShoppingCart', 'UETShoppingView', 'AbandonCart', 'EdgeShoppingCart', 'EdgeShoppingPurchase']

# Positive Event Types (for train/eval split)
ADS_POSITIVE_TYPES = ['NativeClick', 'NativeConversion', 'SearchClick']
WEB_BROWSING_POSITIVE_TYPES = ['ChromePageTitle', 'EdgePageTitle', 'EdgeSearchQuery', 'OrganicSearchQuery', 'MSN', 'UET']
SHOPPING_POSITIVE_TYPES = ['UETShoppingCart', 'UETShoppingView', 'EdgeShoppingCart', 'EdgeShoppingPurchase', 'AbandonCart']

# Domain Configuration
DOMAIN_ID_ADS = 0
DOMAIN_ID_WEB = 1
DOMAIN_ID_SHOPPING = 2
# Domain offsets will be calculated dynamically based on vocab sizes
# Set this to None to auto-calculate, or a fixed number to override
DOMAIN_OFFSET_PATH = OUTPUT_PATH + "domain_offset.tsv"

# Index Starting Points (configurable offsets for each vocab)
ADS_VOCAB_START_INDEX = 20
WEB_VOCAB_START_INDEX = 0
SHOPPING_VOCAB_START_INDEX = 0

# Data Processing Parameters
INVALID_TIME_THRESHOLD = "2010-01-01 00:00:00"
MIN_SEQUENCE_LENGTH = 5
TRAIN_EVAL_SPLIT_RATIO = 0.95  # 95% train, 5% eval
RANDOM_SEED = 42

# Partition/Coalesce Settings
PARTITIONS_ALL_EVENTS = 256
PARTITIONS_ADS_VOCAB = 32
PARTITIONS_SHOPPING_VOCAB = 32
PARTITIONS_WEB_VOCAB = 16
PARTITIONS_EVENTS_SEQ = 1000
PARTITIONS_TRAIN = None  # None means don't repartition
PARTITIONS_EVAL = 16

# Spark Configuration
SPARK_EXECUTOR_MEMORY = "32G"
SPARK_DRIVER_MEMORY = "16G"
SPARK_EXECUTOR_CORES = "8"
SPARK_EXECUTOR_INSTANCES = "50"
SPARK_SHUFFLE_PARTITIONS = "200"


# ============================================================================
# SPARK SESSION INITIALIZATION
# ============================================================================

def create_spark_session():
    """Initialize Spark session with production settings."""
    print("Initializing Spark session...")
    spark = SparkSession\
        .builder\
        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true -Dlog4j.configuration=/usr/local/spark/log4jError.xml") \
        .config("spark.hadoop.fs.adl.oauth2.access.token.provider.type", "Custom") \
        .config("spark.hadoop.fs.adl.oauth2.access.token.provider", "com.microsoft.magnetar.credential.hadoop.adl.MTTokenProvider") \
        .config("spark.hadoop.fs.adl.oauth2.magnetar.tenantid", "72f988bf-86f1-41af-91ab-2d7cd011db47") \
        .config("spark.hadoop.fs.adl.oauth2.magnetar.clientid", "710a9e16-dab8-4d46-97e4-34255ed583be")\
        .config("spark.executor.memory", "20G") \
        .config("spark.driver.memory", "16G") \
        .config("spark.executor.cores", "8") \
        .config("spark.executor.instances", "50") \
        .config("spark.sql.shuffle.partitions", "2000") \
        .config("spark.executor.memoryOverhead", "6G") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .appName("myang-dev") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")
    
    print("Spark session initialized successfully")
    return spark


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def path_exists(spark, path):
    """Check if a path exists in the data lake."""
    try:
        spark.read.parquet(path).limit(1).take(1)
        return True
    except Exception:
        return False


def write_with_partitions(df, path, num_partitions=None):
    """Write dataframe with optional partitioning."""
    if num_partitions:
        df.coalesce(num_partitions).write.mode("overwrite").parquet(path)
    else:
        df.write.mode("overwrite").parquet(path)
    print(f"  ✓ Saved to: {path}")


# ============================================================================
# STAGE 1: PREPARE ALL EVENTS
# ============================================================================

def stage1_prepare_all_events(spark, force_rerun=False):
    """
    Load raw events, parse, filter invalid data, and save cleaned events.
    Output: all_events parquet
    """
    output_path = OUTPUT_PATH + "all_events"

    if not force_rerun and path_exists(spark, output_path):
        print(f"[STAGE 1] Skipping - output already exists: {output_path}")
        return spark.read.parquet(output_path)

    print("[STAGE 1] Preparing all events...")

    # Load raw data
    input_path = INPUT_PATH + "*.parquet"
    print(f"  Loading from: {input_path}")
    df = spark.read.parquet(input_path)
    
    # Create synthetic user_id
    df_with_id = df.withColumn("user_id", monotonically_increasing_id())
    
    # Preserve DemandType and L1RankingV3Score if they exist
    print(f"  Columns in raw data: {df.columns}")
    
    # Define schema for events array
    event_schema = ArrayType(StructType([
        StructField("time", TimestampType(), True),
        StructField("Type", StringType(), True),
        StructField("Market", StringType(), True),
        StructField("CommercialIntentScore", StringType(), True),
        StructField("Texts", ArrayType(StringType()), True)
    ]))
    
    # Parse EventsOnlyWithAd JSON string to array
    df_parsed = df_with_id.withColumn("events", from_json(col("EventsOnlyWithAd"), event_schema))
    
    # Explode events
    select_cols = [col("user_id")]
    if "DemandType" in df.columns:
        select_cols.append(col("DemandType"))
    if "L1RankingV3Score" in df.columns:
        select_cols.append(col("L1RankingV3Score"))
    
    df_exploded = df_parsed.withColumn("event", explode("events")).select(
        *select_cols,
        col("event.time").alias("time"),
        col("event.Type").alias("type"),
        col("event.Market").alias("market"),
        col("event.CommercialIntentScore").alias("commercial_intent_score"),
        col("event.Texts").alias("texts")
    )
    
    # Extract title and url from texts array
    df_extracted = df_exploded.withColumn("title", element_at(col("texts"), 1)) \
                              .withColumn("url", element_at(col("texts"), 2)) \
                              .drop("texts")
    
    # Replace NULL values with placeholders
    df_extracted = df_extracted.withColumn(
        "title", coalesce(col("title"), lit("__MISSING_TITLE__"))
    ).withColumn(
        "url", coalesce(col("url"), lit("__MISSING_URL__"))
    )
    
    # Filter invalid time events
    valid_df = df_extracted.filter(col("time") > lit(INVALID_TIME_THRESHOLD))
    
    # Drop duplicates (excluding DemandType and L1RankingV3Score from dedup logic)
    dedup_cols = ["user_id", "time", "type", "market", "commercial_intent_score", "title", "url"]
    unique_df = valid_df.dropDuplicates(dedup_cols)
    
    row_count = unique_df.count()
    print(f"  Processed {row_count:,} unique events")
    
    # Save
    write_with_partitions(unique_df, output_path, PARTITIONS_ALL_EVENTS)
    
    return unique_df


# ============================================================================
# STAGE 2: CREATE DOMAIN VOCABULARIES
# ============================================================================

def create_vocab_with_index(df, event_types, start_index):
    """Create vocabulary with indices for a specific domain."""
    df_filtered = df.filter(col("type").isin(event_types))
    df_unique = df_filtered.select("title", "url").dropDuplicates()
    
    # Add index using zipWithIndex
    df_with_index = df_unique.rdd.zipWithIndex().map(
        lambda x: Row(**x[0].asDict(), index=x[1] + start_index)
    ).toDF()
    
    # Show index range
    index_stats = df_with_index.select(
        spark_min("index").alias("min_index"),
        spark_max("index").alias("max_index")
    ).collect()[0]
    
    vocab_size = df_with_index.count()
    print(f"    Vocab size: {vocab_size:,} (indices: {index_stats.min_index} to {index_stats.max_index})")
    
    return df_with_index


def stage2_create_ads_vocab(spark, force_rerun=False):
    """Create ads domain vocabulary."""
    output_path = OUTPUT_PATH +"ads_vocab"
    
    if not force_rerun and path_exists(spark, output_path):
        print(f"[STAGE 2A] Skipping - ads vocab already exists: {output_path}")
        return spark.read.parquet(output_path)
    
    print("[STAGE 2A] Creating ads vocab...")
    
    all_events_path = OUTPUT_PATH +"all_events"
    df = spark.read.parquet(all_events_path)
    
    df_with_index = create_vocab_with_index(df, ADS_TYPES, ADS_VOCAB_START_INDEX)
    
    write_with_partitions(df_with_index, output_path, PARTITIONS_ADS_VOCAB)
    
    return df_with_index


def stage2_create_shopping_vocab(spark, force_rerun=False):
    """Create shopping domain vocabulary."""
    output_path = OUTPUT_PATH +"shopping_vocab"
    
    if not force_rerun and path_exists(spark, output_path):
        print(f"[STAGE 2B] Skipping - shopping vocab already exists: {output_path}")
        return spark.read.parquet(output_path)
    
    print("[STAGE 2B] Creating shopping vocab...")
    
    all_events_path = OUTPUT_PATH +"all_events"
    df = spark.read.parquet(all_events_path)
    
    df_with_index = create_vocab_with_index(df, SHOPPING_TYPES, SHOPPING_VOCAB_START_INDEX)
    
    write_with_partitions(df_with_index, output_path, PARTITIONS_SHOPPING_VOCAB)
    
    return df_with_index


def stage2_create_web_vocab(spark, force_rerun=False):
    """Create web browsing domain vocabulary."""
    output_path = OUTPUT_PATH +"web_browsing_vocab"
    
    if not force_rerun and path_exists(spark, output_path):
        print(f"[STAGE 2C] Skipping - web vocab already exists: {output_path}")
        return spark.read.parquet(output_path)
    
    print("[STAGE 2C] Creating web browsing vocab...")
    
    all_events_path = OUTPUT_PATH +"all_events"
    df = spark.read.parquet(all_events_path)
    
    df_with_index = create_vocab_with_index(df, WEB_BROWSING_TYPES, WEB_VOCAB_START_INDEX)
    
    write_with_partitions(df_with_index, output_path, PARTITIONS_WEB_VOCAB)
    
    return df_with_index


# ============================================================================
# STAGE 3: JOIN EVENTS WITH IDS
# ============================================================================

def stage3_create_events_with_ids(spark, force_rerun=False):
    """
    Join all events with their respective domain vocabularies to assign IDs.
    Output: all_events_with_id parquet
    """
    output_path = OUTPUT_PATH +"all_events_with_id"
    
    if not force_rerun and path_exists(spark, output_path):
        print(f"[STAGE 3] Skipping - events with IDs already exist: {output_path}")
        return spark.read.parquet(output_path)
    
    print("[STAGE 3] Creating events with IDs...")
    
    # Load all events
    all_events_path = OUTPUT_PATH +"all_events"
    df = spark.read.parquet(all_events_path)
    
    # Load vocabularies
    df_ads_vocab = spark.read.parquet(OUTPUT_PATH +"ads_vocab")
    df_web_vocab = spark.read.parquet(OUTPUT_PATH +"web_browsing_vocab")
    df_shop_vocab = spark.read.parquet(OUTPUT_PATH +"shopping_vocab")
    
    # Get vocab min/max indices per domain
    ads_stats = df_ads_vocab.select(
        spark_min("index").alias("min_index"),
        spark_max("index").alias("max_index")
    ).collect()[0]
    web_stats = df_web_vocab.select(
        spark_min("index").alias("min_index"),
        spark_max("index").alias("max_index")
    ).collect()[0]
    shop_stats = df_shop_vocab.select(
        spark_min("index").alias("min_index"),
        spark_max("index").alias("max_index")
    ).collect()[0]

    ads_min,  ads_max  = ads_stats.min_index,  ads_stats.max_index
    web_min,  web_max  = web_stats.min_index,  web_stats.max_index
    shop_min, shop_max = shop_stats.min_index, shop_stats.max_index
    
    ads_count = df_ads_vocab.count()
    web_count = df_web_vocab.count()
    shop_count = df_shop_vocab.count()
    
    print(f"  Vocab sizes:")
    print(f"    Ads:      {ads_count:,} (index: {ads_min:,} → {ads_max:,})")
    print(f"    Web:      {web_count:,} (index: {web_min:,} → {web_max:,})")
    print(f"    Shopping: {shop_count:,} (index: {shop_min:,} → {shop_max:,})")
    
    # Calculate domain_offset = smallest power of 10 strictly greater than
    # the largest item_id across all domains. One round number covers all domains,
    # and it is persisted so downstream scripts never recompute it.
    import math
    overall_max = max(ads_max, web_max, shop_max)
    domain_offset = 10 ** math.ceil(math.log10(overall_max + 1))
    print(f"  Overall max item_id: {overall_max:,}  →  domain_offset: {domain_offset:,}")

    # Persist the offset and per-domain index ranges as a single-row TSV.
    # TSV is used instead of parquet because this is a tiny metadata file
    # (1 row, 7 columns) that benefits from being human-readable.
    offset_df = spark.createDataFrame([{
        "domain_offset":  int(domain_offset),
        "ads_min_index":  int(ads_min),  "ads_max_index":  int(ads_max),
        "web_min_index":  int(web_min),  "web_max_index":  int(web_max),
        "shop_min_index": int(shop_min), "shop_max_index": int(shop_max),
    }])
    offset_df.write.mode("overwrite").option("delimiter", "\t").option("header", "true").csv(DOMAIN_OFFSET_PATH)
    print(f"  ✓ domain_offset saved to: {DOMAIN_OFFSET_PATH}")

    offset_web      = domain_offset      # 1 × domain_offset
    offset_shopping = domain_offset * 2  # 2 × domain_offset
    print(f"  Domain offsets → Ads: 0  |  Web: {offset_web:,}  |  Shopping: {offset_shopping:,}")
    
    # Add domain_id column
    df_domain_id = df.withColumn(
        "domain_id",
        when(col("type").isin(ADS_TYPES), DOMAIN_ID_ADS)
        .when(col("type").isin(WEB_BROWSING_TYPES), DOMAIN_ID_WEB)
        .when(col("type").isin(SHOPPING_TYPES), DOMAIN_ID_SHOPPING)
        .otherwise(-1)
    )
    
    # Filter out invalid domain_ids
    filtered_df = df_domain_id.filter(col("domain_id").isin(DOMAIN_ID_ADS, DOMAIN_ID_WEB, DOMAIN_ID_SHOPPING))
    
    # Join with respective vocabularies
    df_ads = filtered_df.filter(col("domain_id") == DOMAIN_ID_ADS).join(
        df_ads_vocab, on=["title", "url"], how="left"
    ).withColumnRenamed("index", "item_id")
    
    df_web = filtered_df.filter(col("domain_id") == DOMAIN_ID_WEB).join(
        df_web_vocab, on=["title", "url"], how="left"
    ).withColumnRenamed("index", "item_id")
    
    df_shop = filtered_df.filter(col("domain_id") == DOMAIN_ID_SHOPPING).join(
        df_shop_vocab, on=["title", "url"], how="left"
    ).withColumnRenamed("index", "item_id")
    
    # Union all three back together
    final_df = df_ads.unionByName(df_web).unionByName(df_shop)
    
    # Check for null item_ids
    null_count = final_df.filter(col("item_id").isNull()).count()
    if null_count > 0:
        print(f"  WARNING: {null_count:,} events have null item_id")
    
    # Create encoded_id using calculated offsets
    df_encode = final_df.withColumn(
        "encoded_id",
        when(col("domain_id") == DOMAIN_ID_ADS, col("item_id"))
        .when(col("domain_id") == DOMAIN_ID_WEB, offset_web + col("item_id"))
        .when(col("domain_id") == DOMAIN_ID_SHOPPING, offset_shopping + col("item_id"))
    )
    
    total_count = df_encode.count()
    print(f"  Total events with IDs: {total_count:,}")
    
    # Save
    write_with_partitions(df_encode, output_path, num_partitions=None)
    
    return df_encode


# ============================================================================
# STAGE 4: CREATE EVENT SEQUENCES
# ============================================================================

def stage4_create_event_sequences(spark, force_rerun=False):
    """
    Group events by user and create time-ordered sequences.
    Output: events_seq parquet
    """
    output_path = OUTPUT_PATH +"events_seq"
    
    if not force_rerun and path_exists(spark, output_path):
        print(f"[STAGE 4] Skipping - event sequences already exist: {output_path}")
        return spark.read.parquet(output_path)
    
    print("[STAGE 4] Creating event sequences...")
    
    # Adjust shuffle partitions for this operation
    spark.conf.set("spark.sql.shuffle.partitions", str(PARTITIONS_EVENTS_SEQ))
    
    # Load events with IDs
    events_path = OUTPUT_PATH +"all_events_with_id"
    df = spark.read.parquet(events_path)
    
    # Add unix timestamp
    df_time_unix = df.withColumn("time_unix", unix_timestamp("time"))
    
    # Sort and write intermediate sorted data
    df_sorted = df_time_unix.orderBy("user_id", "time_unix")
    output_path_temp = OUTPUT_PATH +"events_sorted_temp"
    
    print("  Sorting events by user and time...")
    df_sorted.write.mode("overwrite").parquet(output_path_temp)
    
    # Read back and aggregate
    print("  Aggregating into sequences...")
    df_sorted_read = spark.read.parquet(output_path_temp)
    
    # Determine which optional columns exist
    optional_cols = []
    if "DemandType" in df_sorted_read.columns:
        optional_cols.append("DemandType")
    if "L1RankingV3Score" in df_sorted_read.columns:
        optional_cols.append("L1RankingV3Score")
    
    struct_cols = ["time_unix", "title", "url", "time", "type", "market", 
                   "commercial_intent_score", "domain_id", "item_id", "encoded_id"] + optional_cols
    
    df_struct = df_sorted_read.withColumn(
        "event_struct", struct(*struct_cols)
    )
    
    from pyspark.sql.functions import sort_array

    grouped_df = df_struct.groupBy("user_id").agg(
        collect_list("event_struct").alias("sorted_events")
    )

    grouped_df = grouped_df.withColumn("sorted_events", sort_array("sorted_events"))

    # Build select columns dynamically
    select_cols = [
        "user_id",
        col("sorted_events.encoded_id").alias("encoded_ids"),
        col("sorted_events.title").alias("titles"),
        col("sorted_events.url").alias("urls"),
        col("sorted_events.time_unix").alias("timestamps_unix"),
        col("sorted_events.time").alias("timestamps_readable"),
        col("sorted_events.type").alias("types"),
        col("sorted_events.market").alias("markets"),
        col("sorted_events.commercial_intent_score").alias("commercial_intent_scores"),
        col("sorted_events.domain_id").alias("domain_ids"),
        col("sorted_events.item_id").alias("item_ids"),
    ]
    
    # Add optional columns if they exist
    if "DemandType" in optional_cols:
        select_cols.append(col("sorted_events.DemandType").alias("demand_types"))
    if "L1RankingV3Score" in optional_cols:
        select_cols.append(col("sorted_events.L1RankingV3Score").alias("l1_ranking_scores"))
    
    result_df = grouped_df.select(*select_cols)
    
    user_count = result_df.count()
    print(f"  Created sequences for {user_count:,} users")
    
    # Save
    write_with_partitions(result_df, output_path, num_partitions=None)
    
    return result_df


# ============================================================================
# STAGE 5: CREATE TRAIN/EVAL SPLIT
# ============================================================================

def _last_ads_positive_index_expr(types_col):
    """
    Native Spark expression replacing the Python UDF.

    Uses aggregate() to walk the types array, keeping a running accumulator
    of the current 1-based position and the last position that matched an
    ads-positive event type.  Returns a 1-based cutoff index for use with
    slice(), or 0 if no match is found (unreachable in practice since these
    rows all passed the ads-positive filter upstream).

    aggregate(array, zero, merge_fn, finish_fn) is available in Spark 3.x+.
    The accumulator is a struct<pos INT, last INT>:
      - pos:  increments by 1 on each element
      - last: updated to pos whenever the element is an ads-positive type
    """
    zero = struct(
        lit(0).cast("int").alias("pos"),
        lit(0).cast("int").alias("last"),
    )
    ads_positive_col = array(*[lit(t) for t in ADS_POSITIVE_TYPES])

    merged = aggregate(
        types_col,
        zero,
        lambda acc, x: struct(
            (acc["pos"] + lit(1)).alias("pos"),
            when(array_contains(ads_positive_col, x), acc["pos"] + lit(1))
            .otherwise(acc["last"])
            .alias("last"),
        ),
    )
    return merged["last"]



def stage5_create_train_eval_split(spark, force_rerun=False):
    """
    Split event sequences into train and eval sets.
    - Filter sequences with minimum length
    - Split ads-positive sequences into train/eval
    - Truncate eval sequences to last ads-positive event
    Outputs: train and eval parquet
    """
    train_path = OUTPUT_PATH +"train"
    eval_path = OUTPUT_PATH +"eval"
    
    if not force_rerun and path_exists(spark, train_path) and path_exists(spark, eval_path):
        print(f"[STAGE 5] Skipping - train/eval split already exists")
        return spark.read.parquet(train_path), spark.read.parquet(eval_path)
    
    print("[STAGE 5] Creating train/eval split...")
    
    # Load event sequences
    events_seq_path = OUTPUT_PATH +"events_seq"
    df = spark.read.parquet(events_seq_path)
    
    total_users = df.count()
    print(f"  Total users: {total_users:,}")
    
    # Filter minimum sequence length
    seq_df = df.filter(size(col("encoded_ids")) >= MIN_SEQUENCE_LENGTH)
    filtered_users = seq_df.count()
    print(f"  Users with seq_len >= {MIN_SEQUENCE_LENGTH}: {filtered_users:,}")
    
    # Split into ads-positive and other
    df_ads_positive = seq_df.filter(
        size(array_intersect(col("types"), array(*[lit(x) for x in ADS_POSITIVE_TYPES]))) > 0
    )
    
    df_other = seq_df.filter(
        size(array_intersect(col("types"), array(*[lit(x) for x in ADS_POSITIVE_TYPES]))) == 0
    )
    
    ads_positive_count = df_ads_positive.count()
    other_count = df_other.count()
    print(f"  Users with ads-positive events: {ads_positive_count:,}")
    print(f"  Users without ads-positive events: {other_count:,}")
    
    # Split ads-positive into train/eval
    train_df_ads, eval_df_ads = df_ads_positive.randomSplit(
        [TRAIN_EVAL_SPLIT_RATIO, 1.0 - TRAIN_EVAL_SPLIT_RATIO], 
        seed=RANDOM_SEED
    )
    
    train_ads_count = train_df_ads.count()
    eval_ads_count = eval_df_ads.count()
    print(f"  Train/eval split ({TRAIN_EVAL_SPLIT_RATIO:.1%}/{1.0-TRAIN_EVAL_SPLIT_RATIO:.1%}): {train_ads_count:,} / {eval_ads_count:,}")
    
    # Create train dataset (other + train_ads)
    train_df = df_other.unionByName(train_df_ads)
    final_train_count = train_df.count()
    print(f"  Final train users: {final_train_count:,}")
    
    # Process eval dataset - truncate to last ads-positive event
    print("  Truncating eval sequences to last ads-positive event...")
    eval_df_ads = eval_df_ads.withColumn("cutoff_idx", _last_ads_positive_index_expr(col("types")))
    
    # Truncate all array columns
    array_cols = [f.name for f in eval_df_ads.schema.fields 
                  if str(f.dataType).startswith("ArrayType") and f.name != "cutoff_idx"]
    
    for col_name in array_cols:
        eval_df_ads = eval_df_ads.withColumn(col_name, slice(col(col_name), 1, col("cutoff_idx")))
    
    eval_df_ads = eval_df_ads.drop("cutoff_idx")
    
    # Filter eval to maintain minimum length after truncation
    eval_df = eval_df_ads.filter(size(col("encoded_ids")) >= MIN_SEQUENCE_LENGTH)
    eval_df.cache()
    final_eval_count = eval_df.count()
    print(f"  Final eval users (after truncation and filtering): {final_eval_count:,}")
    
    # Save train
    print("  Saving train dataset...")
    write_with_partitions(train_df, train_path, PARTITIONS_TRAIN)
    
    # Save eval
    print("  Saving eval dataset...")
    write_with_partitions(eval_df, eval_path, PARTITIONS_EVAL)
    eval_df.unpersist()
    
    return train_df, eval_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(start_stage=1, end_stage=None, force_rerun=False):
    """
    Run the complete pipeline or start from a specific stage.
    
    Args:
        start_stage: Stage number to start from (1-5)
        end_stage: Stage number to end at (1-5). If None, runs to completion
        force_rerun: If True, rerun stages even if outputs exist
    """
    spark = create_spark_session()
    
    if end_stage is None:
        end_stage = 5
    
    print("\n" + "="*80)
    print("ID EMBEDDING PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  INPUT_PATH:  {INPUT_PATH}")
    print(f"  OUTPUT_PATH: {OUTPUT_PATH}")
    print(f"  MIN_SEQUENCE_LENGTH: {MIN_SEQUENCE_LENGTH}")
    print(f"  TRAIN_EVAL_SPLIT: {TRAIN_EVAL_SPLIT_RATIO:.1%} / {1.0-TRAIN_EVAL_SPLIT_RATIO:.1%}")
    print(f"  DOMAIN_OFFSET: Power-of-10, computed from vocab sizes and saved to {DOMAIN_OFFSET_PATH}")
    if start_stage == end_stage:
        print(f"  Running ONLY Stage {start_stage}")
    else:
        print(f"  Running Stages {start_stage} to {end_stage}")
    print("="*80 + "\n")
    
    try:
        # Stage 1: Prepare all events
        if start_stage <= 1 <= end_stage:
            stage1_prepare_all_events(spark, force_rerun)
            print()
        
        # Stage 2: Create vocabularies
        if start_stage <= 2 <= end_stage:
            stage2_create_ads_vocab(spark, force_rerun)
            stage2_create_shopping_vocab(spark, force_rerun)
            stage2_create_web_vocab(spark, force_rerun)
            print()
        
        # Stage 3: Join events with IDs
        if start_stage <= 3 <= end_stage:
            stage3_create_events_with_ids(spark, force_rerun)
            print()
        
        # Stage 4: Create event sequences
        if start_stage <= 4 <= end_stage:
            stage4_create_event_sequences(spark, force_rerun)
            print()
        
        # Stage 5: Create train/eval split
        if start_stage <= 5 <= end_stage:
            stage5_create_train_eval_split(spark, force_rerun)
            print()
        
        print("="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with exception:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        spark.stop()


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='ID Embedding Pipeline')
    parser.add_argument('--start-stage', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='Stage to start from (1=prepare events, 2=vocabs, 3=join IDs, 4=sequences, 5=train/eval)')
    parser.add_argument('--end-stage', type=int, default=None, choices=[1, 2, 3, 4, 5],
                       help='Stage to end at. If not specified, runs to completion')
    parser.add_argument('--only-stage', type=int, default=None, choices=[1, 2, 3, 4, 5],
                       help='Run only this specific stage (shorthand for --start-stage X --end-stage X)')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force rerun of all stages even if outputs exist')
    
    args = parser.parse_args()
    
    # Handle --only-stage shorthand
    if args.only_stage is not None:
        start_stage = args.only_stage
        end_stage = args.only_stage
    else:
        start_stage = args.start_stage
        end_stage = args.end_stage
    
    run_pipeline(start_stage=start_stage, end_stage=end_stage, force_rerun=args.force_rerun)