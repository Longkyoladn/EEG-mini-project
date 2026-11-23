"""Feature contracts shared by classical ML pipelines."""

from src.features.ml3_features import (
    DEFAULT_TIME_WINDOWS,
    FREQUENCY_BANDS_HZ,
    FeatureExtractionConfig,
    FeatureExtractor,
    FeatureTable,
    SubjectEpochs,
    TemporalFeatureExtractor,
    TimeWindow,
    WaveletConfig,
    WaveletFeatureExtractor,
    extract_subject_features,
)
from src.features.ml3_feature_store import (
    ProcessedFeatureSource,
    extract_feature_groups,
    load_processed_feature_source,
    save_subject_feature_table,
)

__all__ = [
    "DEFAULT_TIME_WINDOWS",
    "FREQUENCY_BANDS_HZ",
    "FeatureExtractionConfig",
    "FeatureExtractor",
    "FeatureTable",
    "SubjectEpochs",
    "TemporalFeatureExtractor",
    "TimeWindow",
    "WaveletConfig",
    "WaveletFeatureExtractor",
    "extract_subject_features",
    "ProcessedFeatureSource",
    "extract_feature_groups",
    "load_processed_feature_source",
    "save_subject_feature_table",
]
