"""Constants for supported checkpoint-comparison schema versions."""

SCHEMA_NAME = "lerobot-state-atlas.checkpoint-comparison"
SCHEMA_MAJOR = 1
SCHEMA_MINOR = 0
SCHEMA_MINOR_PROJECTION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({(SCHEMA_MAJOR, SCHEMA_MINOR), (1, 1)})

MANIFEST_FILENAME = "manifest.json"
PLANS_FILENAME = "plans.json"

BASE_POLICY_ID = "base-pi05"
BASE_POLICY_LABEL = "Base π0.5"
FINE_TUNED_POLICY_ID = "fine-tuned-pi05"
FINE_TUNED_POLICY_LABEL = "Fine-tuned π0.5"

ACTION_DIMENSION = 14
CHUNK_LENGTH = 50
NOISE_SHAPE = (1, 50, 32)
