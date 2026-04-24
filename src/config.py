DATASET_NAME = "SWE-bench/SWE-smith-trajectories"
DATASET_SPLIT = "tool"

LABEL_COL = "resolved"
INSTANCE_COL = "instance_id"
REPO_COL = "repo_id"
USE_NON_EMPTY_ONLY = True
USE_MESSAGE_TFIDF = True

MODEL_NAME = "xgb"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SUCCESS_RECALL = 0.8

CV_TYPE = "group_kfold"  # "group_kfold" or "leave_one_repo_out"
N_SPLITS = 10

TFIDF_TEXT_SOURCES = ["assistant", "file_path"]

TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"
TFIDF_SUBLINEAR_TF = True

TFIDF_SHOW_TOP_WORDS = 0

PATCH_FEATURES = [
    "patch_char_len",
    "num_lines_added",
    "num_lines_removed",
    "num_files_touched",
    "test_file_modified",
    "num_test_files_touched",
    "num_python_files_touched",
    "touch_requirements_file",
    "touch_config_file",
    "touch_docs_file",
    "touch_init_file",
    "touch_multiple_dirs",
    "contains_assert_change",
    "contains_import_change",
    "contains_exception_handling_change",
]


MESSAGE_FEATURES = [
    "num_messages",
    "messages_char_len", 
    "num_action_messages",
    "num_observation_messages",
    "num_submit_calls",
    "num_bash_calls",
    "num_edit_calls",
]

FEATURE_COLS = PATCH_FEATURES + MESSAGE_FEATURES
    
LOG_FEATURES = [
    "patch_char_len",
    "num_lines_added",
    "num_lines_removed",
    "num_files_touched",
    "num_test_files_touched",
    "num_python_files_touched",
    "num_messages",
    "messages_char_len",  
    "num_submit_calls",
    "num_bash_calls",
    "num_edit_calls",
]
