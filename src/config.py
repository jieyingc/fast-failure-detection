DATASET_NAME="SWE-bench/SWE-smith-trajectories"
DATASET_SPLIT="tool"

LABEL_COL="resolved"
INSTANCE_COL="instance_id"
REPO_COL="repo_id"
USE_MESSAGE_TFIDF = True

MODEL_NAME = "xgb"
THRESHOLDS = [0.6, 0.7]

TFIDF_SHOW_TOP_WORDS = 30

CV_TYPE = "leave_one_repo_out" # "group_kfold" or "leave_one_repo_out"
N_SPLITS = 10

# TFIDF_TEXT_SOURCES = ["message"]
# TFIDF_TEXT_SOURCES = ["message", "file_path"]
# TFIDF_TEXT_SOURCES = ["message", "patch"]
# TFIDF_TEXT_SOURCES = ["message", "file_path", "patch"]
# TFIDF_TEXT_SOURCES = ["user", "assistant", "tool", "file_path"]
TFIDF_TEXT_SOURCES = ["assistant", "file_path"]
# TFIDF_TEXT_SOURCES = ["function_name", "assistant", "file_path"]
# TFIDF_TEXT_SOURCES = ["class_name", "assistant", "file_path"]
# TFIDF_TEXT_SOURCES = ["function_name", "class_name", "assistant", "file_path"]
# TFIDF_TEXT_SOURCES = ["assistant", "file_path", "function_name", "import_name"]

TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 5
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"
TFIDF_SUBLINEAR_TF = True

# Current default: use non-empty patch only
USE_NON_EMPTY_ONLY=True

PATCH_FEATURES = [
    "patch_char_len",
    "num_lines_added",
    "num_lines_removed",
    "num_files_touched",
    "test_file_modified",
    "contain_repeated_blocks",
    "num_test_files_touched",
    "num_python_files_touched",
    "touch_requirements_file",
    "touch_config_file",
    "touch_docs_file",
    "touch_init_file",
    "touch_multiple_dirs",
    # "contains_assert_change",
    # "contains_import_change",
    # "contains_exception_handling_change",
]
    
MESSAGE_FEATURES = [ 
    "num_messages",  
    "messages_char_len", 
    "num_assistant_messages",
    "num_user_messages",
    "num_tool_messages",
    "num_action_messages",
    "num_observation_messages",
    "has_submit",
    "num_submit_calls",
    "has_bash",
    "num_bash_calls",
    "has_str_replace_editor",
    "num_edit_calls",
    "num_error_messages",
]

REDUCED_MESSAGE_FEATURES = [
    "num_messages",
    "messages_char_len", 
    "num_action_messages",
    "num_observation_messages",
    "num_submit_calls",
    "num_bash_calls",
    "num_edit_calls",
    "num_error_messages",
]
    
FULL_FEATURES = PATCH_FEATURES + MESSAGE_FEATURES
REDUCED_FEATURES = PATCH_FEATURES + REDUCED_MESSAGE_FEATURES
    
# FEATURE_COLS = FULL_FEATURES
# FEATURE_COLS = PATCH_FEATURES
# FEATURE_COLS = MESSAGE_FEATURES
FEATURE_COLS = REDUCED_FEATURES
    
    
LOG_FEATURES = [
    "patch_char_len",
    "num_lines_added",
    "num_lines_removed",
    "num_files_touched",
    "num_messages",
    "messages_char_len",  
    "num_submit_calls",
    "num_bash_calls",
    "num_edit_calls",
    "num_error_messages",
]
