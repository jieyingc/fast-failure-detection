DATASET_NAME="SWE-bench/SWE-smith-trajectories"
DATASET_SPLIT="tool"

LABEL_COL="resolved"
INSTANCE_COL="instance_id"
REPO_COL="repo_id"
USE_MESSAGE_TFIDF = True

MODEL_NAME = "xgb"
THRESHOLD = 0.4

# Current default: use non-empty patch only
USE_NON_EMPTY_ONLY=True

PATCH_FEATURES = [
    "patch_char_len",
    "num_lines_added",
    "num_lines_removed",
    "num_files_touched",
    "test_file_modified",
    "contain_repeated_blocks",
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
