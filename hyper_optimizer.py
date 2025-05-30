

def get_keys():
    adjusted_feature_tracker_names = {key - 1: value for key, value in feature_tracker_names.items()}
    print(adjusted_feature_tracker_names)
    return adjusted_feature_tracker_names

