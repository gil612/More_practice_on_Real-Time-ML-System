from typing import List, Dict

import hopsworks
import pandas as pd

from src.config import config, hopsworks_config

# If we want to keep the connection alive, we can use the following code instead in the push_value_to_feature_group function:

# project = hopsworks.login(
#     project=hopsworks_config.hopsworks_project_name,
# )

# feature_store = project.get_feature_store()

def push_value_to_feature_group(
    project_name: str,
    feature_group_name: str,
    feature_group_version: int,
    features_dict: Dict,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool
):
    """
    Pushes the given `value` to the given `feature_group_name` in the Feature Store.

    Args:
        project_name (str): The name of the project
        feature_group_name (str): The name of the Feature Group
        feature_group_version (int): The version of the Feature Group
        features_dict (Dict): The dictionary containing the features
        feature_group_primary_keys (List[str]): The primary key of the Feature Group
        feature_group_event_time (str): The event time of the Feature Group
        start_offline_materialization (bool): Whether to start offline materialization or not when we save the 'value' to the feature group

    Returns:
        None
    """
    project = hopsworks.login(
    project=hopsworks_config.hopsworks_project_name,
)

    feature_store = project.get_feature_store()
    

    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        primary_key=feature_group_primary_keys,
        event_time=feature_group_event_time,
        online_enabled=True,

        # TODO: either as homework or I will show one example.
        # expectation_suite=expectation_suite_transactions,
    )

    # transform the value dict into a pandas DataFrame
    value_df = pd.DataFrame([features_dict])

    # push the value to the Feature Store
    feature_group.insert(
        value_df,
        write_options={"start_offline_materialization": start_offline_materialization},
    )

