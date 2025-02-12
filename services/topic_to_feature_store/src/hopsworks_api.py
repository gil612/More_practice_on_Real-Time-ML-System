from typing import List

import hopsworks
import pandas as pd

from config import hopsworks_config as config

# connect to ours Hopsworks project
project = hopsworks.login(
    project=config.hopsworks_project_name,
    api_key_value=config.hopsworks_api_key
)

# get a handle to the Feature Store
feature_store = project.get_feature_store()


def push_value_to_feature_group(
    value: List[dict],
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
):
    """
    Pushes the given `value` to the given `feature_group_name` in the Feature Store.

    Args:
        value (List[dict]): The value to push to the Feature Store
        feature_group_name (str): The name of the Feature Group
        feature_group_version (int): The version of the Feature Group
        feature_group_primary_keys (List[str]): The primary key of the Feature Group
        feature_group_event_time (str): The event time of the Feature Group
        start_offline_materialization (bool): Whether to start the offline
            materialization or not when we save the `value` to the feature group

    Returns:
        None
    """
    # get a handle to the Feature Group we want to save the `value` to
    feature_group = feature_store.get_or_create_feature_group(
        name=feature_group_name,
        version=feature_group_version,
        primary_key=feature_group_primary_keys,
        event_time=feature_group_event_time,
        online_enabled=False,

        # TODO: either as homework or I will show one example.
        # expectation_suite=expectation_suite_transactions,
    )

    # transform the value dict into a pandas DataFrame
    value_df = pd.DataFrame(value)

    # breakpoint()

    # push the value to the Feature Store
    feature_group.insert(
        value_df,
        write_options={"start_offline_materialization" : start_offline_materialization}
    )