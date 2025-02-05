import os
from typing import List, Optional, Tuple, Dict, Any
import time

from loguru import logger
import pandas as pd
import hopsworks
from hsfs.feature_view import FeatureView
from hsfs.feature_store import FeatureStore

from src.config import HopsworksConfig



class OhlcDataReader:
    """
    A class to help us read our OHLC data from the feature store.
    
    The Hopsworks credentials are read from the environment variables.
    - HOPSWORKS_PROJECT_NAME
    - HOPSWORKS_API_KEY
    """
    def __init__(
        self,
        ohlc_window_sec: int,
        hopsworks_config: HopsworksConfig,
        feature_view_name: str,
        feature_view_version: int,
        feature_group_name: Optional[str] = None,
        feature_group_version: Optional[int] = None,
    ):
        self.ohlc_window_sec = ohlc_window_sec
        self.feature_view_name = feature_view_name
        self.feature_view_version = feature_view_version
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version

        self._fs = self._get_feature_store(hopsworks_config)
    
    def _get_primary_keys_to_read_from_online_store(
        self,
        product_id: str,
        last_n_minutes: int,
    ) -> List[Dict[str, Any]]:
        """
        Returns the primary keys we will use to read the OHLC data from the feature store.

        Args:
            product_id (str): The product ID for which we want to get the OHLC data.
            last_n_minutes (int): The number of minutes to go back in time.

        Returns:
            List[Dict[str, Any]]: The list of primary keys we will use to read the OHLC data.
        """
        timestamp_keys: List[int] = self._get_timestamp_keys(
            last_n_minutes=last_n_minutes,
        )
        # timestamp_keys = [1719101940000]

        primary_keys = [
            {
                'product_id': product_id,
                'timestamp': timestamp,
            } for timestamp in timestamp_keys
        ]
        
        return primary_keys
    
    def read_from_online_store(
        self,
        product_id: str,
        last_n_minutes: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Reads OHLC data from the online feature store for the given `product_ids`
        and the time range `[from_timestamp_ms, to_timestamp_ms]` in `self.ohlc_window_sec`
        steps

        Args:
            product_ids (List[str]): The product IDs for which we want to get the OHLC data.
            from_timestamp_ms (Optional[int]): The starting timestamp in milliseconds.
            to_timestamp_ms (Optional[int]): The ending timestamp in milliseconds.
            last_n_minutes (Optional[int]): The number of minutes to go back in time.

        Returns:
        """
        # list of primary keys we will use to read the OHLC data
        primary_keys = self._get_primary_keys_to_read_from_online_store(
            product_id=product_id,
            last_n_minutes=last_n_minutes,
        )
        logger.debug(f'Primary keys: {primary_keys}')

        feature_view = self._get_feature_view()
        features = feature_view.get_feature_vectors(
            entry=primary_keys,
            return_type="pandas"
        )

        # features.sort_values(by='timestamp', inplace=True)
        features = features.sort_values(by='timestamp').reset_index(drop=True)

        # breakpoint()

        return features

    def _get_timestamp_keys(
        self,
        last_n_minutes: int,
    ) -> List[int]:
        """
        Returns the tuple (from_timestamp_ms, to_timestamp_ms) that we will use to
        read the OHLC data from the feature store.

        We also validate the input parameters to make sure they are consistent.

        Args:
            from_timestamp_ms (Optional[int]): The starting timestamp in milliseconds.
            to_timestamp_ms (Optional[int]): The ending timestamp in milliseconds.
            last_n_minutes (Optional[int]): The number of minutes to go back in time.

        Returns:
            List[int]: The list of timestamps we will use to read the OHLC data.
        """        
        to_timestamp_ms = int(time.time() * 1000)
        to_timestamp_ms -= to_timestamp_ms % 60000

        n_candles_per_minutes = 60 // self.ohlc_window_sec

        timestamps = [to_timestamp_ms - i * self.ohlc_window_sec * 1000 \
                      for i in range(last_n_minutes * n_candles_per_minutes)]
        
        return timestamps

    def _get_feature_view(self) -> FeatureView:
        """
        Returns the feature view object that reads data from the feature store
        """
        if self.feature_group_name is None:
            # We try to get the feature view without creating it.
            # If it does not exist, we will raise an error because we would
            # need the feature group info to create it.
            try:
                return self._fs.get_feature_view(
                    name=self.feature_view_name,
                    version=self.feature_view_version,
                )
            except Exception as e:
                raise ValueError(
                    'The feature group name and version must be provided if the feature view does not exist.'
                )
        
        # We have the feature group info, so we first get it
        feature_group = self._fs.get_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
        )

        # and we now create it if it does not exist
        feature_view = self._fs.get_or_create_feature_view(
            name=self.feature_view_name,
            version=self.feature_view_version,
            query=feature_group.select_all(),
        )
        # and if it already existed, we check that its feature group name and version match
        # the ones we have in `self.feature_group_name` and `self.feature_group_version`
        # otherwise we raise an error
        possibly_different_feature_group = \
            feature_view.get_parent_feature_groups().accessible[0]
        
        if possibly_different_feature_group.name != feature_group.name or \
            possibly_different_feature_group.version != feature_group.version:
            raise ValueError(
                'The feature view and feature group names and versions do not match.'
            )
        
        return feature_view
        
    def read_from_offline_store(
        self,
        product_id: str,
        # from_timestamp_ms: int,
        # to_timestamp_ms: int
        last_n_days: int,
    ) -> pd.DataFrame:
        """
        Reads OHLC data from the offline feature store for the given product_id
        """
        to_timestamp_ms = int(time.time() * 1000)
        from_timestamp_ms = to_timestamp_ms - last_n_days * 24 * 60 * 60 * 1000
        
        feature_view = self._get_feature_view()
        features = feature_view.get_batch_data()

        # filter the features for the given product_id and time range
        features = features[features['product_id'] == product_id]
        features = features[features['timestamp_ms'] >= from_timestamp_ms]
        features = features[features['timestamp_ms'] <= to_timestamp_ms]
        # sort the features by timestamp (ascending)
        features = features.sort_values(by='timestamp_ms').reset_index(drop=True)
        

        # breakpoint()

        return features

    @staticmethod
    def _get_feature_store(hopsworks_config: HopsworksConfig) -> FeatureStore:
        """
        Returns the feature store object that we will use to read our OHLC data.
        """
        project = hopsworks.login(
            project=hopsworks_config.hopsworks_project_name,
            api_key_value=hopsworks_config.hopsworks_api_key,
        )


        return project.get_feature_store()


if __name__ == '__main__':

    ohlc_data_reader = OhlcDataReader(
        feature_view_name='ohlc_feature_view',
        feature_view_version=10,
        feature_group_name='ohlc_feature_group',
        feature_group_version=7,
        ohlc_window_sec=60
    )

    # check if reading from the online store works
    output = ohlc_data_reader.read_from_online_store(
        product_id='BTC/USD',
        last_n_minutes=20,
    )
    logger.debug(f'Live OHLC data: {output}')

    # check if reading from the offline store works
    output = ohlc_data_reader.read_from_offline_store(
        product_id='BTC/USD',
        last_n_days=90,
    )
    logger.debug(f'Historical OHLC data: {output}')
    output.to_csv('ohlc_data.csv', index=False)