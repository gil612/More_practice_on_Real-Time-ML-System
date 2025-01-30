from typing import List

from quixstreams import Application
from loguru import logger

from src.hopsworks_api import push_value_to_feature_group
from src.config import config, hopsworks_config


def topic_to_feature_store(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_consumer_group: str,
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool
):
    """
    Reads incoming messages from the given `kafka_input_topic`, and pushes them to the
    given `feature_group_name` in the Feature Store.

    Args:
        kafka_broker_address (str): The address of the Kafka broker
        kafka_input_topic (str): The Kafka topic to read the messages from
        kafka_consumer_group (str): The Kafka consumer group
        feature_group_name (str): The name of the Feature Group
        feature_group_version (int): The version of the Feature Group
        feature_group_primary_keys (List[str]): The primary key of the Feature Group
        feature_group_event_time (str): The event time of the Feature Group
        start_offline_materialization (bool): Whether to start offline materialization or not when we save the 'value' to the feature group

    Returns:
        None
    """
    # Configure an Application. 
    # The config params will be used for the Consumer instance too.
    app = Application(
        broker_address=kafka_broker_address,  
        consumer_group=kafka_consumer_group,
    )

    # Create a consumer and start a polling loop
    with app.get_consumer() as consumer:

        consumer.subscribe(topics=[kafka_input_topic])

        while True:
            msg = consumer.poll(0.1)

            if msg is None:
                continue
            elif msg.error():
                logger.error('Kafka error:', msg.error())
                continue

            value = msg.value()

            # decode the message bytes into a dictionary
            import json
            value = json.loads(value.decode('utf-8'))
            
            push_value_to_feature_group(
                project_name=hopsworks_config.hopsworks_project_name,
                feature_group_name=feature_group_name,
                feature_group_version=feature_group_version,
                features_dict=value,
                feature_group_primary_keys=feature_group_primary_keys,
                feature_group_event_time=feature_group_event_time,
                start_offline_materialization=start_offline_materialization,
            )

            # breakpoint()

            # we need to push the value to the Feature Store here

            
            # Store the offset of the processed message on the Consumer 
            # for the auto-commit mechanism.
            # It will send it to Kafka in the background.
            # Storing offset only after the message is processed enables at-least-once delivery
            # guarantees.
            consumer.store_offsets(message=msg)

if __name__ == "__main__":

    topic_to_feature_store(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        feature_group_primary_keys=config.feature_group_primary_keys,
        feature_group_event_time=config.feature_group_event_time,
        start_offline_materialization=config.start_offline_materialization,
    )