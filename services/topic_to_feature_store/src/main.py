from typing import List
from quixstreams import Application
from loguru import logger
from hopsworks_api import push_value_to_feature_group


def topic_to_feature_store(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_consumer_group: str,
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_key: List[str],
    feature_group_event_time: str,
):
    """
    Reds incoming messages frim the given 'kafka_input_topic', and pushes them to the given 'feature_group_name"

    Args:
        kafka_broker_address: The address of the Kafka broker
        kafka_input_topic: The topic to read from
        kafka_consumer_group: The consumer group to use
        feature_group_name: The name of the feature group to push to
        feature_group_version: The version of the feature group to push to
        feature_group_primary_key: The primary key of the feature group to push to
        feature_group_event_time: The event time of the Feature Group
    Returns:
        None
    """
    app = Application(
        broker_address=kafka_broker_address,
        consumer_group=kafka_consumer_group,
    )
    
    with app.get_consumer() as consumer:
        consumer.subscribe(topics=[kafka_input_topic])

        while True:
            msg = consumer.poll(0.1)
            
            if msg is None:
                continue
            elif msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            
            value = msg.value()

            # decode the message bytes into a dictionary
            import json
            value = json.loads(value.decode("utf-8"))
            
            # Push the value to the feature store
            push_value_to_feature_group(
                value,
                feature_group_name,
                feature_group_version,
                feature_group_primary_key,
                feature_group_event_time
            )

            # Store the offset of the processed message on the Consumer 
            # for the auto-commit mechanism.
            # It will send it to Kafka in the background.
            # Storing offset only after the message is processed enables at-least-once delivery
            # guarantees.
            consumer.store_offsets(message=msg)

if __name__ == "__main__":
    from config import config
    topic_to_feature_store(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        feature_group_primary_key=config.feature_group_primary_key,
        feature_group_event_time=config.feature_group_event_time,
    )

