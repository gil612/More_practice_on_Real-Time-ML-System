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
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
    batch_size: int,
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
        start_offline_materialization (bool): Whether to start the offline
            materialization or not when we save the `value` to the feature group
        batch_size (int): The number of messages to accumulate in-memory before pushing
            to the Feature Store

    Returns:
        None
    """
    # Configure an Application
    app = Application(
        broker_address=kafka_broker_address,  
        consumer_group=kafka_consumer_group,
        auto_offset_reset="earliest",  # Add this to read from the beginning
    )

    batch = []
    total_messages = 0  # Add counter

    # Create a consumer and start a polling loop
    with app.get_consumer() as consumer:
        logger.info(f"Subscribing to topic: {kafka_input_topic}")
        consumer.subscribe(topics=[kafka_input_topic])

        while True:
            msg = consumer.poll(0.1)

            if msg is None:
                if len(batch) > 0:
                    logger.info(f"No more messages. Processing final batch of size {len(batch)}")
                    push_value_to_feature_group(
                        batch,
                        feature_group_name,
                        feature_group_version,
                        feature_group_primary_keys,
                        feature_group_event_time,
                        start_offline_materialization,
                    )
                    batch = []
                continue
            elif msg.error():
                logger.error('Kafka error:', msg.error())
                continue

            value = msg.value()
            total_messages += 1  # Increment counter

            # decode the message bytes into a dictionary
            import json
            value = json.loads(value.decode('utf-8'))
            
            # Append the message to the batch
            batch.append(value)

            # Log progress periodically
            if total_messages % 1000 == 0:
                logger.info(f"Processed {total_messages} messages so far")

            # If the batch is not full yet, continue polling
            if len(batch) < batch_size:
                logger.debug(f'Batch has size {len(batch)} < {batch_size}...')
                continue
            
            logger.debug(f'Batch has size {len(batch)} >= {batch_size}... Pushing data to Feature Store')
            push_value_to_feature_group(
                batch,
                feature_group_name,
                feature_group_version,
                feature_group_primary_keys,
                feature_group_event_time,
                start_offline_materialization,
          )

            # Clear the batch
            batch = []
            
            # Store the offset of the processed message on the Consumer 
            # for the auto-commit mechanism.
            # It will send it to Kafka in the background.
            # Storing offset only after the message is processed enables at-least-once delivery
            # guarantees.
            consumer.store_offsets(message=msg)

if __name__ == "__main__":
    # Fix the import path
    from config import config  # Remove 'src.' from the import

    topic_to_feature_store(
        kafka_broker_address=config.kafka_broker_address,
        kafka_input_topic=config.kafka_input_topic,
        kafka_consumer_group=config.kafka_consumer_group,
        feature_group_name=config.feature_group_name,
        feature_group_version=config.feature_group_version,
        feature_group_primary_keys=config.feature_group_primary_keys,
        feature_group_event_time=config.feature_group_event_time,
        start_offline_materialization=config.start_offline_materialization,
        batch_size=config.batch_size,
    )