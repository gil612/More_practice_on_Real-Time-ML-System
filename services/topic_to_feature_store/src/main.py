from typing import List

from quixstreams import Application
from loguru import logger




def topic_to_feature_store(
    kafka_broker_address: str,
    kafka_input_topic: str,
    kafka_consumer_group: str,
    feature_group_name: str,
    feature_group_version: int,
    feature_group_primary_keys: List[str],
    feature_group_event_time: str,
    start_offline_materialization: bool,
    batch_size: int
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
    logger.info(f"Connecting to Kafka at {kafka_broker_address}")
    logger.info(f"Using topic: {kafka_input_topic}")
    logger.info(f"Using consumer group: {kafka_consumer_group}")

    try:
        app = Application(
            broker_address=kafka_broker_address,
            consumer_group=kafka_consumer_group,
        )
        logger.info("Successfully created Kafka application")
    except Exception as e:
        logger.error(f"Failed to create Kafka application: {str(e)}")
        raise

    batch = []

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

            # Append the messages to batch
            batch.append(value)

            # If the batch is not full yet, continue polling
            if len(batch) < batch_size:
                logger.debug(f"Batch is not full yet, continuing to poll. Batch size: {len(batch)} < {batch_size}")
                continue
            
            logger.debug(f"Batch is full, pushing to feature group. Batch size: {len(batch)} >= {batch_size}")
            push_value_to_feature_group(
                value=batch,
                project_name=hopsworks_config.hopsworks_project_name,
                feature_group_name=feature_group_name,
                feature_group_version=feature_group_version,
                feature_group_primary_keys=feature_group_primary_keys,
                feature_group_event_time=feature_group_event_time,
                start_offline_materialization=start_offline_materialization,
            )
            # clear the batch
            batch = []

            # Store the offset
            consumer.store_offsets(message=msg)

if __name__ == "__main__":
    from config import config, hopsworks_config
    from hopsworks_api import push_value_to_feature_group

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