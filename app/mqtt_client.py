import json

import paho.mqtt.client as mqtt

from .config import settings


def publish(
    topic: str,
    payload: dict
):

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2
    )

    if settings.MQTT_USERNAME:

        client.username_pw_set(
            settings.MQTT_USERNAME,
            settings.MQTT_PASSWORD
        )

    client.connect(
        settings.MQTT_HOST,
        settings.MQTT_PORT,
        60
    )

    client.loop_start()

    info = client.publish(
        topic,
        json.dumps(payload),
        qos=1
    )

    info.wait_for_publish()

    client.loop_stop()

    client.disconnect()
