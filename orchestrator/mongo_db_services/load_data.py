import json
import os

from pymongo import MongoClient

from core import logger, settings


async def load_initial_data(db: MongoClient):
    await db.drop_collection(settings.MONGO_COLLECTION_PROMPTS)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(current_dir, "../data.json"), "r") as file:
        data = json.load(file)
    await db.db[settings.MONGO_COLLECTION_PROMPTS].insert_many(data)
    logger.info(f"Initial {settings.MONGO_COLLECTION_PROMPTS} data successfully loaded")
