import motor.motor_asyncio

from core import logger, settings  # type: ignore


MONGO_URI: str = (f"mongodb://{settings.MONGO_USERNAME}:{settings.MONGO_PASSWORD}@"
                  f"{settings.MONGO_HOST}:{settings.MONGO_PORT}")


class MongoClient:
    def __init__(self):

        self._client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        self.db = self._client[settings.MONGO_DBNAME]

    async def drop_all_collections(self) -> None:
        collection_names = await self.db.list_collection_names()
        for collection in collection_names:
            await self.db.drop_collection(collection)
        logger.info(f"All collections have been dropped from {self.db.name}")

    async def drop_collection(self, collection_name: str) -> None:
        collection_names = self.db[f"{collection_name}"]
        await self.db.drop_collection(collection_names)

        logger.info(
            f"Collection {collection_name} have been dropped from {self.db.name}"
        )
