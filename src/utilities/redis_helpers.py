import redis.asyncio as redis
from typing import Dict, List
from config.setting import Settings
import json
from collections.abc import Awaitable

settings = Settings()  # type: ignore

class RedisHelper:
    def __init__(self, conn: redis.Redis) -> None:
        self.redis: redis.Redis = conn  # async Redis client
        self.key = settings.REDIS_KEY

    async def save_data(self, features: List[Dict], preds: List[str]):
        """Update features with predictions and push each to a Redis list"""
        for idx, pred in enumerate(preds):
            features[idx]["Churn"] = pred
            result = self.redis.lpush(self.key, json.dumps(features[idx]))
            if isinstance(result, Awaitable):
                await result

    async def retrieve_data(self) -> List[Dict]:
        """Retrieve all JSON objects from Redis list"""
        data = await self.redis.lrange(self.key, 0, -1)   # type: ignore
        return [json.loads(item) for item in data] 

    async def clear_data(self):
        """Clear all items under the Redis key"""
        await self.redis.delete(self.key)

    async def push_database(self, features: List[Dict]):
        """Push raw features to Redis without predictions (optional)"""
        key = settings.REDIS_KEY
        print("saved to my database -- up next")
