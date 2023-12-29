import redis


def test_redis_connection(redis_host, redis_port):
    try:
        # Create a Redis client
        redis_client = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)

        # Ping the Redis server
        response = redis_client.ping()

        if response == True:
            print("Successfully connected to the Redis server.")
        else:
            print("Failed to connect to the Redis server.")
    except Exception as e:
        print(f"Error connecting to Redis: {e}")

if __name__ == "__main__":
    # Update with your Redis server information
    redis_host = 'localhost'
    redis_port = 6379

    test_redis_connection(redis_host, redis_port)
