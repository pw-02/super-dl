import redis

# Replace these values with your actual ElastiCache endpoint and port
elasticache_endpoint = "ec2-34-210-71-230.us-west-2.compute.amazonaws.com"
elasticache_port = 6378

# Create a Redis connection
redis_client = redis.StrictRedis(host=elasticache_endpoint, port=elasticache_port)

# Example: Set a key-value pair
redis_client.set("example_key", "example_value")

# Example: Get the value for a key
value = redis_client.get("example_key")
print("Value for 'example_key':", value)
