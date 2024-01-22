import time
from datetime import datetime
import math
import redis



def how_long_until_data_evicted():
    elasticache_endpoint = "ec2-54-200-1-251.us-west-2.compute.amazonaws.com"
    elasticache_port = 6378
    # Create a Redis connection
    redis_client = redis.StrictRedis(host=elasticache_endpoint, port=elasticache_port)

    # Example: Set a key-value pair
    redis_client.set("example_key90", "example_value")
    
    # Sleep without invoking the Lambda function
    sleeptime = 60*5
    end = time.time()

    while redis_client.get("example_key90") is not None:
        print('Retrieved Value. Time elapsed(s): {}'.format(time.time() - end))
        end =  time.time()
        time.sleep(sleeptime)
        sleeptime = sleeptime*2
    
    
    #reach here when data is no longer in the cache
    time_until_eviction = time.time() - end
    print('Data Evicted. Time elapsed(s): {}'.format(time_until_eviction))
    return time_until_eviction


if __name__ == "__main__":
    evition_duarations = []    

    for i in range(10):

        evition_duarations.append(how_long_until_data_evicted())

    print(evition_duarations)
    
