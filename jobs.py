from flask_rq2 import RQ
import time

rq = RQ()
rq.redis_url = 'redis://redis:6379/0'


@rq.job(timeout=180)
def background_task(msg):
    """ lol
    """

    delay = 2

    print("Task running")
    print(f"Simulating a {delay} second delay")

    time.sleep(delay)
    result = msg + '-lol'
    print(result)
    print("Task complete")

    return result
