# from flask import Flask, request
# import redis
# from rq import Queue
# import os
#
# import time

from flask import Flask, request
# import rq
import jobs


app = Flask(__name__)
jobs.rq.init_app(app)


@app.route("/task")
def add_task():

    if request.args.get("n"):

        job = jobs.background_task.queue(request.args.get("n"))


        return f"Task ({job.id}) added to queue at {job.enqueued_at}"

    return "No value for count provided"


@app.route("/")
def hello_world():
    return "Hello world"


if __name__ == "__main__":
    app.run()
