
import mozg.common.util.Log as lg
from ie.api.IntentApi import app, Start_Intent_Engine


#
# If Gunicorn gives too many problems (worker threads rebooting, etc.)
# another option is to run multiple intent engines on multiple ports,
# then point the BiBot to load balance on them
#
intent_engine = Start_Intent_Engine()

if __name__ == "__main__":
    lg.Log.log("Starting intent engine..")
    app.run(
        host = '0.0.0.0',
        port = intent_engine.port
    )