#!/bin/bash

yyyymmdd=`date '+%Y-%m-%d'`
tail -f "../app.data/server/intentserver.log.$yyyymmdd"

