#!/bin/sh
xvfb-run --auto-servernum --server-args='-screen 0 640x480x24 -ac +extension GLX +render -noreset' ./build.x86_64 "$@" -logFile log.log
