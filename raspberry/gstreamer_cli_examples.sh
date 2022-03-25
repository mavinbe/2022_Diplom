#!/usr/bin/env bash

# show rtsp
gst-launch-1.0 rtspsrc location=rtsp://malte:diplom@192.168.0.105:554//h264Preview_07_main latency=100 ! queue ! rtph264depay ! avdec_h264 ! videoconvert ! videoscale  ! autovideosink


# send test
gst-launch-1.0 videotestsrc ! videoconvert ! videorate ! video/x-raw,width=1280,height=720,framerate=10/1 ! jpegenc ! rtpjpegpay ! udpsink host=192.168.0.0 port=5000


# receive
# gst-launch-1.0 udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, framerate=30/1, payload=26, clock-rate=90000 ! rtpjpegdepay ! jpegdec ! videoconvert ! autovideosink