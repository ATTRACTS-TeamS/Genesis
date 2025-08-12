#!/usr/bin/env sh

tensorboard --logdir "${1:-./logs}"
