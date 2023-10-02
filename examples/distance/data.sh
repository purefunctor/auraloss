#!/usr/bin/bash

aws s3 cp --recursive s3://meeshkan-datasets/distance_augment/ ./data

cd data
tar -xvf day1_unsilenced.tar.gz --no-same-owner && rm day1_unsilenced.tar.gz
tar -xvf day2_unsilenced.tar.gz --no-same-owner && rm day2_unsilenced.tar.gz
