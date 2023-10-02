#!/usr/bin/bash

python train.py --nblocks=6 --dilation_growth=3 --kernel_size=20 --channel_width=32 --half  # 103841
python train.py --nblocks=10 --dilation_growth=2 --kernel_size=13 --channel_width=32 --half  # 121537
python train.py --nblocks=10 --dilation_growth=2 --kernel_size=15 --channel_width=32 --half  # 140033
python train.py --nblocks=7 --dilation_growth=3 --kernel_size=25 --channel_width=32 --half  # 155329
python train.py --nblocks=8 --dilation_growth=2 --kernel_size=25 --channel_width=32 --half  # 181057


python train.py --nblocks=5 --dilation_growth=5 --kernel_size=25 --channel_width=32 --half  # 103873
python train.py --nblocks=7 --dilation_growth=3 --kernel_size=20 --channel_width=32 --half  # 124449
python train.py --nblocks=4 --dilation_growth=10 --kernel_size=5 --channel_width=96 --half  # 140353
python train.py --nblocks=4 --dilation_growth=10 --kernel_size=13 --channel_width=64 --half  # 161665
python train.py --nblocks=10 --dilation_growth=2 --kernel_size=20 --channel_width=32 --half  # 186273


python train.py --nblocks=6 --dilation_growth=5 --kernel_size=5 --channel_width=64 --half  # 104321
python train.py --nblocks=9 --dilation_growth=2 --kernel_size=15 --channel_width=32 --half  # 124545
python train.py --nblocks=8 --dilation_growth=2 --kernel_size=20 --channel_width=32 --half  # 145057
python train.py --nblocks=9 --dilation_growth=2 --kernel_size=20 --channel_width=32 --half  # 165665
python train.py --nblocks=4 --dilation_growth=10 --kernel_size=15 --channel_width=64 --half  # 186369


python train.py --nblocks=9 --dilation_growth=2 --kernel_size=13 --channel_width=32 --half  # 108097
python train.py --nblocks=6 --dilation_growth=3 --kernel_size=25 --channel_width=32 --half  # 129601
python train.py --nblocks=8 --dilation_growth=3 --kernel_size=5 --channel_width=64 --half  # 145793
python train.py --nblocks=9 --dilation_growth=3 --kernel_size=5 --channel_width=64 --half  # 166529
