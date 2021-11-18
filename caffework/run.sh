#!/bin/bash
rm -rf bnn_train_iter*
caffe train -gpu 7 -solver solver.prototxt 2>&1 | tee log/fall_bnn.log
cd log
if [ -f fall_bnn.log ]; then
    rm -f accuracy.png  loss.png fall_bnn.log.test  fall_bnn.log.train 
    /usr/local/caffe/tools/extra/parse_log.sh fall_bnn.log
    gnuplot accuracy.gnuplot
    gnuplot loss.gnuplot
    cd ..
fi
