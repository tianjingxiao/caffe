# These snippets serve only as basic examples.
# Customization is a must.
# You can copy, paste, edit them in whatever way you want.
# Be warned that the fields in the training log may change in the future.
# You had better check the data files before designing your own plots.

# Please generate the necessary data files with 
# /path/to/caffe/tools/extra/parse_log.sh before plotting.
# Example usage: 
#     ./parse_log.sh mnist.log
# Now you have mnist.log.train and mnist.log.test.
#     gnuplot mnist.gnuplot

# The fields present in the data files that are usually proper to plot along
# the y axis are test accuracy, test loss, training loss, and learning rate.
# Those should plot along the x axis are training iterations and seconds.
# Possible combinations:
# 1. Test accuracy (test score 0) vs. training iterations / time;
# 2. Test loss (test score 1) time;
# 3. Training loss vs. training iterations / time;
# 4. Learning rate vs. training iterations / time;
# A rarer one: Training time vs. iterations.

# What is the difference between plotting against iterations and time?
# If the overhead in one iteration is too high, one algorithm might appear
# to be faster in terms of progress per iteration and slower when measured
# against time. And the reverse case is not entirely impossible. Thus, some
# papers chose to only publish the more favorable type. It is your freedom
# to decide what to plot.

reset
set terminal png

set style data lines
set key right

###### Fields in the data file your_log_name.log.train are
###### Iters Seconds TrainingLoss LearningRate

# Training loss vs. training iterations
set output "loss.png"
set title "Training loss vs. training iterations"
set xlabel "Training iterations"
set ylabel "Training loss"
plot "fall_bnn.log.train" using 1:3 title "fall bnn"

# Training loss vs. training time
# plot "fall_bnn.log.train" using 2:3 title "fall\_bnn"

# Learning rate vs. training iterations;
# plot "fall_bnn.log.train" using 1:4 title "fall\_bnn"

# Learning rate vs. training time;
# plot "fall_bnn.log.train" using 2:4 title "fall\_bnn"


###### Fields in the data file your_log_name.log.test are
###### Iters Seconds TestAccuracy TestLoss

# Test loss vs. training iterations
# plot "fall_bnn.log.test" using 1:4 title "fall\_bnn"

# Test accuracy vs. training iterations
#set output "accuracy.png"
#set title "Test accuracy vs. training iterations"
#set xlabel "Training iterations"
#set ylabel "Training accuracy"
#plot "fall_bnn.log.test" using 1:3 title "fall bnn"

# Test loss vs. training time
# plot "fall_bnn.log.test" using 2:4 title "fall\_bnn"

# Test accuracy vs. training time
# plot "fall_bnn.log.test" using 2:3 title "fall\_bnn"
