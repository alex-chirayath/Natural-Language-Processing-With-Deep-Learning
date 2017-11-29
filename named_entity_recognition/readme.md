Named ENtity Recognition using Deep MEMM and Viterbi


Deep Maximum Entropy Markov Model
(DMEMM). DMEMM extends MEMM by using a neural network to build the conditional probability.
Implementation of the deep memm is avaiable in the source

A]There are two implementations of the training here:
1. We use an online training. However, this did not give good performance since there are too many O tags and thus the network was biased toward outputting O tags
2. Batch Training significantly improved te performance and overcame the drawback of bias in online training

B]There are two implementations of input vectors:
1. Giving the i/p as [current word embedding,prev label embedding]. The f1 score was around 0.66 with this implementation
2. Giving the i/p as [prev word embedding,current word embedding, prev label embedding]. THis is kind of a bigram implementation and this helped getting in some context for the tagging(especially for the location tags since it was now easier to differentitate between from-location and to-location)

C]There are also two implementations of the testing network
1. Using greedy-viterbi approach which considers only the maximum to find the labels for the consecutive word labels.  #implemented only for B1
2. Using Viterbi to find the optimal answer. This improved the f1 score by a factor of more than 0.05. Thus, it was a better approach, but is more time consuming #implemented for B2

We have a flag is_train which is 1 for training and 0 for testing

The implementation used by default is:
is_train =0 which means it is on test mode
A2 : Batch Training with batch size 10,000
B2 : i/p vector consists [prev word embedding,current word embedding, prev label embedding]
C2 : Viterbi Implementation to get optimal answer

The standard training time is 5-10 minutes and testing time is 15-30 minutes
