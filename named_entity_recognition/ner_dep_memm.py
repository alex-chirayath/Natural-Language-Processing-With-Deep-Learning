import subprocess
import argparse
import sys
import gzip
import cPickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
import numpy
import time
global EMBED_SIZE
EMBED_SIZE=8

is_train=0

global EPOCHS
EPOCHS=80

global output_embed_size
output_embed_size=7 
no_of_labels=127 # we know this since the idx2label has no_of_labels entries

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(output_embed_size+2*EMBED_SIZE, 150)
        self.fc2 = nn.Linear(150, no_of_labels)
    
    def forward(self, x):
	    x = F.relu(self.fc1(x))
	    x = self.fc2(x)
	    return F.log_softmax(x)

def create_word_embed_dict(idx2entity):
	"""
	Create dict with same keys but values are embedding
	"""
	idx2entity_dict={}
	dict_keys=idx2entity.keys()
	dict_len=len(idx2entity)
	embeds=nn.Embedding(dict_len, EMBED_SIZE)
	
	for key in dict_keys:
			lookup_tensor = torch.LongTensor([key])		
			word_embed = embeds(autograd.Variable(lookup_tensor))
			idx2entity_dict[key]=word_embed
			
	return idx2entity_dict

def toBinary(n):
	"""
	Convert the number to its binary in a list of integers form
	"""
	x= ''.join(str(1 & int(n) >> i) for i in range(64)[::-1])
	x=x[-7:]
	x=[int(y) for y in x]
	return x

def create_output_embed_dict(idx2entity):
	"""
	Create embedding of the labels
	"""

	idx2entity_dict={}
	dict_keys=idx2entity.keys()
	dict_len=len(idx2entity)

	if (output_embed_size==no_of_labels):#one hot vector representation if size is no_of_labels
		for key in dict_keys:
			vector=[0]*dict_len
			vector[key]=1
			idx2entity_dict[key]=torch.FloatTensor([vector])
			#print idx2entity_dict[key]
	if (output_embed_size==7): #convert to binary representation
		for key in dict_keys:
			vector=toBinary(key)
			idx2entity_dict[key]=torch.FloatTensor([vector])
			
	return idx2entity_dict


def create_startlabel_embed():
	"""
	Create embeding for arbitrary start label
	"""
	if (output_embed_size==output_embed_size):
		return torch.zeros(output_embed_size)
	if (output_embed_size==7):
		return torch.LongTensor([1]*7)

def create_arbitrary_word_embed():
	"""
	Create embediing for arbitrary start/end word
	"""
	return torch.zeros(EMBED_SIZE)
	
def nnet_train_online(network,train_lex,train_y,idx2word_embed_dict,idx2label_out_embed,learning_rate,momentum):
	
	#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
	optimizer = optim.Adadelta(network.parameters(), lr=learning_rate, momentum=momentum)
	
	# create a loss function
	criterion = nn.NLLLoss()
	start_label=torch.zeros(output_embed_size)
	word_count=0

	for epoch in range(EPOCHS):
		for i in range(len(train_lex)):
			for j in range(len(train_lex[i])):
				current_word_embed=idx2word_embed_dict[train_lex[i][j]]
				if (j==0):
					word_embed=torch.cat((current_word_embed.data,( start_label.view(1,output_embed_size) )),1)	
				else :
					word_embed=torch.cat((current_word_embed.data, (idx2label_out_embed[train_y[i][j-1]].view(1,output_embed_size))),1)	
				optimizer.zero_grad()
		        word_embed= autograd.Variable(word_embed)
		        net_out = network(word_embed)
		        y_embed=autograd.Variable(torch.LongTensor([train_y[i][j].tolist()]))
		        loss = criterion(net_out,y_embed)
		        loss.backward()
        		optimizer.step()
        		

	return network

def create_batch(train_lex,train_y,idx2word_embed_dict,idx2label_out_embed):
	"""
	Sub function for batch training
	Uses embeding [word + previous word] instead of just current word to make a vector of 2*EMBED SIZE
	Output embedding is then concatenated to the previously formed vector
	"""
	start_label=create_startlabel_embed()
	batch_word=[]
	for i in range(len(train_lex)):
			for j in range(len(train_lex[i])):
				current_word_embed=idx2word_embed_dict[train_lex[i][j]]
				if (j==0):
					bigram_word_embed=torch.cat((create_arbitrary_word_embed().view(1,EMBED_SIZE),current_word_embed.data),1)	
					word_embed=torch.cat((bigram_word_embed,( start_label.view(1,output_embed_size) )),1)	
				else :
					bigram_word_embed=torch.cat((idx2word_embed_dict[train_lex[i][j-1]].data,current_word_embed.data),1)	
					word_embed=torch.cat((bigram_word_embed, (idx2label_out_embed[train_y[i][j-1]].view(1,output_embed_size))),1)	
				
				if (i==0 and j==0):
					batch_word_embed=word_embed
					y_batch_embed=(torch.LongTensor([train_y[i][j].tolist()]))
					
				else :
					batch_word_embed=torch.cat((batch_word_embed,word_embed),0)
					y_batch_embed=torch.cat((y_batch_embed,torch.LongTensor([train_y[i][j].tolist()])),0)
					#print y_batch_embed

	torch.save(batch_word_embed,'batch_word_embed_bin.pt')
	torch.save(y_batch_embed,'y_batch_embed_bin.pt')

def nnet_train_batch(train_lex,train_y,idx2word_embed_dict,idx2label_out_embed,learning_rate,momentum,batch_size):
	"""
	Main function to carry out training of the network in batches
	"""

	network=NNet()
	
	#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
	#optimizer = optim.Adadelta(network.parameters(), lr=learning_rate,eps=1e-06, weight_decay=0)
	optimizer = optim.Adam(network.parameters(), lr=learning_rate)
	
	# create a loss function
	criterion = nn.NLLLoss()

	create_batch(train_lex,train_y,idx2word_embed_dict,idx2label_out_embed)
	
	batch_word_embed=torch.load('batch_word_embed_bin.pt')
	y_batch_embed=torch.load('y_batch_embed_bin.pt')
	
	for epoch in range(EPOCHS):
		for i in range(0,len(batch_word_embed),batch_size):
			batch_word=batch_word_embed[i:i+batch_size]
			batch_y=y_batch_embed[i:i+batch_size]
			optimizer.zero_grad()
			batch_word= autograd.Variable(batch_word)
			net_out = network(batch_word)
			batch_y=autograd.Variable(batch_y)
			loss = criterion(net_out,batch_y)
			loss.backward()
			optimizer.step()
			
	return network

def greedy_nnet_test(network,test_lex,idx2label_out_embed,idx2word_embed_dict,idx2label):
	"""
	Uses greedy approch to find max probability label for each word 
	Current implementation works only for the current word appraoch(needs to be modified to include bigram/trigram case)
	"""
	data_prediction=[]
	for i in range(len(test_lex)):
		sentence_prediction=[]
		for k in range(len(test_lex[i])):
			if (k==0):
				prev_label_embed=create_startlabel_embed()
			current_word_embed=idx2word_embed_dict[test_lex[i][k]]
			#print test_lex[i][k]
			word_embed = torch.cat((current_word_embed.data, prev_label_embed.view(1,output_embed_size) ),1)
			#print word_embed
			#print test_lex[i][k]
			net_2 = network(autograd.Variable(word_embed))
			#print net_2
			prob,key=net_2.topk(5)
			label_index=key[0,0].data.numpy()[0]
			prev_label_embed=idx2label_out_embed[label_index]
			sentence_prediction.append(label_index)
	return sentence_prediction


def print_matrix(my_matrix):
	for i in range(len(my_matrix)):
		for j in range(len(my_matrix[i])):
			print my_matrix[i][j]

def viterbi_back_track(probab_matrix,back_matrix):
	"""
	Backtrack process for the viterbi implementation
	"""
	j =len(probab_matrix[0])-1
	output=[]
	max_row=probab_matrix[:,j].argmax(axis=0)
	output.append(max_row)
	while (j>0):
		prev_label=int(back_matrix[max_row][j])
		output.append(prev_label)
		max_row=prev_label
		j-=1
	output.reverse()
	return output

def viterbi_nnet_test(network,test_lex,idx2label_out_embed,idx2word_embed_dict,idx2label):
	"""
	Testing function
	Using Viterbi Implementation
	Uses bigram approach [previous word+current word]
	"""
	for i in range(len(test_lex)):
		probab_matrix=numpy.zeros((no_of_labels,len(test_lex[i]) ))
		back_matrix=numpy.zeros((no_of_labels,len(test_lex[i]) ))
		for k in range(len(test_lex[i])):
			current_word_embed=idx2word_embed_dict[test_lex[i][k]]
			if (k==0):
				prev_label_embed=create_startlabel_embed()
				bigram_word_embed=torch.cat((create_arbitrary_word_embed().view(1,EMBED_SIZE),current_word_embed.data),1)	
				word_embed = torch.cat((bigram_word_embed, prev_label_embed.view(1,output_embed_size) ),1)
				net_2 = network(autograd.Variable(word_embed))
				probab_matrix[:,0]=net_2.data.view(no_of_labels).numpy()
				for l in range(no_of_labels):
					back_matrix[l,0]=l
			else :
				for l in range(no_of_labels): #to traverse every label
					prev_label_embed=idx2label_out_embed[l]
					bigram_word_embed=torch.cat((idx2word_embed_dict[test_lex[i][k-1]].data,current_word_embed.data),1)	
					word_embed = torch.cat((bigram_word_embed, prev_label_embed.view(1,output_embed_size) ),1)
					net_2 = network(autograd.Variable(word_embed))
					if (l == 0):
						probab_matrix[:,k]=net_2.data.view(no_of_labels).numpy() + probab_matrix[l,k-1]
						back_matrix[:,k]=l
					else:
						for n in range(len(probab_matrix)):
							if ( probab_matrix[n][k]< probab_matrix[l][k-1]+ net_2.data.view(no_of_labels).numpy()[n] ):
								probab_matrix[n][k]=probab_matrix[l][k-1]+ net_2.data.view(no_of_labels).numpy()[n]
								back_matrix[n][k]=l

	#print(back_matrix)
	return viterbi_back_track(probab_matrix,back_matrix)

def conlleval(p, g, w, filename='tempfile.txt'):
    '''
    INPUT:
    p :: predictions
    g :: groundtruth
    w :: corresponding words

    OUTPUT:
    filename :: name of the file where the predictions
    are written. it will be the input of conlleval.pl script
    for computing the performance in terms of precision
    recall and f1 score
    '''
    out = ''
    for sl, sp, sw in zip(g, p, w):
        out += 'BOS O O\n'
        for wl, wp, ww in zip(sl, sp, sw):
            out += ww + ' ' + wl + ' ' + wp + '\n'
        out += 'EOS O O\n\n'

    f = open(filename, 'w')
    f.writelines(out)
    f.close()

    return get_perf(filename)

def get_perf(filename):
    ''' run conlleval.pl perl script to obtain precision/recall and F1 score '''
    _conlleval = 'conlleval.pl'

    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break

    precision = float(out[6][:-2])
    recall    = float(out[8][:-2])
    f1score   = float(out[10])

    return (precision, recall, f1score)


def run_hyperparameter_tuning(train_lex,train_y,test_lex,test_y,dicts):
	learning_rate_array=[0.06]
	momentum_array=[0.01]#[0,0.001,0.1]
	batch_size_array=[2000]#[15000,10000]
	idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
	idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())
	idx2label_out_embed= create_output_embed_dict(idx2label)
	idx2word_embed_dict= create_word_embed_dict(idx2word)
	groundtruth_test = [ map(lambda t: idx2label[t], y) for y in test_y ]
	words_test = [ map(lambda t: idx2word[t], w) for w in test_lex ]
	print "LRate\tMomentum\tBSize\tPreci\tRecall\tF1"
	for bs in batch_size_array:
		for lr in learning_rate_array:
			for mom in momentum_array:
				network2= nnet_train_batch(train_lex,train_y,idx2word_embed_dict,idx2label_out_embed,lr,mom,bs)
				predictions_test = [ map(lambda t: idx2label[t], viterbi_nnet_test(network2,[y],idx2label_out_embed,idx2word_embed_dict,idx2label)) for y  in test_lex ]
				test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)
				print str(lr) +"\t"+str(mom)+"\t"+str(bs)+"\t" +str(test_precision) +"\t"+str(test_recall)+"\t"+str(test_f1score)


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", type=str, default="atis.small.pkl.gz", help="The zipped dataset")

    parsed_args = argparser.parse_args(sys.argv[1:])

    filename = parsed_args.data
    f = gzip.open(filename,'rb')
    train_set, valid_set, test_set, dicts = cPickle.load(f)

    train_lex, _, train_y = train_set
    valid_lex, _, valid_y = valid_set
    test_lex,  _,  test_y  = test_set

    idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())
    
    ##### Tuning the parameters #####
    #run_hyperparameter_tuning(train_lex,train_y,valid_lex,valid_y,dicts)
    
    #Training
    if is_train==1:
    	idx2label_out_embed= create_output_embed_dict(idx2label)
    	idx2word_embed_dict= create_word_embed_dict(idx2word)
    	torch.save(idx2label_out_embed,'idx2label_out_embed.pt')
    	torch.save(idx2word_embed_dict,'idx2word_embed_dict.pt')
    	network2= nnet_train_batch(train_lex,train_y,idx2word_embed_dict,idx2label_out_embed,0.06,0.01,10000)
    	if (output_embed_size==no_of_labels):
    		torch.save(network2,'network1.pt')
    	if (output_embed_size==7):
    		torch.save(network2,'main_network.pt')
    	#print "Completed Training"
    else:
    	start_time = time.time()
    	idx2label_out_embed=torch.load('idx2label_out_embed.pt')
    	idx2word_embed_dict=torch.load('idx2word_embed_dict.pt')
    	network2=torch.load('main_network.pt')
    	groundtruth_test = [ map(lambda t: idx2label[t], y) for  y in valid_y ]
    	words_test = [ map(lambda t: idx2word[t], w) for w in valid_lex ]
    	predictions_test = [ map(lambda t: idx2label[t], viterbi_nnet_test(network2,[y],idx2label_out_embed,idx2word_embed_dict,idx2label)) for y  in valid_lex ]
    	test_precision, test_recall, test_f1score = conlleval(predictions_test, groundtruth_test, words_test)
    	print test_precision, test_recall, test_f1score	
    	minutes_to_test= (time.time() - start_time)/60
    	print minutes_to_test
    	
if __name__ == '__main__':
    main()
