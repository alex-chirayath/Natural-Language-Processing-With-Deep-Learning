import numpy as np
import argparse
import random
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from math import exp
from nltk.stem.porter import *
import csv
import pickle
import io
import sys



def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

########################### Data Processing Functions ###############################
def clean_review_list(review_data):
    #print "Cleaning the data list..."
    cleaned_review_list=[]
    for review in review_data:
        cleaned_review_list.append(clean_review(review))
    return cleaned_review_list

def check_stopword(word):
    stopwords_list=stopwords.words('english')
    for stopword in stopwords_list:    
        if (word==stopwords):
            return True
    return False

def clean_review(review):
    ps = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    word_list= tokenizer.tokenize(review)
    filtered_words = []
    for word in word_list:
        if ((word) not in stopwords_list):
            try:
                filtered_words.append(ps.stem(unicode(word,'utf-8')))    
            except Exception as e:
                pass 
    return ( " ".join(filtered_words))

def generate_ngrams(review , n):
    words = review.split()
    ngram_list=[]
    if(n>1):
        for i in range(len(words)-n+1):
            ngram_list.append(tuple(words[i:i+n]))
    elif n==1:
        return words        
    return (ngram_list)

def shuffle_dataset(data,label):
    print "Shuffling the dataset..."    
    # Shuffle Dataset
    full_dataset=list(zip(data,label))
    random.shuffle(full_dataset)
    shuffled_data,shuffled_label=zip(*full_dataset)
    return list(shuffled_data),list(shuffled_label)

def split_dataset(data,label):   
    print "Splitting the dataset..."
    #GLobal variables
    final_dataset={}
    dataset_len=len(label)
    train_indexes=(0,int(0.8*dataset_len))
    test_indexes= (train_indexes[1],train_indexes[1]+(dataset_len-train_indexes[1])/2)
    validation_indexes=(test_indexes[1],dataset_len)

    final_dataset['train_data']=data[train_indexes[0]:train_indexes[1]]
    final_dataset['train_label']=label[train_indexes[0]:train_indexes[1]]
    final_dataset['test_data']=data[test_indexes[0]:test_indexes[1]]
    final_dataset['test_label']=label[test_indexes[0]:test_indexes[1]]
    final_dataset['validation_data']=data[validation_indexes[0]:validation_indexes[1]]
    final_dataset['validation_label']=label[validation_indexes[0]:validation_indexes[1]]
    
    with open('train_data.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_dataset['train_data'])
    with open('train_label.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_dataset['train_label'])
    with open('test_data.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_dataset['test_data'])
    with open('test_label.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_dataset['test_label'])
    with open('validation_data.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_dataset['validation_data'])
    with open('validation_label.csv', 'wb') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(final_dataset['validation_label'])


def create_bigram_list(review_list):
    #print "Creating bigrams list for the data..."
    
    bigrams_list=[]
    for review in review_list:
        tokens = nltk.word_tokenize(review)
        
        #Create your bigrams
        bgs = nltk.bigrams(tokens)
        bigrams_list+= (list(bgs))
    return bigrams_list

def create_unigram_list(review_list):
    #print "Creating unigrams list for the data..."
    unigrams_list=[]
    for review in review_list:
        tokens = nltk.word_tokenize(review)
        unigrams_list+= tokens
    return unigrams_list


def print_ngram_list_mostcommon_count(ngrams_list,most_common_count):
    print "Printing most common bigrams in the data..."
    
    #compute frequency distribution for all the bigrams in the text
    fdist = nltk.FreqDist(ngrams_list)
    print fdist.most_common(most_common_count)

def create_ngram_vector_list(review_list,ngram_unique_list,n):
    
    #print "Creating ngram vectors of data..."
    ngram_vector_list=[]
    ngram_list_len=len(ngram_unique_list)
    #print ngram_unique_list
    for review in review_list:
        ngram_vector=[0]*ngram_list_len
        if(n==3): #mix of unigram and bigram
            ngram_list=create_unigram_list([review])+create_bigram_list([review])     
        if(n==2):
            ngram_list=create_bigram_list([review])        
        if(n==1):
            ngram_list=create_unigram_list([review])
        #print ngram_list
        for ngram in ngram_list:
                if ngram in ngram_unique_list:
                    ngram_vector[ngram_unique_list.index(ngram)]+=1
        ngram_vector_list.append(ngram_vector)
        #print ngram_vector
    return ngram_vector_list
    

def get_ngram_list_mostcommon_count(ngrams_list,most_common_count):
    
    print "Finding most common ngrams in the data..."
    
    most_common_ngram_list=[]
    #compute frequency distribution for all the bigrams in the text and get the max
    fdist = nltk.FreqDist(ngrams_list)
    for ngram_count in fdist.most_common(most_common_count):
        most_common_ngram_list.append(ngram_count[0])
    return most_common_ngram_list

def create_final_vector(vector_data,label_data):
    
    features_vector=np.array(vector_data)
    # normalize and standardize
    
    max_column_list=features_vector.max(axis=0)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        features_vector= np.true_divide(features_vector,max_column_list)
        features_vector[ ~ np.isfinite( features_vector )] = 0 
    
    train_label_list= list(label_data)
    final_label_array=np.array([train_label_list])
    final_vector_data=np.concatenate((features_vector,final_label_array.T),axis=1)
    return final_vector_data

def feature_extractor(data,label,final_ngram_list):
    stopwords_unicode_list=stopwords.words('english')
    global stopwords_list
    stopwords_list=[x.encode('UTF8') for x in stopwords_unicode_list]
    cleaned_review_list=clean_review_list(data)
    test_ngram_vector_list=create_ngram_vector_list(data,final_ngram_list,3)
    final_test_data=create_final_vector(test_ngram_vector_list,label)
    return final_test_data
    
################################### End of functions#################################


######################################################## PERCPETRON #########################################################
def perceptron_initialize_weights(train_data):
    print "Random initialization of weights..."
    weight_list=[]
    for i in range(len(train_data[0])):
        weight_list.append(random.uniform(-0.0001,0.0001))
    return weight_list

# Make a prediction with weights
def perceptron_predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

def check_max(my_data_list):
    max_f1=-1
    max_ind=-1
    for my_data_dict in my_data_list:
        if (max_f1 < my_data_dict['f1']):
            max_f1=my_data_dict['f1']
            max_ind=my_data_list.index(my_data_dict)
    print "Found max at epoch " + str(my_data_list[max_ind]['epoch'])    
    return my_data_list[max_ind] 

# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron_train(train,test, l_rate, n_epoch):
    weights = perceptron_initialize_weights(train)
    f1=old_f1=-1
    my_data_list=[]
    for epoch in range(n_epoch):
        for row in train:
            prediction = perceptron_predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        if ( (epoch>30 and  epoch%25==0) ):
        #if (epoch %2==0 and epoch>1):
            my_data_dict={}
            predictions = list()
            test_label=[]
            for row in test:
                test_label.append(row[-1])
                prediction = perceptron_predict(row, weights)
                predictions.append(prediction)
            precision,recall,f1=evaluate(predictions,test_label)
            my_data_dict['epoch']=epoch
            my_data_dict['precision']=precision
            my_data_dict['recall']=recall
            my_data_dict['f1']=f1
            my_data_dict['weights']=weights
            my_data_dict['lrate']=l_rate
            my_data_list.append(my_data_dict)
            print "Epoch Prec Rec F1\t" + str(epoch) +"\t"+str(precision) +"\t" +str(recall) +"\t"+ str(f1)
            if(f1==old_f1):
                print "Breaking because repeated f1"
                break
            old_f1=f1
    return my_data_list

def perceptron_inference(test,weights):
    test_label=[]
    predictions=[]
    for row in test:
        test_label.append(row[-1])
        prediction = perceptron_predict(row, weights)
        predictions.append(prediction)
    return predictions

################################################## MULTI LAYER PERCEPTRON ####################################################
# Initialize a network
def mlp_initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] #random initialization of weights
    network.append(hidden_layer)
    output_layer = [{'weights':[random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] #random initialization of weights
    network.append(output_layer)
    return network
 
# Calculate neuron activation for an input
def mlp_activate(weights, inputs):
    activation = weights[-1]
    for i in range((len(weights)-1)):
        activation += (weights[i] * inputs[i])
    return activation
 

# Forward propagate input to a network output
def mlp_forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = mlp_activate( neuron['weights'], inputs)
            neuron['output'] = 1.0 / (1.0 + exp(-activation)) # using sigmoid function
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Backpropagation of error 
def mlp_backward_error_propogation(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * ((neuron['output'])*(1.0 -(neuron['output']))) #using sigmoid derivative

# Update network weights with error
def mlp_update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Make a prediction with a network
def mlp_predict(network, row):
    outputs = mlp_forward_propagate(network, row)
    return outputs.index(max(outputs))
 
# Train a network for a fixed number of epochs
def mlp_train_network(network, train,test, l_rate, n_epoch, n_outputs):
    print "Training network..."
    my_data_list=[]
    f1=-1
    old_f1=-1
    for epoch in range(n_epoch):
        for row in train:
            outputs = mlp_forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            mlp_backward_error_propogation(network, expected)
            mlp_update_weights(network, row, l_rate)
        if(epoch%25 ==0 and epoch>20):
        #if(epoch>2):
            my_data_dict={}
            my_data_dict['epoch']=epoch
            my_data_dict['lrate']=l_rate
            my_data_dict['network']=network
            predictions=[]
            test_label=[]
            for row in test:
                test_label.append(row[-1])
                predictions.append( mlp_predict(network, row))
            precision,recall,f1=evaluate(predictions,test_label)
            my_data_dict['f1']=f1
            my_data_dict['precision']=precision
            my_data_dict['recall']=recall
            my_data_list.append(my_data_dict)
            print "Epoch Prec Rec F1\t" + str(epoch) +"\t"+str(precision) +"\t" +str(recall) +"\t"+ str(f1)
            if(f1==old_f1):
                print "Breaking because repeated f1"
                break;
            old_f1=f1
    return my_data_list  



def mlp_inference(test_data,network):
    test_label=[]
    predictions=[]
    for row in test_data:
        test_label.append(row[-1])
        predictions.append( mlp_predict(network, row))
    return predictions



###########################################################   MAIN  ############################################################


def main():
    #To run training, change is_train flag to 1
    is_train=0;     #if not train, skip to testing
    argparser = argparse.ArgumentParser()
        
    if(is_train==1):
        with open("sentences.txt") as f:
            data = f.readlines()
        with open("labels.txt") as g:
            labels = [int(label) for label in g.read()[:-1].split("\n")]

        stopwords_unicode_list=stopwords.words('english')
        global stopwords_list
        stopwords_list=[x.encode('UTF8') for x in stopwords_unicode_list]
        cleaned_review_list=clean_review_list(data)
        
        #Shuffle data
        shuffled_data,shuffled_labels=shuffle_dataset(cleaned_review_list,labels)
        #Split data  for train,test,validate data
        split_dataset(shuffled_data,shuffled_labels)
        
        
        #Read in the shuffled data as train,test and validation data sets
        
        with open('train_label.csv', 'rb') as f:
            train_label = list(csv.reader(f, delimiter=','))[0]
        train_label=map(int, train_label)
        with open('test_label.csv', 'rb') as f:
            test_label = list(csv.reader(f, delimiter=','))[0]
        test_label=map(int, test_label)
        with open('validation_label.csv', 'rb') as f:
            validation_label = list(csv.reader(f, delimiter=','))[0]
        validation_label=map(int, validation_label)
        
        with open('validation_data.csv', 'rb') as f:
            validation_data = list(csv.reader(f, delimiter=','))[0]
        with open('test_data.csv', 'rb') as f:
            test_data = list(csv.reader(f, delimiter=','))[0]
        with open('train_data.csv', 'rb') as f:
            train_data = list(csv.reader(f, delimiter=','))[0]
                
        #Create unigram list
        
        unigram_list=create_unigram_list(train_data)
        perc_final_unigram_list=get_ngram_list_mostcommon_count(unigram_list,7500)
        mlp_final_unigram_list=get_ngram_list_mostcommon_count(unigram_list,2500)
                

        #Create bigram list
        
        bigram_list=create_bigram_list(train_data)
        mlp_final_bigram_list=get_ngram_list_mostcommon_count(bigram_list,2500)
        perc_final_bigram_list=get_ngram_list_mostcommon_count(bigram_list,250)
        
        #Create the final ngram combination list of bigram and uigram
        perc_final_ngram_list= perc_final_unigram_list+perc_final_bigram_list
        mlp_final_ngram_list= mlp_final_unigram_list+mlp_final_bigram_list
        
        perc_train_ngram_vector_list=create_ngram_vector_list(train_data,perc_final_ngram_list,3)
        perc_final_train_data= create_final_vector(perc_train_ngram_vector_list,train_label)
        
        perc_test_ngram_vector_list=create_ngram_vector_list(test_data,perc_final_ngram_list,3)
        perc_final_test_data=create_final_vector(perc_test_ngram_vector_list,test_label )
        
        perc_validation_ngram_vector_list=create_ngram_vector_list(validation_data,perc_final_ngram_list,3)
        perc_final_validation_data=create_final_vector(perc_validation_ngram_vector_list,validation_label )
        


        mlp_train_ngram_vector_list=create_ngram_vector_list(train_data,mlp_final_ngram_list,3)
        mlp_final_train_data= create_final_vector(mlp_train_ngram_vector_list,train_label)
        
        mlp_test_ngram_vector_list=create_ngram_vector_list(test_data,mlp_final_ngram_list,3)
        mlp_final_test_data=create_final_vector(mlp_test_ngram_vector_list,test_label )
        
        mlp_validation_ngram_vector_list=create_ngram_vector_list(validation_data,mlp_final_ngram_list,3)
        mlp_final_validation_data=create_final_vector(mlp_validation_ngram_vector_list,validation_label )
        

        
        lrate=0.5   # Fixed learning rate of 0.5 
        
        #MLP PARAMETERS
        n_inputs = len(mlp_final_train_data[0]) - 1
        n_outputs = 2
        n_hidden = 5#n_inputs/8
        network = mlp_initialize_network(n_inputs, n_hidden, n_outputs)
        
        my_mlp_data_list=mlp_train_network(network, mlp_final_train_data,mlp_final_test_data, lrate, 101, n_outputs)#101
        my_mlp_final_dict=check_max(my_mlp_data_list)

        my_perc_data_list=perceptron_train(perc_final_train_data,perc_final_test_data,lrate,251) #251)
        my_perc_final_dict=check_max(my_perc_data_list)


        #Store the list as a pickle
        mlp={}
        perceptron={}
        mlp['network']=my_mlp_final_dict['network']
        mlp['final_ngram_list']=mlp_final_ngram_list
        perceptron['weights']=my_perc_final_dict['weights']
        perceptron['final_ngram_list']=perc_final_ngram_list
        

        deep_learn=(mlp,perceptron)
        f = open('my_deep_learnx.pickle','w')
        pickle.dump(deep_learn,f)
        f.close()

#######################################Testing Phase ##########################3
    else :
        #Load a pickle  
        
        with open("my_deep_learn.pickle", "rb") as input_file:
            deep_learn = pickle.load(input_file)
            
        mlp,perceptron=deep_learn
        
        """
    	Testing on unseen testing data in grading
    	"""
    	argparser.add_argument("--test_data", type=str, default="../test_sentences.txt", help="The real testing data in grading")
    	argparser.add_argument("--test_labels", type=str, default="../test_labels.txt", help="The labels for the real testing data in grading")
    	parsed_args = argparser.parse_args(sys.argv[1:])
    	real_test_sentences = parsed_args.test_data
    	real_test_labels = parsed_args.test_labels
    	with io.open(real_test_sentences,encoding="latin-1") as f:
    		real_test_x = f.readlines()
    	with open(real_test_labels) as g:
    		real_test_y = g.readlines()

    	data=real_test_x
    	labels=[int(label.strip()) for label in real_test_y]
        
        
        weights=perceptron['weights']
        perc_final_ngram_list=perceptron['final_ngram_list']
        
        
        network=mlp['network']
        mlp_final_ngram_list=mlp['final_ngram_list']

        mlp_final_test_data=feature_extractor(data,labels,mlp_final_ngram_list)
        mlp_predictions=mlp_inference(mlp_final_test_data,network)
        
        perc_final_test_data=feature_extractor(data,labels,perc_final_ngram_list)
        perc_predictions=perceptron_inference(perc_final_test_data,weights)
        
        precision,recall,f1=evaluate(mlp_predictions,labels)
        print "MLP Results:" , precision,recall,f1
        
        precision,recall,f1=evaluate(perc_predictions,labels)
        print "Perceptron Results:" , precision,recall,f1
        

        
        

if __name__ == '__main__':
    main()
