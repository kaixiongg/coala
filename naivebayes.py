#kaixiong
#uniq:kaixiong
import os, sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import operator
import math
import string

truth_docs = dict()
lie_docs = dict()

#0 for lie and 1 for truth!

punctation_stopwords = list(string.punctuation)
punctation_stopwords.extend(stopwords.words('english'))

def preprocess(doc):
    #basic preprocess!
    doc = nltk.word_tokenize(doc)
    #ps = PorterStemmer()
   # doc = [word for word in doc if word not in list(string.punctuation) ]
    #for removing stopword
    #doc = [word for word in doc if word not in punctation_stopwords ]
    #output_str = []
    #for word in doc:
    #   output_str.append(ps.stem(word))
    
    return doc
    
#the input is the content of train docs instead of list! SINCE I cache them into memory first! to save time
#return the datastructure for testing, two dict each word -> times
def trainNaiveBayes(iter_train_lie_docs, iter_train_truth_docs):

    train_lie = dict()
    train_truth = dict()  
    unique_word = dict()
    words_lie, words_truth = 0, 0
    for key in iter_train_lie_docs:
            for j in range(len(iter_train_lie_docs[key])):
                words_lie = words_lie + 1
                if iter_train_lie_docs[key][j] not in train_lie:
                    train_lie[iter_train_lie_docs[key][j]] = 1
                else:            
                    train_lie[iter_train_lie_docs[key][j]] = train_lie[iter_train_lie_docs[key][j]] + 1
                #calculate for # of unique word!!!
                if iter_train_lie_docs[key][j] not in unique_word:
                    unique_word[iter_train_lie_docs[key][j]] = 1
                else:            
                    unique_word[iter_train_lie_docs[key][j]] = unique_word[iter_train_lie_docs[key][j]] + 1
                    
    for key in iter_train_truth_docs:
        for j in range(len(iter_train_truth_docs[key])):
            words_truth = words_truth + 1
            if iter_train_truth_docs[key][j] not in train_truth:
                train_truth[iter_train_truth_docs[key][j]] = 1
            else:
                train_truth[iter_train_truth_docs[key][j]] = train_truth[iter_train_truth_docs[key][j]] + 1
                
            #calculate for # of unique word!!!
            if iter_train_truth_docs[key][j] not in unique_word:
                unique_word[iter_train_truth_docs[key][j]] = 1
            else:            
                unique_word[iter_train_truth_docs[key][j]] = unique_word[iter_train_truth_docs[key][j]] + 1
                

    return train_lie, train_truth, len(unique_word), words_lie, words_truth     
                
def testNaiveBayes(file_name,doc_folder, train_lie, train_truth, num_lie, num_truth, total_vol, words_lie, words_truth):  

    be_tested = preprocess(open(os.path.join(doc_folder, file_name), 'r').read())
    prob_lie, prob_truth = 0, 0
    total_doc = num_lie + num_truth
    prob_lie, prob_truth = num_lie/total_doc, num_truth/total_doc
    
    condition_lie = dict()
    condition_truth = dict()
    for i in range(len(be_tested)):
        if(be_tested[i] in train_lie):   
            prob_lie = prob_lie + math.log((train_lie[be_tested[i]] + 1)/(words_lie + total_vol))
            
            condition_lie[be_tested[i]] = (train_lie[be_tested[i]] + 1)/(words_lie + total_vol)
        else:
            prob_lie = prob_lie + math.log((1)/(words_lie + total_vol))

            condition_lie[be_tested[i]] = ( 1)/(words_lie + total_vol)
    for i in range(len(be_tested)):
        
        if(be_tested[i] in train_truth):   
            prob_truth = prob_truth + math.log((train_truth[be_tested[i]] + 1)/(words_truth + total_vol))

            condition_truth[be_tested[i]] = (train_truth[be_tested[i]] + 1)/(words_truth + total_vol)
        else:
            prob_truth = prob_truth + math.log((1)/(words_truth + total_vol))

            condition_truth[be_tested[i]] =  (1)/(words_truth + total_vol)
    
    return prob_lie <= prob_truth, condition_lie, condition_truth



def main(doc_folder):
    sys.stdout = open('naivebayes.output', 'w')
    input_dir = os.listdir(doc_folder)
    global truth_docs 
    global lie_docs 
    #read the input into memory!
    num_lie, num_truth = 0, 0
    for i in range(len(input_dir)):
        content = open(os.path.join(doc_folder, input_dir[i]), 'r')
        if (input_dir[i][0] == 'l'):
           base, ext = os.path.splitext(input_dir[i])    
           lie_docs[base[3:]] = preprocess(content.read())
           num_lie = num_lie + 1
        elif (input_dir[i][0] == 't'):
           base, ext = os.path.splitext(input_dir[i])
           truth_docs[base[4:]] = preprocess(content.read())
           num_truth = num_truth + 1
    #get started

    correct = 0
   
    average_cond_lie, average_cond_truth = dict(), dict()
    
    for i in range(len(input_dir)):
        train_lie, train_truth = dict(), dict()

        predict = -1
        if (input_dir[i][0] == 'l'):
           base, ext = os.path.splitext(input_dir[i])    
           #get rid of will_be_test doc 
           iter_lie_docs = lie_docs.copy()
           iter_lie_docs.pop(base[3:], None)
           
           #the input is the content of train files instead of the list, SINCE I cache them into memory to save time!
           train_lie, train_truth, total_vol, words_lie, words_truth  =  trainNaiveBayes(iter_lie_docs, truth_docs)
           predict, current_cond_lie, current_cond_truth =  testNaiveBayes(input_dir[i],doc_folder, train_lie, train_truth,
                          num_lie - 1, num_truth,
                          total_vol,words_lie, words_truth) 
           
           #for calculating the condional prob
           if (len(average_cond_lie) == 0):
               average_cond_lie = current_cond_lie
           else:
               for key in current_cond_lie:
                   if key in average_cond_lie:
                       average_cond_lie[key] = average_cond_lie[key] + (1/i)*(current_cond_lie[key] - average_cond_lie[key])
                   else:
                       average_cond_lie[key] = current_cond_lie[key] 
                       
           if (predict == 0):
               correct = correct + 1
               print(input_dir[i],'lie')
           else:
               print(input_dir[i],'truth')
           #print (predict == 0)
        elif (input_dir[i][0] == 't'):
           base, ext = os.path.splitext(input_dir[i])
           #get rid of will_be_test doc 
           iter_truth_docs = truth_docs.copy()
           iter_truth_docs.pop(base[4:], None)
           
           #the input is the content of train files instead of the list, SINCE I cache them into memory to save time!
           train_lie, train_truth, total_vol, words_lie, words_truth  =  trainNaiveBayes(lie_docs, iter_truth_docs)
           predict, current_cond_lie, current_cond_truth =  testNaiveBayes(input_dir[i],doc_folder, train_lie, train_truth,
                          num_lie, num_truth - 1,
                          total_vol,words_lie, words_truth)  
           
           #for calculating the condional prob
           if (len(average_cond_truth) == 0):
               average_cond_truth = current_cond_truth
           else:
               for key in current_cond_truth:
                   if key in average_cond_truth:
                       average_cond_truth[key] = average_cond_truth[key] + (1/i)*(current_cond_truth[key] - average_cond_truth[key])
                   else:
                       average_cond_truth[key] = current_cond_truth[key] 
                       
           if (predict == 1):
               correct = correct + 1
               print(input_dir[i],'truth')
           else:
               print(input_dir[i],'lie')
               
    print ("accuracy:", correct/(len(input_dir) - 1))          
    sorted_lie = sorted(average_cond_lie.items(), key=operator.itemgetter(1))
    sorted_truth = sorted(average_cond_truth.items(), key=operator.itemgetter(1))
    sorted_lie.reverse()
    sorted_truth.reverse()
    cond_output = open('cond_output', 'w')
    
    for i in range(10):
        cond_output.write(str(sorted_truth[i]))
        cond_output.write('\n')

           
           
        
if __name__ == '__main__':

    doc_folder = sys.argv[1]


    #main('bestfriend.deception' )
    #main("test")
    main(doc_folder)