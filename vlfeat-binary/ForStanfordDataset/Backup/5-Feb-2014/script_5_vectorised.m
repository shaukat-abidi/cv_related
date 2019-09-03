%%Script 5 will load BoWs with Labels and generate 
%%% probabilities for L-SVM input files

%clc

close all
clear all

save_path = '/home/ssabidi/Desktop/vlfeat-binary/toolbox/staticAR/ForStanfordDataset/load_files/';
load_filename = strcat(save_path,'for_training.mat');

load(load_filename,'-mat');
load('learned_weights.mat','-mat');



%%%Training file's data for L-SVM%%%
%data file format: action_label hidden_var_init probability_vals
probability_vals = [];
hidden_var_init = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%For debugging%%%%%%%%%%%%%%%%%%%%%
%%%To check logistic regression classfier
% scores = sigmoid(weights,example)
% scores: 23-valued vector (probability that example belong to 23 classes)
% weights: learned weights by logistic regression classifier
% example: bag-of-words for superpixel with 400 dimensions

%%for debugging logistic-regression classifier
weights = []; % weights for 23 classifiers(400x23)
offsets = []; % offsets for 23 classifiers (1 x 23)

for iter_classifier=1:23
    weights = [weights, w.data{iter_classifier}.data.rot];
    offsets = [offsets;w.data{iter_classifier}.data.offset];
end

% logistic classifier for speed
match_probs = [];%calculating probs of each example
                 %manually using logistic function
match_latentlabels = [];%labels for superpixel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



tic    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Classifying Superpixels%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter_examples = 1:size(examples,1)
    x=examples(iter_examples,:);
    scores=sigmoid(weights,offsets,x');
    [val,ind] = max(scores);
    match_probs = [match_probs;scores'];
    match_latentlabels = [match_latentlabels;ind];
end
classifying_all = toc

save_filename_1 = strcat(save_path,'for_writingToFile.mat');
save_filename_2 = strcat(save_path,'latent_variable_init.mat');


save(save_filename_1,'match_probs','labels');
save(save_filename_2,'match_latentlabels');
