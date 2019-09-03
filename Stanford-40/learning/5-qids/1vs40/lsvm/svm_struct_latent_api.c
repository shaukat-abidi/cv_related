/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_api.c                                            */
/*                                                                      */
/*   API function definitions for Latent SVM^struct, multiclass case    */
/*                                                                      */  
/*   Adaptation of orginal code from Yu and Joachims                    */
/*   Authors of adaptation: Massimo Piccardi, Ehsan Zare Borzeshi       */
/*   Date: July 2013                                                    */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "svm_struct_latent_api_types.h"
#include "./SFMT-src-1.3.3/SFMT.h"

#define MAX_INPUT_LINE_LENGTH 10000
#define MAX(x,y) ((x) < (y) ? (y) : (x))


double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm);
LATENT_VAR copy_latentvar(LATENT_VAR h,long length);
double impute_score(SVECTOR *fvec,STRUCTMODEL *sm);
SVECTOR *psi_superpixel(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_superpixel_comp(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_superpixel_emission(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_superpixel_trans_1(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_superpixel_trans_2(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_comp(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_emission(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
SVECTOR *psi_trans(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm);
void print_fvec(SVECTOR *fvec);
void print_fvec_wvec(SVECTOR *fvec,STRUCTMODEL *sm);
void print_wvec(STRUCTMODEL *sm);
int max_label(SVECTOR *fvec,int numf );
void optimized_inference(SVECTOR *f,PATTERN x, LATENT_VAR hbar,LABEL ybar,long iter_tokens);
long find_best_class(LABEL *y,long length);
void free_label(LABEL y);

/*********************************************************************************************/
double loss(LABEL y, LABEL ybar, LATENT_VAR hbar, STRUCT_LEARN_PARM *sparm) {
/*
  Computes the loss of prediction (ybar,hbar) against the
  correct label y. 
*/ 
  if(y == ybar)
	{return(0.0);}
  else
	{return(1.0);}
	
}
/*********************************************************************************************/

SVECTOR *psi_comp(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
*/
  SVECTOR *fvec=NULL; 
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
  long cfeat_dim = length + 1;
  WORD cfeat[cfeat_dim];//compatibility feature
  long cfeat_ind = 0;
  	
  for(i=0;i<length;i++) //for each token(sequence)
  {
	
  //////////////////////////////////////////////////////////////////////
	  /* 3rd Factor: Generate Compatibility Features */
  //////////////////////////////////////////////////////////////////////

  //long cfeat_dim = length + 1;
  //WORD cfeat[cfeat_dim];//compatibility feature
  //long cfeat_ind = 0;

  cfeat[cfeat_dim-1].wnum = 0; //end of compatibility feature
      cfeat_ind = ( (y-1) * (fnum) ) + h.states[i];
	  cfeat[i].wnum= compbaseindex + cfeat_ind;
	  cfeat[i].weight= 1;
  
  
  //////////////////////////////////////////////////////////////////////
  
 }
  
  ////Comp Vec////////
  fshift=create_svector(cfeat,"",1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec);
  fvec = fshift;
  ////////////////////
  return(fvec);
}

/************************************************************************/
SVECTOR *psi_emission(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
*/
  SVECTOR *fvec=NULL; 
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
   //////////////////////////////////////////////////////////////////////
				/* 1st Factor: Generate emission features: */
  //////////////////////////////////////////////////////////////////////
  long iter_trans_words;
  long tot_trans_words = (length  * length) + 1 ;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  WORD tfeat[tot_trans_words];
  long cfeat_dim = length + 1;
  WORD cfeat[cfeat_dim];//compatibility feature
  long cfeat_ind = 0;
  	
  for(i=0;i<length;i++) //for each token(sequence)
  {
		long shift_index = 0;
		shift_index = emitbase + ( (y-1) * (states*states) )  + ( fnum * (h.states[i] - 1) );
		
		//printf("shift_index : %ld\n",shift_index); 

		fshift=shift_s(x.tokens[i]->fvec, shift_index);
		/* the above function creates an SVECTOR and copies the tokens in it.
		   Such tokens are one observation, "o_t", dimension by dimension.
		   However, it also shifts the index of each dimension (originally,
		   from 1 to fnum) to make it unique for a particular state and
		   also unique in the parameter vector */
		
		append_svector_list(fshift,fvec);
		fvec=fshift;
  
  //////////////////////////////////////////////////////////////////////
  
 }
  
  return(fvec);
}
/************************************************************************/
SVECTOR *psi_trans(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
*/
  SVECTOR *fvec=NULL; 
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
   //////////////////////////////////////////////////////////////////////
				/* 1st Factor: Generate emission features: */
  //////////////////////////////////////////////////////////////////////
  long iter_trans_words;
  long tot_trans_words = (length  * length) + 1 ;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  WORD tfeat[tot_trans_words];
  long cfeat_dim = length + 1;
  WORD cfeat[cfeat_dim];//compatibility feature
  long cfeat_ind = 0;
  	
  for(i=0;i<length;i++) //for each token(sequence)
  {
		
  
  //////////////////////////////////////////////////////////////////////
    
  //////////////////////////////////////////////////////////////////////
		/* 2nd Factor: Generate Transition Features */
  //////////////////////////////////////////////////////////////////////
  //long tot_trans_words = (length  * length) + 1 ;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  //WORD tfeat[tot_trans_words];
  
  if(i==0)
	{iter_trans_words = 0;}
  
  tfeat[tot_trans_words-1].wnum=0;   /* ! this element of tfeat is created only as terminator of the list of words */
  
  //printf("tot_trans_words: %ld\n",tot_trans_words); 


	  for(j=0;j<length;j++)
		{
				hlabel_i = h.states[i]; 
				hlabel_j = h.states[j];
				desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				tfeat[iter_trans_words].wnum = transbase + desired_index;
				
				tfeat[iter_trans_words].weight = 1.0;
				if(i==j){tfeat[iter_trans_words].weight = 0.0;}
				/*printf("i: %ld j:%ld h_i:%ld h_j:%ld wnum:%d weight:%f \n",i,j,
				hlabel_i,hlabel_j,tfeat[iter_trans_words].wnum 
				,tfeat[iter_trans_words].weight);*/
				iter_trans_words++;				
		}     
   
 //////////////////////////////////////////////////////////////////////  
    
  
  

  
  
 }
  
  /////Trans vec////
  fshift=create_svector(tfeat,"",1.0); 
  //fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec); //void append_svector_list(SVECTOR *a, SVECTOR *b) 
									/* appends SVECTOR b to the end of SVECTOR a. */
  fvec=fshift;
  /////////////////
  
  
  
  return(fvec);
}
/************************************************************************/
/************************************************************************/

/************************************************************************/
long get_wnum(LABEL y,long token_i,long token_j,LATENT_VAR h,STRUCT_LEARN_PARM *sparm)
{
	long first_ind;
	long second_ind;
	long hlabel_i;
	long hlabel_j;
	long desired_index;
	long desired_wnum;
 
    long classes = sparm->num_classes;
    long states = sparm->num_states;
    long fnum=sparm->num_features;
    long emitbase;
    long transbase;
  
    emitbase = 0;
    transbase = emitbase + (classes * states * fnum);
	
		  
	first_ind = token_i;
	second_ind = token_j;
	
	hlabel_i = h.states[first_ind]; 
	hlabel_j = h.states[second_ind];
	desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
	desired_wnum = transbase + desired_index;
	
	return(desired_wnum);
}
/***********************************************************************/

SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples=NULL;
  long     n;       /* number of examples */
  DOC      **examples_flat=NULL;
  double   *labels_flat=NULL;
  long     totwords=0,tottokens=0,i,queryid,maxlabel=0;

  /* Read as one set of examples using the read_documents function
     from SVM-light */
     /* totwords : no. of features 
      * tottokens: no. of superpixels
      * */
  read_documents(file,&examples_flat,&labels_flat,&totwords,&tottokens);
  printf(" totwords=%ld tottokens:%ld .. \n",totwords,tottokens);

  /* Create examples by splitting the input into sequences of tokens. */
  queryid=-1;
  n=0;
  for(i=0;i<tottokens;i++) {
    if(queryid < (long)examples_flat[i]->queryid) {
      queryid=(long)examples_flat[i]->queryid;
      n++;
      examples=(EXAMPLE *)realloc(examples,sizeof(EXAMPLE)*(n));
      examples[n-1].x.length=1;
      //examples[n-1].y.length=1; /*commented for Static Action Recognition experiment*/
      examples[n-1].x.tokens=(DOC **)my_malloc(sizeof(DOC **));
      //examples[n-1].y.labels=(long *)my_malloc(sizeof(long *)); /*commented for Static Action Recognition experiment (SAR)*/
      examples[n-1].y=(long)labels_flat[i]; /*added for SAR*/
    }
    else if(queryid > (long)examples_flat[i]->queryid) {
      printf("ERROR (Line %ld): qid example ID's have to be in increasing order.\n",i+1);
      exit(1);
    }
    else {
      examples[n-1].x.length++;
      //examples[n-1].y.length++;/*commented for Static Action Recognition experiment*/
      examples[n-1].x.tokens=(DOC **)realloc(examples[n-1].x.tokens,
				       sizeof(DOC **)*examples[n-1].x.length);
      //examples[n-1].y.labels=(long *)realloc(examples[n-1].y.labels,
				      //sizeof(long *)*examples[n-1].y.length);/*commented for Static Action Recognition experiment*/
    }
    examples[n-1].x.tokens[examples[n-1].x.length-1]=examples_flat[i];
    //examples[n-1].y.labels[examples[n-1].y.length-1]=(long)labels_flat[i];/*commented for Static Action Recognition experiment*/
	//printf(" i=%ld labels:%ld ..\n ",i,(long)labels_flat[i]);
	//printf(" %ld\n",examples[n-1].y.labels[examples[n-1].y.length-1]);
	//printf(" example# %ld\n",n);fflush(stdout);

    if(labels_flat[i] < 0) {
      printf("ERROR (Line %ld): Token label IDs cannot be negative.\n",i+1);
      exit(1);
    }
    maxlabel=MAX(maxlabel,(long)labels_flat[i]);
  }

  if(1)//if(struct_verbosity>=1)
    {
		printf(" %ld examples, %ld tokens, %ld features, %ld classes... ",n,tottokens,totwords,maxlabel);
	}

  free(examples_flat); 
  free(labels_flat);

  sample.n=n;
  sample.examples=examples;
  return(sample);
}

/************************************************************************/
/*read_struct_examples function removed: See in removed_org_functions.txt*/

/************************************************************************/

void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) {

  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */
  
  EXAMPLE  *examples=sample.examples;
  PATTERN  *x;
  LABEL    *y;
  long     maxlabel=0,maxfeat=0,i,j;
  WORD     *ai;
  double   sizePsi;
 
  long totwords=0;
  WORD *w;

  /* find number of classes and number of features in training set */
  for(i=0;i<sample.n;i++) {
    x=&examples[i].x;
    //y=&examples[i].y;/*commented for Static Action Recognition*/
    for(j=0;j<x->length;j++) /*changed for Static Action Recogntion: Previously, it was for(j=0;j<y->length;j++) */ 
    {
      maxlabel=MAX(maxlabel,examples[i].y); /*changed for Static Action Recogntion: Previously, it was: maxlabel=MAX(maxlabel,y->labels[j]) */ 
      if(examples[i].y<1) /*changed for Static Action Recogntion: Before it was: y->labels[j]<1 */
      { 
	printf("ERROR: Found token label ID '%ld'. Token label IDs must be greater or equal to 1!\n",examples[i].y);
	/*changed for Static Action Recogntion: Before it was: y->labels[j] */
	exit(1);
      }
      for(ai=x->tokens[j]->fvec->words;ai->wnum;ai++) {
	maxfeat=MAX(maxfeat,ai->wnum);
      }
    }
  }
  sparm->num_classes=1; //pre-initialisation
  sparm->num_classes=maxlabel;//actual number of classes
  sparm->num_features=maxfeat;// no. of features
  
  sm->svm_model = (MODEL *) my_malloc(sizeof(MODEL));
  sm->svm_model->kernel_parm.kernel_type = 0;
  
  sparm->num_states = 23; /* we define this arbitrarily as it cannot be read from the input file*/
	
  int feat_size = 1  
                   + (sparm->num_classes * sparm->num_features * sparm->num_states)
                   + (sparm->num_classes * sparm->num_states * sparm->num_states)
                   + (sparm->num_classes * sparm->num_states); 
  
  sm->sizePsi = feat_size;
  
  sm->svm_model->lin_weights = (double *) my_malloc(sizeof(double)*sm->sizePsi);

}

/************************************************************************/

void init_latent_variables(SAMPLE *sample, LEARN_PARM *lparm, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm,
                            long *sp_id, long tot_sp ) {
/*
  Initialize latent variables in the first iteration of training.
  Latent variables are stored at sample.examples[i].h, for 1<=i<=sample.n.
*/
   /********/
  /*sp_id: Superpixel IDs*/
  /*tot_sp: total no. of superpixels*/
  int i; 
  int j;
  int iter_sp;//iterate over superpixels pointed by sp_id
  iter_sp=0;
  
  for (i=0;i<sample->n;i++) {
	
	  sample->examples[i].h.states = (long *) my_malloc( sizeof(long) * sample->examples[i].x.length );	
      sample->examples[i].x.initial_states = (long *) my_malloc( sizeof(long) * sample->examples[i].x.length );
	
	for (j=0;j<sample->examples[i].x.length;j++) {
		  sample->examples[i].h.states[j] = sp_id[iter_sp];
		  sample->examples[i].x.initial_states[j] = sp_id[iter_sp];		  
		  iter_sp++;
	  }
  
  }
}

/************************************************************************/
/************************************************************************/

SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
*/
  SVECTOR *fvec=NULL; 
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
   //////////////////////////////////////////////////////////////////////
				/* 1st Factor: Generate emission features: */
  //////////////////////////////////////////////////////////////////////
  long iter_trans_words;
  long tot_trans_words = (length  * length) + 1 ;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  WORD tfeat[tot_trans_words];
  long cfeat_dim = length + 1;
  WORD cfeat[cfeat_dim];//compatibility feature
  long cfeat_ind = 0;
  	
  for(i=0;i<length;i++) //for each token(sequence)
  {
		long shift_index = 0;
		shift_index = emitbase + ( (y-1) * (states*states) )  + ( fnum * (h.states[i] - 1) );
		
		//printf("shift_index : %ld\n",shift_index); 

		fshift=shift_s(x.tokens[i]->fvec, shift_index);
		/* the above function creates an SVECTOR and copies the tokens in it.
		   Such tokens are one observation, "o_t", dimension by dimension.
		   However, it also shifts the index of each dimension (originally,
		   from 1 to fnum) to make it unique for a particular state and
		   also unique in the parameter vector */
		
		append_svector_list(fshift,fvec);
		fvec=fshift;
  
  //////////////////////////////////////////////////////////////////////
    
  //////////////////////////////////////////////////////////////////////
		/* 2nd Factor: Generate Transition Features */
  //////////////////////////////////////////////////////////////////////
  //long tot_trans_words = (length  * length) + 1 ;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  //WORD tfeat[tot_trans_words];
  
  if(i==0)
	{iter_trans_words = 0;}
  
  tfeat[tot_trans_words-1].wnum=0;   /* ! this element of tfeat is created only as terminator of the list of words */
  
  //printf("tot_trans_words: %ld\n",tot_trans_words); 


	  for(j=0;j<length;j++)
		{
				hlabel_i = h.states[i]; 
				hlabel_j = h.states[j];
				desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				tfeat[iter_trans_words].wnum = transbase + desired_index;
				
				tfeat[iter_trans_words].weight = 1.0;
				if(i==j){tfeat[iter_trans_words].weight = 0.0;}
				/*printf("i: %ld j:%ld h_i:%ld h_j:%ld wnum:%d weight:%f \n",i,j,
				hlabel_i,hlabel_j,tfeat[iter_trans_words].wnum 
				,tfeat[iter_trans_words].weight);*/
				iter_trans_words++;				
		}     
   
 //////////////////////////////////////////////////////////////////////  
    
  
  

  //////////////////////////////////////////////////////////////////////
	  /* 3rd Factor: Generate Compatibility Features */
  //////////////////////////////////////////////////////////////////////

  //long cfeat_dim = length + 1;
  //WORD cfeat[cfeat_dim];//compatibility feature
  //long cfeat_ind = 0;

  cfeat[cfeat_dim-1].wnum = 0; //end of compatibility feature
      cfeat_ind = ( (y-1) * (fnum) ) + h.states[i];
	  cfeat[i].wnum= compbaseindex + cfeat_ind;
	  cfeat[i].weight= 1.0;
  
  
  //////////////////////////////////////////////////////////////////////
  
 }
  
  /////Trans vec////
  fshift=create_svector(tfeat,"",1.0); 
  //fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec); //void append_svector_list(SVECTOR *a, SVECTOR *b) 
									/* appends SVECTOR b to the end of SVECTOR a. */
  fvec=fshift;
  /////////////////
  
  
  ////Comp Vec////////
  fshift=create_svector(cfeat,"",1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec);
  fvec = fshift;
  ////////////////////
  return(fvec);
}
/************************************************************************/
/************************************************************************/

void classify_struct_example(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
//PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  long length;
  long state;
  long class;
  long bestclass;
  long beststate;
  long total_classes;
  long iter_classes;
  long iter_tokens;
  long iter_tokens_for_CA;
  long repeat_iteration;
  SVECTOR *f = NULL;
  SVECTOR *old = NULL;
  SVECTOR *f_base = NULL;
  SVECTOR *f_storage = NULL;
  LATENT_VAR best_hbar_c[2];
  long check_iterations;
  
  int first;
  double score;
  double bestscore;
  double best_class_scores[sparm->num_classes];

  length = x.length;
  bestclass = -1;
  beststate = -1;
  total_classes = sparm->num_classes;
  first = 1; /* toggle for the first iteration */
  score=0;
  bestscore=0;
  iter_classes=0;
  iter_tokens=0;//iterate tokens
  iter_tokens_for_CA=0;//iterate tokens for Class Assignment
  repeat_iteration = 0;
  check_iterations=0;	

  //empty ybar,hbar are passed to this function
  h->states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[0].states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[1].states = (long *)my_malloc(sizeof(long) * length);
  

  for(iter_classes=1;iter_classes<=total_classes;iter_classes++)
  {

	  /*Changing class assignments for x to find the best class*/
	  /* best_class_scores[0]   = best score for class-1 */
	  /* best_class_scores[1]   = best score for class-2 */
	  /* best_class_scores[2]   = best score for class-3 */
	  /* best_class_scores[n-1] = best score for class-n */
	      best_class_scores[iter_classes - 1] = 0; //Zero based index
	      *y = iter_classes;
	 	  
	 	  
	 	  //double t1 = get_runtime();
	 	  //Resetting latent_vars for the evaluation of next class
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
		 {
		      h->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		 }	
		//printf("Debugging line # %d and file name %s \n",__LINE__,__FILE__); fflush(stdout);	  	  
		for (repeat_iteration=0;repeat_iteration<4;repeat_iteration++) 
			{	
				  /*Start here of repeat iteration*/
				  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				  {
					first = 1; //when calculating score for new superpixel
				  
					for(state=1;state<=sparm->num_states;state++)
					  {
						  
						  h->states[iter_tokens] = state;
						  //printf(" class:%ld,iteration:%ld,token:%ld,state:%ld\n",iter_classes,repeat_iteration,iter_tokens,state);	
						  //SVECTOR* psi_superpixel(PATTERN x, long token_id, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
						  f = psi_superpixel(x,iter_tokens,*y,*h,sm,sparm);
						  
						  score = 0;
						  
  						  while(f != NULL){
							    score = score + sprod_ns(sm->w,f);
 	  							old = f;
								f=f->next;
								free_svector_block(old);
							}
							  
						  if((bestscore < score)  || (first)) {
								bestscore = score;
								beststate = state;
								first = 0;
							}
  							
  							//printf(" class:%ld,iteration:%ld,token:%ld,state:%ld, score:%f \n",iter_classes,repeat_iteration,iter_tokens,state,score);	
					  }
					  
					  
					  (*h).states[iter_tokens] = beststate;
					  
				 }
	  			/*End of repeat iteration */
	  			//double t2 = get_runtime();
	  			//printf("time taken for each-class iteration = %.5f secs\n",(t2-t1)/100);
			}
			
			    /*Calculate the feature vector with the best asssigned h's*/
				f = psi(x,*y,*h,sm,sparm);
				score=0;
				best_class_scores[iter_classes - 1] = 0;
				/* Calculate score for this class, after finding best h's 
				 * with repeated iterations over tokens */		  
				while(f != NULL)
				{
						score = score + sprod_ns(sm->w,f);
						old = f;
						f=f->next;
						free_svector_block(old);
				}
				best_class_scores[iter_classes - 1] = score;
				//Copy best states for this class inside best_hbar_c, so that they 
				//can be recovered afterwards
				for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				{
					best_hbar_c[iter_classes - 1].states[iter_tokens] = (*h).states[iter_tokens];
				}  
			    	  
	  }
	  
	  
		  if(best_class_scores[0] > best_class_scores[1])
		  {*y = 1;}
		  else
		  {*y = 2;} 
		  //printf("final bestclass = %ld \n",*ybar);
		 
		/* Restoring best set for hidden state */
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
			{
				 h->states[iter_tokens] = best_hbar_c[*y - 1].states[iter_tokens];  
				 //printf("Initial h[%ld] = %ld , Assigned hbar[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]);
			}
  	  
  	  free(best_hbar_c[0].states);
  	  free(best_hbar_c[1].states);

}
/***********************************************************************************************************************************************/
/***********************************************************************/

SVECTOR *psi_superpixel(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
  
  New Input:
  SVECTOR *fvec: feature vector of superpixel with id "long token id"
  long token_id: ID of current superpixel [range: 0 to (total superpixels in image)-1]
*/
  SVECTOR *fvec=NULL;
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
   //////////////////////////////////////////////////////////////////////
				/* 1st Factor: Generate emission features: */
  //////////////////////////////////////////////////////////////////////
	
	long shift_index = 0;
    shift_index = emitbase + ( (y-1) * (states*states) )  + ( fnum * (h.states[iter_tokens] - 1) );
    //printf("shift_index : %ld\n",shift_index); 

	fshift=shift_s(x.tokens[iter_tokens]->fvec, shift_index);
	/* the above function creates an SVECTOR and copies the tokens in it.
	   Such tokens are one observation, "o_t", dimension by dimension.
	   However, it also shifts the index of each dimension (originally,
	   from 1 to fnum) to make it unique for a particular state and
	   also unique in the parameter vector */
	
	append_svector_list(fshift,fvec);
	fvec=fshift;
  
  //////////////////////////////////////////////////////////////////////
    
    
    
  //////////////////////////////////////////////////////////////////////
		/* 2nd Factor: Generate Transition Features */
  //////////////////////////////////////////////////////////////////////
  long tot_trans_words = length + 1;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  WORD tfeat[tot_trans_words];
  long first_ind;
  long second_ind;
  long iter_trans_words = 0;

  tfeat[tot_trans_words-1].wnum=0;   /* ! this element of tfeat is created only as terminator of the list of words */
  
  //printf("tot_trans_words: %ld\n",tot_trans_words); 

	for(i=0;i<length;i++){
		   first_ind = iter_tokens;
		   second_ind = i;
  		   hlabel_i = h.states[first_ind]; 
		   hlabel_j = h.states[second_ind];
 		   desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;

		   tfeat[iter_trans_words].wnum = transbase + desired_index;
		   tfeat[iter_trans_words].weight = 1.0;
		   if(first_ind==second_ind){tfeat[iter_trans_words].weight = 0.0;}
		   iter_trans_words++;		   
	  }
	 
   fshift=create_svector(tfeat,"",1.0); 
  //fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec); //void append_svector_list(SVECTOR *a, SVECTOR *b) 
									/* appends SVECTOR b to the end of SVECTOR a. */
  fvec=fshift;
  
  ////////////////////////////////////////
  //Now generating transition features_2//
  ////////////////////////////////////////
  tot_trans_words = length  + 1;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  WORD tfeat_2[tot_trans_words];
  iter_trans_words = 0;

  tfeat_2[tot_trans_words-1].wnum=0;   /* ! this element of tfeat is created only as terminator of the list of words */
  
  //printf("tot_trans_words: %ld\n",tot_trans_words); 

	for(i=0;i<length;i++){
		   first_ind = i;
		   second_ind = iter_tokens;
		   	//(i,j) has been modified. Now modify (j,i)
			hlabel_i = h.states[first_ind]; 
			hlabel_j = h.states[second_ind];
			desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
            tfeat_2[iter_trans_words].wnum = transbase + desired_index;
			tfeat_2[iter_trans_words].weight = 1.0;
  		    if(first_ind==second_ind){tfeat_2[iter_trans_words].weight = 0.0;}
			
			iter_trans_words++;
		   
	  }
	 
   fshift=create_svector(tfeat_2,"",1.0); 
  //fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec); //void append_svector_list(SVECTOR *a, SVECTOR *b) 
									/* appends SVECTOR b to the end of SVECTOR a. */
  fvec=fshift;
 //////////////////////////////////////////////////////////////////////
  

  //////////////////////////////////////////////////////////////////////
	  /* 3rd Factor: Generate Compatibility Features */
  //////////////////////////////////////////////////////////////////////

  long cfeat_dim = 2;
  WORD cfeat[cfeat_dim];//compatibility feature
  long cfeat_ind = 0;

  cfeat[cfeat_dim-1].wnum = 0; //end of compatibility feature
  
  cfeat_ind = ( (y-1) * (fnum) ) + h.states[iter_tokens];

  cfeat[0].wnum= compbaseindex + cfeat_ind;
  cfeat[0].weight= 1.0;
  
  fshift=create_svector(cfeat,"",1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec);
  fvec = fshift;
  //////////////////////////////////////////////////////////////////////
  
  
  return(fvec);
}
/***********************************************************************/
SVECTOR *psi_superpixel_emission(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
  
  New Input:
  SVECTOR *fvec: feature vector of superpixel with id "long token id"
  long token_id: ID of current superpixel [range: 0 to (total superpixels in image)-1]
*/
  SVECTOR *fvec=NULL;
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
   //////////////////////////////////////////////////////////////////////
				/* 1st Factor: Generate emission features: */
  //////////////////////////////////////////////////////////////////////
	
	long shift_index = 0;
    shift_index = emitbase + ( (y-1) * (states*states) )  + ( fnum * (h.states[iter_tokens] - 1) );
	//printf("shift_index : %ld\n",shift_index); 

	fshift=shift_s(x.tokens[iter_tokens]->fvec, shift_index);
	/* the above function creates an SVECTOR and copies the tokens in it.
	   Such tokens are one observation, "o_t", dimension by dimension.
	   However, it also shifts the index of each dimension (originally,
	   from 1 to fnum) to make it unique for a particular state and
	   also unique in the parameter vector */
	
	append_svector_list(fshift,fvec);
	fvec=fshift;
  
  //////////////////////////////////////////////////////////////////////
    
  
  return(fvec);
}
/***********************************************************************/
SVECTOR *psi_superpixel_comp(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
  
  New Input:
  SVECTOR *fvec: feature vector of superpixel with id "long token id"
  long token_id: ID of current superpixel [range: 0 to (total superpixels in image)-1]
*/
  SVECTOR *fvec=NULL;
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
  //////////////////////////////////////////////////////////////////////
	  /* 3rd Factor: Generate Compatibility Features */
  //////////////////////////////////////////////////////////////////////

  long cfeat_dim = 2;
  WORD cfeat[cfeat_dim];//compatibility feature
  long cfeat_ind = 0;

  
  cfeat_ind = ( (y-1) * (fnum) ) + h.states[iter_tokens];

  cfeat[0].wnum= compbaseindex + cfeat_ind;
  cfeat[0].weight= 1.0;
  
  cfeat[1].wnum = 0; //end of compatibility feature
  cfeat[1].weight = 0.0; //end of compatibility feature

  fshift=create_svector(cfeat,"",1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec);
  fvec = fshift;
  //////////////////////////////////////////////////////////////////////
  
  
  return(fvec);
}
/***********************************************************************/
SVECTOR *psi_superpixel_trans_1(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
  
  New Input:
  SVECTOR *fvec: feature vector of superpixel with id "long token id"
  long token_id: ID of current superpixel [range: 0 to (total superpixels in image)-1]
*/
  SVECTOR *fvec=NULL;
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */

  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
    
  //////////////////////////////////////////////////////////////////////
		/* 2nd Factor: Generate Transition Features_1 */
  //////////////////////////////////////////////////////////////////////
  long tot_trans_words = length + 1;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  WORD tfeat[tot_trans_words];
  long first_ind;
  long second_ind;
  long iter_trans_words = 0;

  tfeat[tot_trans_words-1].wnum=0;   /* ! this element of tfeat is created only as terminator of the list of words */
  tfeat[tot_trans_words-1].weight=0.0;   /* ! this element of tfeat is created only as terminator of the list of words */
  
  //printf("tot_trans_words: %ld\n",tot_trans_words); 
  for(i=0;i<length;i++){
		   first_ind = iter_tokens;
		   second_ind = i;
  		   hlabel_i = h.states[first_ind]; 
		   hlabel_j = h.states[second_ind];
 		   desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;

		   tfeat[iter_trans_words].wnum = transbase + desired_index;
		   tfeat[iter_trans_words].weight = 1.0;
		   if(first_ind==second_ind){tfeat[iter_trans_words].weight = 0.0;}
		   iter_trans_words++;		   
   }
	      
   fshift=create_svector(tfeat,"",1.0); 
  //fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec); //void append_svector_list(SVECTOR *a, SVECTOR *b) 
									/* appends SVECTOR b to the end of SVECTOR a. */
  fvec=fshift;
  
  return(fvec);
}
/***********************************************************************/
SVECTOR *psi_superpixel_trans_2(PATTERN x, long iter_tokens, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Creates the feature vector \Psi(x,y,h) and returns a pointer to 
  sparse vector SVECTOR in SVM^light format. The dimension of the 
  feature vector returned has to agree with the dimension in sm->sizePsi. 
  
  Each SVECTOR is in a sparse representation of pairs
  <featurenumber:featurevalue>, where the last pair has
  featurenumber 0 as a terminator. Featurenumbers start with 1 and
  end with sizePsi + 1. Featuresnumbers that are not specified default
  to value 0. As mentioned before, psi() actually returns a list of
  SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
  specifies the next element in the list, terminated by a NULL
  pointer. The list can be thought of as a linear combination of
  vectors, where each vector is weighted by its 'factor' (always set to 1.0 here).
  This feature vector is multiplied with the learned (kernelized) weight vector
  to score label y for pattern x. Without kernels, there will be one weight
  in sm.w for each feature. Note that psi has to match
  find_most_violated_constraint_???(x, y, sm) and vice versa. 
  
  New Input:
  SVECTOR *fvec: feature vector of superpixel with id "long token id"
  long token_id: ID of current superpixel [range: 0 to (total superpixels in image)-1]
*/
  SVECTOR *fvec=NULL;
  SVECTOR *fshift=NULL;
  long length=x.length;
  long classes = sparm->num_classes;
  long states = sparm->num_states;
  long fnum=sparm->num_features;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
  
  /*
   printf("fnum: %ld\n",fnum); 
   printf("length: %ld\n",length); 
   printf("classes: %ld\n",classes); 
   printf("states: %ld\n",states); 
   */
   
  /* Initialise base addresses for emission,transition and 
   * compatibility portions of weight vector
   */
  emitbase = 0;
  transbase = emitbase + (classes * states * fnum);
  compbaseindex = transbase + (classes * states * states);
  
  /* 
  printf("emitbase: %ld\n",emitbase); 
  printf("transbase: %ld\n",transbase); 
  printf("compbaseindex: %ld\n",compbaseindex); 
  */
  
  /* shifts the feature numbers by shift positions */
  /* Shaukat: For instance, if a->words->wnum = [0,1,2] and and shift = 3 
   * then a->words->wnum = [3,4,5]  
   * */

  
   /* There are three factors in scoring function */
   
   
  //////////////////////////////////////////////////////////////////////
		/* 2nd Factor: Generate Transition Features */
  //////////////////////////////////////////////////////////////////////
  long tot_trans_words = length + 1;  
  //NB: What if we have 2 superpixels,although its impossible? Remove this Bug
  long first_ind;
  long second_ind;
  long iter_trans_words = 0;

  ////////////////////////////////////////
  //Now generating transition features_2//
  ////////////////////////////////////////
  WORD tfeat_2[tot_trans_words];
  
  tfeat_2[tot_trans_words-1].wnum=0;   /* ! this element of tfeat is created only as terminator of the list of words */
  tfeat_2[tot_trans_words-1].weight=0.0;   /* ! this element of tfeat is created only as terminator of the list of words */
  
  //printf("tot_trans_words: %ld\n",tot_trans_words); 

	for(i=0;i<length;i++){
		   first_ind = i;
		   second_ind = iter_tokens;
		   	//(i,j) has been modified. Now modify (j,i)
			hlabel_i = h.states[first_ind]; 
			hlabel_j = h.states[second_ind];
			desired_index = ( (y-1) * (states*states) ) + ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
            tfeat_2[iter_trans_words].wnum = transbase + desired_index;
			tfeat_2[iter_trans_words].weight = 1.0;
  		    if(first_ind==second_ind){tfeat_2[iter_trans_words].weight = 0.0;}
			
			iter_trans_words++;
		   
	  }
	 
   fshift=create_svector(tfeat_2,"",1.0); 
  //fshift=create_svector(tfeat,NULL,1.0); /* prototype is: SVECTOR *create_svector(WORD *words,char *userdefined,double factor) */
  append_svector_list(fshift,fvec); //void append_svector_list(SVECTOR *a, SVECTOR *b) 
									/* appends SVECTOR b to the end of SVECTOR a. */
  fvec=fshift;
 //////////////////////////////////////////////////////////////////////
  
  return(fvec);
}
/***********************************************************************/


/**************************************************************************************************************************************************/
void find_most_violated_constraint_marginrescaling_optimized_psi_broken_emission_temp_comp(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  /*This is the version which has reduced computation time*/
  long length;
  long state;
  long class;
  long bestclass;
  long beststate;
  long total_classes;
  long iter_classes;
  long iter_tokens;
  long iter_tokens_for_CA;
  long repeat_iteration;
  
  SVECTOR *f = NULL;
  SVECTOR *f_emission = NULL;
  SVECTOR *f_trans_1 = NULL;
  SVECTOR *f_trans_2 = NULL;
  SVECTOR *f_comp = NULL;
  SVECTOR *old = NULL;
  SVECTOR *f_base = NULL;
  SVECTOR *f_storage = NULL;
  LATENT_VAR best_hbar_c[2];
  long check_iterations;
  
  int first;
  double score;
  double bestscore;
  double best_class_scores[2];
  
  double trans_score_p1[24];
  double trans_score_p2[24];
  double *weight_vector;
  
  /*Optimization#2*/
  double emission_score;
  double trans_score_1;
  double trans_score_2;
  double comp_score;
  long wnum_id;
  	
  length = x.length;
  bestclass = -1;
  beststate = -1;
  total_classes = sparm->num_classes;
  first = 1; /* toggle for the first iteration */
  score=0.0;
  bestscore=0.0;
  iter_classes=0;
  iter_tokens=0;//iterate tokens
  iter_tokens_for_CA=0;//iterate tokens for Class Assignment
  repeat_iteration = 0;
  check_iterations=0;	
  weight_vector = sm->w;
  best_class_scores[0] = 0.0;
  best_class_scores[1] = 0.0;
  
  /*Optimization#2*/
  emission_score=0.0;
  trans_score_1=0.0;
  trans_score_2=0.0;
  comp_score=0.0;
  wnum_id = 0;
  
  //empty ybar,hbar are passed to this function
  hbar->states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[0].states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[1].states = (long *)my_malloc(sizeof(long) * length);
  
  
  for(iter_classes=1;iter_classes<=total_classes;iter_classes++)
  {

	  /*Changing class assignments for x to find the best class*/
	  /* best_class_scores[0]   = best score for class-1 */
	  /* best_class_scores[1]   = best score for class-2 */
	  /* best_class_scores[2]   = best score for class-3 */
	  /* best_class_scores[n-1] = best score for class-n */
	      best_class_scores[iter_classes - 1] = 0.0; //Zero based index
	      *ybar = iter_classes;
	 	  
  
	 	  //double t1 = get_runtime();
	 	  //Resetting latent_vars for the evaluation of next class
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
		 {
		      hbar->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		 }	
	
	  	  
		for (repeat_iteration=0;repeat_iteration<4;repeat_iteration++) 
			{	
				  /*Start here of repeat iteration*/
				  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				  {
					first = 1; //when calculating score for new superpixel
					for(state=1;state<=sparm->num_states;state++)
					  {						  
							  hbar->states[iter_tokens] = state;
							  //f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_emission = psi_emission(x,*ybar,*hbar,sm,sparm);
							  f_trans_1 = psi_trans(x,*ybar,*hbar,sm,sparm);
							  //f_trans_1 = psi_superpixel_trans_1(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  //f_trans_2 = psi_superpixel_trans_2(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  //f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_comp(x,*ybar,*hbar,sm,sparm);
							  
							  score = 0.0;
							  emission_score=0.0;
							  trans_score_1=0.0;
							  trans_score_2=0.0;
							  comp_score=0.0;
							  
							  while(f_emission != NULL){
								emission_score = emission_score + sprod_ns(sm->w,f_emission);
								old = f_emission;
								f_emission=f_emission->next;
								free_svector_block(old);
							  }
							  while(f_trans_1 != NULL){
								trans_score_1 = trans_score_1 + sprod_ns(sm->w,f_trans_1);
								old = f_trans_1;
								f_trans_1=f_trans_1->next;
								free_svector_block(old);
							  }
							  /*
							  while(f_trans_2 != NULL){
								trans_score_2 = trans_score_2 + sprod_ns(sm->w,f_trans_2);
								old = f_trans_2;
								f_trans_2=f_trans_2->next;
								free_svector_block(old);
							  }
							  */ 
							  while(f_comp != NULL){
								comp_score = comp_score + sprod_ns(sm->w,f_comp);
								old = f_comp;
								f_comp=f_comp->next;
								free_svector_block(old);
							  }
							  
							  //score = emission_score + 	trans_score_1 + trans_score_2 + comp_score; 
							  score = emission_score + 	trans_score_1 + comp_score; 
							  score = score + loss(y,*ybar,*hbar,sparm);
							  if( (bestscore < score)  || (first))
							  {
								  bestscore = score;
								  beststate = state;
								  first = 0;
							  }
							  //printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  //emission_score,trans_score_1,trans_score_2 , comp_score);							  
							  if(f_emission != NULL)
								free_svector(f_emission);
							  if(f_trans_1 != NULL)
								free_svector(f_trans_1);
							  if(f_trans_2 != NULL)
								free_svector(f_trans_2);
							  if(f_comp != NULL)
								free_svector(f_comp);
  							  /***************************************************************/
						  						  
					  }/*End of iteration over state*/
					  
				  	  /*Assigning the best state to this token*/
					  (*hbar).states[iter_tokens] = beststate;
					  
					  if(repeat_iteration == 3){
						  /*We need to store the scores and state of 3rd iteration only*/
						  /* Accumulate the scores of superpixels keeping class constant */
						  best_class_scores[iter_classes - 1] = best_class_scores[iter_classes - 1] + bestscore;
						  best_hbar_c[iter_classes - 1].states[iter_tokens] = beststate;			  
						}
					  
				 }/*End of Iter Tokens [1 L]*/
	  			
	  		}/*End of Repeat Iter [0 3]*/
	  		
			    /* *****************************************************  
			     * *****  Deactivate this area so that it is consistent
			     * *****   with margin_rescaling_fullPsi - Start   
			     * *****************************************************/
			    /*Calculate the feature vector with the best asssigned h's*/
				//f = psi(x,*ybar,*hbar,sm,sparm);
				//score=0.0;
				//best_class_scores[iter_classes - 1] = 0.0;
				/* Calculate score for this class, after finding best h's 
				 * with repeated iterations over tokens */		  
				//while(f != NULL)
				//{
						//score = score + sprod_ns(sm->w,f);
						//old = f;
						//f=f->next;
						//free_svector_block(old);
				//}
				//score = score + loss(y,*ybar,*hbar,sparm);
				//best_class_scores[iter_classes - 1] = score;
				/* *****************************************************  
			     * *****  Deactivate this area so that it is consistent
			     * *****   with margin_rescaling_fullPsi - Finished    
			     * *****************************************************/
			    
			    	  
	  }/*End of Iter Classes [1 2]*/
	  	  
	  	  
		  if(best_class_scores[0] > best_class_scores[1])
		  {*ybar = 1;}
		  else
		  {*ybar = 2;} 
		  //print_wvec(sm);
		  printf("bestclass[1] = %f, bestclass[2] = %f \n",best_class_scores[0] , best_class_scores[1]);
		  //printf("final bestclass = %ld \n",*ybar);
		 
		/* Restoring best set for hidden state */
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
			{
				 hbar->states[iter_tokens] = best_hbar_c[*ybar - 1].states[iter_tokens];  
				 //printf("Initial h[%ld] = %ld , Assigned hbar[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]);
 				 //printf("Initial h[%ld] = %ld , hbar_class2[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,best_hbar_c[1].states[iter_tokens]);
 				 printf("Init_h[%ld] = %ld , hbar[%ld] = %ld , h_c1[%ld]=%ld , h_c2[%ld]=%ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]
															 ,iter_tokens,best_hbar_c[0].states[iter_tokens],iter_tokens,best_hbar_c[1].states[iter_tokens]);

			}
  	  
  	  
  	  //printf("GT_y = %ld [%f], y_bar = %ld [%f] best_class_scores=[%f , %f]\n",y,best_class_scores[y-1],*ybar,best_class_scores[*ybar-1]
			 //,best_class_scores[0],best_class_scores[1]); 
			 
	
  	  
  	  /***************Checking Assignment - Start*******************************/
  	  best_class_scores[0] = 0.0;
  	  for(iter_tokens=0;iter_tokens<length;iter_tokens++){
		  
	  /* Check assignment scores */
	  f = psi(x,1,best_hbar_c[0],sm,sparm);
	  score=0.0;
	  /* Calculate score for this class, after finding best h's 
	     * with repeated iterations over tokens */		  
		 while(f != NULL)
		 {
				score = score + sprod_ns(sm->w,f);
				old = f;
				f=f->next;
				free_svector_block(old);
		}
		score = score + loss(y,1,best_hbar_c[0],sparm);
		best_class_scores[0] += score;
	  }
	  
	  
	  best_class_scores[1] = 0.0;
	  for(iter_tokens=0;iter_tokens<length;iter_tokens++){
	  
	  f = psi(x,2,best_hbar_c[1],sm,sparm);
	  score=0.0;
	  /* Calculate score for this class, after finding best h's 
	     * with repeated iterations over tokens */		  
		 while(f != NULL)
		 {
				score = score + sprod_ns(sm->w,f);
				old = f;
				f=f->next;
				free_svector_block(old);
		}
		score = score + loss(y,2,best_hbar_c[1],sparm);
		best_class_scores[1] += score;
		}
	  /***************Checking Assignment - End*******************************/
	  f = psi(x,2,best_hbar_c[1],sm,sparm);
  	  //printf("***printing fvec for new example with y=2,best_hbar_c[1]****\n");
	  //print_fvec(f);
	  old = NULL;
	  while(f != NULL){
			old = f;
			f=f->next;
			free_svector_block(old);
		}
	  
  	  printf("Evaluation of best_class_scores with full psi =[%f , %f]\n",best_class_scores[0],best_class_scores[1]); 
			 
  	  free(best_hbar_c[0].states);
  	  free(best_hbar_c[1].states);
	  
}
/**************************************************************************************************************************************************/
void find_most_violated_constraint_marginrescaling_beforeremovingIterTokensPart(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  /*This is the version which has reduced computation time*/
  long length;
  long state;
  long class;
  long bestclass;
  long beststate;
  long total_classes;
  long iter_classes;
  long iter_tokens;
  long iter_tokens_for_CA;
  long repeat_iteration;
  
  SVECTOR *f = NULL;
  SVECTOR *f_emission = NULL;
  SVECTOR *f_trans_1 = NULL;
  SVECTOR *f_trans_2 = NULL;
  SVECTOR *f_comp = NULL;
  SVECTOR *old = NULL;
  SVECTOR *f_base = NULL;
  SVECTOR *f_storage = NULL;
  LATENT_VAR best_hbar_c[2];
  long check_iterations;
  
  int first;
  double score;
  double bestscore;
  double best_class_scores[2];
  
  double trans_score_p1[24];
  double trans_score_p2[24];
  double *weight_vector;
  
  /*Optimization#2*/
  double emission_score;
  double trans_score_1;
  double trans_score_2;
  double comp_score;
  long wnum_id;
  	
  length = x.length;
  bestclass = -1;
  beststate = -1;
  total_classes = sparm->num_classes;
  first = 1; /* toggle for the first iteration */
  score=0.0;
  bestscore=0.0;
  iter_classes=0;
  iter_tokens=0;//iterate tokens
  iter_tokens_for_CA=0;//iterate tokens for Class Assignment
  repeat_iteration = 0;
  check_iterations=0;	
  weight_vector = sm->w;
  best_class_scores[0] = 0.0;
  best_class_scores[1] = 0.0;
  
  /*Optimization#2*/
  emission_score=0.0;
  trans_score_1=0.0;
  trans_score_2=0.0;
  comp_score=0.0;
  wnum_id = 0;
  
  //empty ybar,hbar are passed to this function
  hbar->states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[0].states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[1].states = (long *)my_malloc(sizeof(long) * length);
  
  
  for(iter_classes=1;iter_classes<=total_classes;iter_classes++)
  {

	  /*Changing class assignments for x to find the best class*/
	  /* best_class_scores[0]   = best score for class-1 */
	  /* best_class_scores[1]   = best score for class-2 */
	  /* best_class_scores[2]   = best score for class-3 */
	  /* best_class_scores[n-1] = best score for class-n */
	      best_class_scores[iter_classes - 1] = 0.0; //Zero based index
	      *ybar = iter_classes;
	 	  
  
	 	  //double t1 = get_runtime();
	 	  //Resetting latent_vars for the evaluation of next class
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
		 {
		      hbar->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		 }	
	
	  	  
		for (repeat_iteration=0;repeat_iteration<4;repeat_iteration++) 
			{	
				  /*Start here of repeat iteration*/
				  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				  {
					first = 1; //when calculating score for new superpixel
					for(state=1;state<=sparm->num_states;state++)
					  {
						  if(iter_tokens==0)
						  {
							  hbar->states[iter_tokens] = state;
							  f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_trans_1 = psi_superpixel_trans_1(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_trans_2 = psi_superpixel_trans_2(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  
							  score = 0.0;
							  emission_score=0.0;
							  trans_score_1=0.0;
							  trans_score_2=0.0;
							  comp_score=0.0;
							  
							  //emission_score = sprod_ns(sm->w,f_emission);
							  //trans_score_1 = sprod_ns(sm->w,f_trans_1);
							  //trans_score_2 = sprod_ns(sm->w,f_trans_2);
							  //comp_score = sprod_ns(sm->w,f_comp);
							  
							  
							  while(f_emission != NULL){
								emission_score = emission_score + sprod_ns(sm->w,f_emission);
								old = f_emission;
								f_emission=f_emission->next;
								free_svector_block(old);
							  }
							  while(f_trans_1 != NULL){
								trans_score_1 = trans_score_1 + sprod_ns(sm->w,f_trans_1);
								old = f_trans_1;
								f_trans_1=f_trans_1->next;
								free_svector_block(old);
							  }
							  while(f_trans_2 != NULL){
								trans_score_2 = trans_score_2 + sprod_ns(sm->w,f_trans_2);
								old = f_trans_2;
								f_trans_2=f_trans_2->next;
								free_svector_block(old);
							  }
							  while(f_comp != NULL){
								comp_score = comp_score + sprod_ns(sm->w,f_comp);
								old = f_comp;
								f_comp=f_comp->next;
								free_svector_block(old);
							  }
							  
							  score = emission_score + 	trans_score_1 + trans_score_2 + comp_score; 
							  score = score + loss(y,*ybar,*hbar,sparm);
							  //printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  //emission_score,trans_score_1,trans_score_2 , comp_score);							  
							  /***************************************************************/
							  /**  UPDATE trans_score_p1[state] and trans_score_p2[state]   **/
							  /***************************************************************/
							  /* Get wnum_1 such that trans_score_1 = trans_score_1 - w[wnum]
							   * 								OR
							   * Subtract the score of transitioning from this superpixel into very next superpixel
							  */
							  wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens + 1,*hbar,sparm);
							  //trans_score_1 = trans_score_1 - weight_vector[wnum_id];
							  trans_score_1 = trans_score_1 - sm->w[wnum_id];
							  
							  /*Get wnum_2 such that trans_score_2 = trans_score_2 - w[wnum]
							   *								OR 
							   * Subtract the score of transitioning from next superpixel into this superpixel
							   */
							  wnum_id = get_wnum(*ybar,iter_tokens + 1,iter_tokens,*hbar,sparm);
							  //trans_score_2 = trans_score_2 - weight_vector[wnum_id];
							  trans_score_2 = trans_score_2 - sm->w[wnum_id];
							  
							  /* Populate trans_score_p1[state] and trans_score_p2[state] */
							  trans_score_p1[state]=trans_score_1;
							  trans_score_p2[state]=trans_score_2;
							  
							  /*Free the SVectors*/
							  //free_svector_block(f_emission);
							  //free_svector_block(f_trans_1);
							  //free_svector_block(f_trans_2);
							  //free_svector_block(f_comp);
							  if(f_emission != NULL)
								free_svector(f_emission);
							  if(f_trans_1 != NULL)
								free_svector(f_trans_1);
							  if(f_trans_2 != NULL)
								free_svector(f_trans_2);
							  if(f_comp != NULL)
								free_svector(f_comp);
  							  //f_emission=NULL;
							  //f_trans_1=NULL;
							  //f_trans_2=NULL;
							  //f_comp=NULL;
							  /***************************************************************/
							  
						  } /*End of IF TOKEN == 0 */
						if(iter_tokens!=0)
						  {
							  hbar->states[iter_tokens] = state;
							  f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  
							  score = 0.0;
							  emission_score=0.0;
							  trans_score_1=0.0;
							  trans_score_2=0.0;
							  comp_score=0.0;
							  
							  //emission_score = sprod_ns(sm->w,f_emission);
							  //comp_score = sprod_ns(sm->w,f_comp);
							  
							   while(f_emission != NULL){
								emission_score = emission_score + sprod_ns(sm->w,f_emission);
								old = f_emission;
								f_emission=f_emission->next;
								free_svector_block(old);
							  }
							  while(f_comp != NULL){
								comp_score = comp_score + sprod_ns(sm->w,f_comp);
								old = f_comp;
								f_comp=f_comp->next;
								free_svector_block(old);
							  }
							  
							  //Add transition of current superpixel into previous one to get trans_score_1
							  wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens - 1,*hbar,sparm);
							  if(wnum_id<1059 || wnum_id>2116){ printf("wnum_id=%ld \n",wnum_id);}
							  //trans_score_1 = trans_score_p1[state] + weight_vector[wnum_id];
							  trans_score_1 = trans_score_p1[state] + sm->w[wnum_id];
							  
							  //Add transition of previous sp into current sp to get trans_score_2
							  wnum_id = get_wnum(*ybar,iter_tokens - 1,iter_tokens,*hbar,sparm);
							  if(wnum_id<1059 || wnum_id>2116){printf("wnum_id=%ld \n",wnum_id);}
							  //trans_score_2 = trans_score_p2[state] + weight_vector[wnum_id];
							  trans_score_2 = trans_score_p2[state] + sm->w[wnum_id];
							  
							  score = emission_score + 	trans_score_1 + trans_score_2 + comp_score;
							  score = score + loss(y,*ybar,*hbar,sparm);
						      //printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  //emission_score,trans_score_1,trans_score_2 , comp_score);
							  /***************************************************************/
							  /**  UPDATE trans_score_p1[state] and trans_score_p2[state]   **/
							  /***************************************************************/
							  /* Update trans_score_p1[state] for next token
							   * 								OR	 
							   * Get wnum_1 such that trans_score_1 = trans_score_1 - w[wnum]
							   * 								OR
							   * Subtract the score of transitioning from this superpixel into very next superpixel
							  */
							  if(iter_tokens < (length-1) ){ 
									//Subtract the score of transitioning from this superpixel into very next superpixel
									wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens + 1,*hbar,sparm);
									//trans_score_p1[state] = trans_score_1 - weight_vector[wnum_id];
									trans_score_p1[state] = trans_score_1 - sm->w[wnum_id];
								}
								
							  /* Update trans_score_p2[state] for next token
							   * 								OR
							   * Get wnum_2 such that trans_score_2 = trans_score_2 - w[wnum]
							   *								OR 
							   * Subtract the score of transitioning from next superpixel into this superpixel
							   */	
							  
							  if(iter_tokens < (length-1) ){ 
									//Subtract the score of transitioning very next superpixel into this superpixel
									wnum_id = get_wnum(*ybar,iter_tokens + 1,iter_tokens,*hbar,sparm);
									//trans_score_p2[state] = trans_score_2 - weight_vector[wnum_id];
									trans_score_p2[state] = trans_score_2 - sm->w[wnum_id];
								}
								
							  /*Free the SVectors*/
							  //free_svector_block(f_emission);
							  //free_svector_block(f_comp);
							  if(f_emission != NULL)
								free_svector(f_emission);
							  if(f_comp != NULL)
								free_svector(f_comp);
							  //f_emission=NULL;
							  //f_comp=NULL;
							  /***************************************************************/
							  
							  
						  } /*End of IF TOKEN != 0 */
						  
						  
						  /*Very Strange: Add scores here*/
						  //score = score + loss(y,*ybar,*hbar,sparm);	
						  if( (bestscore < score)  || (first))
						  {
							  bestscore = score;
							  beststate = state;
							  first = 0;
						  }
						  						  
					  }/*End of iteration over state*/
					  
				  	  /*Assigning the best state to this token*/
					  (*hbar).states[iter_tokens] = beststate;
					  
					  if(repeat_iteration == 3){
						  /*We need to store the scores and state of 3rd iteration only*/
						  /* Accumulate the scores of superpixels keeping class constant */
						  best_class_scores[iter_classes - 1] = best_class_scores[iter_classes - 1] + bestscore;
						  best_hbar_c[iter_classes - 1].states[iter_tokens] = beststate;			  
						}
					  
				 }/*End of Iter Tokens [1 L]*/
	  			
	  		}/*End of Repeat Iter [0 3]*/
	  		
			    /* *****************************************************  
			     * *****  Deactivate this area so that it is consistent
			     * *****   with margin_rescaling_fullPsi - Start   
			     * *****************************************************/
			    /*Calculate the feature vector with the best asssigned h's*/
				//f = psi(x,*ybar,*hbar,sm,sparm);
				//score=0.0;
				//best_class_scores[iter_classes - 1] = 0.0;
				/* Calculate score for this class, after finding best h's 
				 * with repeated iterations over tokens */		  
				//while(f != NULL)
				//{
						//score = score + sprod_ns(sm->w,f);
						//old = f;
						//f=f->next;
						//free_svector_block(old);
				//}
				//score = score + loss(y,*ybar,*hbar,sparm);
				//best_class_scores[iter_classes - 1] = score;
				/* *****************************************************  
			     * *****  Deactivate this area so that it is consistent
			     * *****   with margin_rescaling_fullPsi - Finished    
			     * *****************************************************/
			    
			    	  
	  }/*End of Iter Classes [1 2]*/
	  	  
	  	  //best_class_scores[0] = best_class_scores[0] + loss(y,1,*hbar,sparm);
	  	  //best_class_scores[1] = best_class_scores[1] + loss(y,2,*hbar,sparm);
	  	  
		  if(best_class_scores[0] > best_class_scores[1])
		  {*ybar = 1;}
		  else
		  {*ybar = 2;} 
		  //print_wvec(sm);
		  printf("bestclass[1] = %f, bestclass[2] = %f \n",best_class_scores[0] , best_class_scores[1]);
		  //printf("final bestclass = %ld \n",*ybar);
		 
		/* Restoring best set for hidden state */
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
			{
				 hbar->states[iter_tokens] = best_hbar_c[*ybar - 1].states[iter_tokens];  
				 //printf("Initial h[%ld] = %ld , Assigned hbar[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]);
 				 //printf("Initial h[%ld] = %ld , hbar_class2[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,best_hbar_c[1].states[iter_tokens]);
 				 printf("Init_h[%ld] = %ld , hbar[%ld] = %ld , h_c1[%ld]=%ld , h_c2[%ld]=%ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]
															 ,iter_tokens,best_hbar_c[0].states[iter_tokens],iter_tokens,best_hbar_c[1].states[iter_tokens]);

			}
  	  
  	  
  	  //printf("GT_y = %ld [%f], y_bar = %ld [%f] best_class_scores=[%f , %f]\n",y,best_class_scores[y-1],*ybar,best_class_scores[*ybar-1]
			 //,best_class_scores[0],best_class_scores[1]); 
			 
	
  	  
  	  /***************Checking Assignment - Start*******************************/
  	  best_class_scores[0] = 0.0;
  	  for(iter_tokens=0;iter_tokens<length;iter_tokens++){
		  
	  /* Check assignment scores */
	  f = psi(x,1,best_hbar_c[0],sm,sparm);
	  score=0.0;
	  /* Calculate score for this class, after finding best h's 
	     * with repeated iterations over tokens */		  
		 while(f != NULL)
		 {
				score = score + sprod_ns(sm->w,f);
				old = f;
				f=f->next;
				free_svector_block(old);
		}
		score = score + loss(y,1,best_hbar_c[0],sparm);
		best_class_scores[0] += score;
	  }
	  
	  
	  best_class_scores[1] = 0.0;
	  for(iter_tokens=0;iter_tokens<length;iter_tokens++){
	  
	  f = psi(x,2,best_hbar_c[1],sm,sparm);
	  score=0.0;
	  /* Calculate score for this class, after finding best h's 
	     * with repeated iterations over tokens */		  
		 while(f != NULL)
		 {
				score = score + sprod_ns(sm->w,f);
				old = f;
				f=f->next;
				free_svector_block(old);
		}
		score = score + loss(y,2,best_hbar_c[1],sparm);
		best_class_scores[1] += score;
		}
	  /***************Checking Assignment - End*******************************/
	  f = psi(x,2,best_hbar_c[1],sm,sparm);
  	  //printf("***printing fvec for new example with y=2,best_hbar_c[1]****\n");
	  //print_fvec(f);
	  old = NULL;
	  while(f != NULL){
			old = f;
			f=f->next;
			free_svector_block(old);
		}
	  
  	  printf("Evaluation of best_class_scores with full psi =[%f , %f]\n",best_class_scores[0],best_class_scores[1]); 
			 
  	  free(best_hbar_c[0].states);
  	  free(best_hbar_c[1].states);
	  
}
/**************************************************************************************************************************************************/
void find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  /*This is the version which has reduced computation time*/
  long length;
  long state;
  long class;
  long bestclass;
  long beststate;
  long total_classes;
  long iter_classes;
  long iter_tokens;
  long iter_tokens_for_CA;
  long repeat_iteration;
  
  SVECTOR *f = NULL;
  SVECTOR *f_emission = NULL;
  SVECTOR *f_trans_1 = NULL;
  SVECTOR *f_trans_2 = NULL;
  SVECTOR *f_comp = NULL;
  SVECTOR *old = NULL;
  SVECTOR *f_base = NULL;
  SVECTOR *f_storage = NULL;
  LATENT_VAR best_hbar_c[2];
  long check_iterations;
  
  int first;
  double score;
  double bestscore;
  double best_class_scores[2];
  
  double trans_score_p1[24];
  double trans_score_p2[24];
  double *weight_vector;
  
  /*Optimization#2*/
  double emission_score;
  double trans_score_1;
  double trans_score_2;
  double comp_score;
  long wnum_id;
  	
  length = x.length;
  bestclass = -1;
  beststate = -1;
  total_classes = sparm->num_classes;
  first = 1; /* toggle for the first iteration */
  score=0.0;
  bestscore=0.0;
  iter_classes=0;
  iter_tokens=0;//iterate tokens
  iter_tokens_for_CA=0;//iterate tokens for Class Assignment
  repeat_iteration = 0;
  check_iterations=0;	
  weight_vector = sm->w;
  best_class_scores[0] = 0.0;
  best_class_scores[1] = 0.0;
  
  /*Optimization#2*/
  emission_score=0.0;
  trans_score_1=0.0;
  trans_score_2=0.0;
  comp_score=0.0;
  wnum_id = 0;
  
  //empty ybar,hbar are passed to this function
  hbar->states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[0].states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[1].states = (long *)my_malloc(sizeof(long) * length);
  
  
  for(iter_classes=1;iter_classes<=total_classes;iter_classes++)
  {

	  /*Changing class assignments for x to find the best class*/
	  /* best_class_scores[0]   = best score for class-1 */
	  /* best_class_scores[1]   = best score for class-2 */
	  /* best_class_scores[2]   = best score for class-3 */
	  /* best_class_scores[n-1] = best score for class-n */
	      best_class_scores[iter_classes - 1] = 0.0; //Zero based index
	      *ybar = iter_classes;
	 	  
  
	 	  //double t1 = get_runtime();
	 	  //Resetting latent_vars for the evaluation of next class
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
		 {
		      hbar->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		 }	
	
	  	  
		for (repeat_iteration=0;repeat_iteration<4;repeat_iteration++) 
			{	
				  /*Start here of repeat iteration*/
				  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				  {
					first = 1; //when calculating score for new superpixel
					for(state=1;state<=sparm->num_states;state++)
					  {
						  if(iter_tokens==0)
						  {
							  hbar->states[iter_tokens] = state;
							  f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_trans_1 = psi_superpixel_trans_1(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_trans_2 = psi_superpixel_trans_2(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  
							  score = 0.0;
							  emission_score=0.0;
							  trans_score_1=0.0;
							  trans_score_2=0.0;
							  comp_score=0.0;
							  
							  emission_score = sprod_ns(sm->w,f_emission);
							  trans_score_1 = sprod_ns(sm->w,f_trans_1);
							  trans_score_2 = sprod_ns(sm->w,f_trans_2);
							  comp_score = sprod_ns(sm->w,f_comp);
							  
							  score = emission_score + 	trans_score_1 + trans_score_2 + comp_score; 
							  score = score + loss(y,*ybar,*hbar,sparm);
							  //printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  //emission_score,trans_score_1,trans_score_2 , comp_score);							  
							  /***************************************************************/
							  /**  UPDATE trans_score_p1[state] and trans_score_p2[state]   **/
							  /***************************************************************/
							  /* Get wnum_1 such that trans_score_1 = trans_score_1 - w[wnum]
							   * 								OR
							   * Subtract the score of transitioning from this superpixel into very next superpixel
							  */
							  wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens + 1,*hbar,sparm);
							  //trans_score_1 = trans_score_1 - weight_vector[wnum_id];
							  trans_score_1 = trans_score_1 - sm->w[wnum_id];
							  
							  /*Get wnum_2 such that trans_score_2 = trans_score_2 - w[wnum]
							   *								OR 
							   * Subtract the score of transitioning from next superpixel into this superpixel
							   */
							  wnum_id = get_wnum(*ybar,iter_tokens + 1,iter_tokens,*hbar,sparm);
							  //trans_score_2 = trans_score_2 - weight_vector[wnum_id];
							  trans_score_2 = trans_score_2 - sm->w[wnum_id];
							  
							  /* Populate trans_score_p1[state] and trans_score_p2[state] */
							  trans_score_p1[state]=trans_score_1;
							  trans_score_p2[state]=trans_score_2;
							  
							  /*Free the SVectors*/
							  free_svector_block(f_emission);
							  free_svector_block(f_trans_1);
							  free_svector_block(f_trans_2);
							  free_svector_block(f_comp);
  							  f_emission=NULL;
							  f_trans_1=NULL;
							  f_trans_2=NULL;
							  f_comp=NULL;
							  /***************************************************************/
							  
						  } /*End of IF TOKEN == 0 */
						if(iter_tokens!=0)
						  {
							  hbar->states[iter_tokens] = state;
							  f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  
							  score = 0.0;
							  emission_score=0.0;
							  trans_score_1=0.0;
							  trans_score_2=0.0;
							  comp_score=0.0;
							  
							  emission_score = sprod_ns(sm->w,f_emission);
							  comp_score = sprod_ns(sm->w,f_comp);
							  
							  //Add transition of current superpixel into previous one to get trans_score_1
							  wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens - 1,*hbar,sparm);
							  if(wnum_id<1059 || wnum_id>2116){ printf("wnum_id=%ld \n",wnum_id);}
							  //trans_score_1 = trans_score_p1[state] + weight_vector[wnum_id];
							  trans_score_1 = trans_score_p1[state] + sm->w[wnum_id];
							  
							  //Add transition of previous sp into current sp to get trans_score_2
							  wnum_id = get_wnum(*ybar,iter_tokens - 1,iter_tokens,*hbar,sparm);
							  if(wnum_id<1059 || wnum_id>2116){printf("wnum_id=%ld \n",wnum_id);}
							  //trans_score_2 = trans_score_p2[state] + weight_vector[wnum_id];
							  trans_score_2 = trans_score_p2[state] + sm->w[wnum_id];
							  
							  score = emission_score + 	trans_score_1 + trans_score_2 + comp_score;
  							  score = score + loss(y,*ybar,*hbar,sparm);

						      //printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  //emission_score,trans_score_1,trans_score_2 , comp_score);
							  /***************************************************************/
							  /**  UPDATE trans_score_p1[state] and trans_score_p2[state]   **/
							  /***************************************************************/
							  /* Update trans_score_p1[state] for next token
							   * 								OR	 
							   * Get wnum_1 such that trans_score_1 = trans_score_1 - w[wnum]
							   * 								OR
							   * Subtract the score of transitioning from this superpixel into very next superpixel
							  */
							  if(iter_tokens < (length-1) ){ 
									//Subtract the score of transitioning from this superpixel into very next superpixel
									wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens + 1,*hbar,sparm);
									//trans_score_p1[state] = trans_score_1 - weight_vector[wnum_id];
									trans_score_p1[state] = trans_score_1 - sm->w[wnum_id];
								}
								
							  /* Update trans_score_p2[state] for next token
							   * 								OR
							   * Get wnum_2 such that trans_score_2 = trans_score_2 - w[wnum]
							   *								OR 
							   * Subtract the score of transitioning from next superpixel into this superpixel
							   */	
							  
							  if(iter_tokens < (length-1) ){ 
									//Subtract the score of transitioning very next superpixel into this superpixel
									wnum_id = get_wnum(*ybar,iter_tokens + 1,iter_tokens,*hbar,sparm);
									//trans_score_p2[state] = trans_score_2 - weight_vector[wnum_id];
									trans_score_p2[state] = trans_score_2 - sm->w[wnum_id];
								}
								
							  /*Free the SVectors*/
							  free_svector_block(f_emission);
							  free_svector_block(f_comp);
							  f_emission=NULL;
							  f_comp=NULL;
							  /***************************************************************/
							  
							  
						  } /*End of IF TOKEN != 0 */
						  
						  if((bestscore < score)  || (first))
						  {
							  bestscore = score;
							  beststate = state;
							  first = 0;
						  }
						  						  
					  }/*End of iteration over state*/
					  
				  	  /*Assigning the best state to this token*/
					  (*hbar).states[iter_tokens] = beststate;
					  
					  if(repeat_iteration == 3){
						  /*We need to store the scores and state of 3rd iteration only*/
						  /* Accumulate the scores of superpixels keeping class constant */
						  best_class_scores[iter_classes - 1] = best_class_scores[iter_classes - 1] + bestscore;
						  best_hbar_c[iter_classes - 1].states[iter_tokens] = beststate;			  
						}
					  
				 }/*End of Iter Tokens [1 L]*/
	  			
	  		}/*End of Repeat Iter [0 3]*/
	  		
			    
			    /*Calculate the feature vector with the best asssigned h's*/
				f = psi(x,*ybar,*hbar,sm,sparm);
				score=0.0;
				best_class_scores[iter_classes - 1] = 0;
				/* Calculate score for this class, after finding best h's 
				 * with repeated iterations over tokens */		  
				while(f != NULL)
				{
						score = score + sprod_ns(sm->w,f);
						old = f;
						f=f->next;
						free_svector_block(old);
				}
				score = score + loss(y,*ybar,*hbar,sparm);
				best_class_scores[iter_classes - 1] = score;
				
			    	  
	  }/*End of Iter Classes [1 2]*/
	  	  
	  	  
		  if(best_class_scores[0] > best_class_scores[1])
		  {*ybar = 1;}
		  else
		  {*ybar = 2;} 
		 
		/* Restoring best set for hidden state */
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
			{
				 hbar->states[iter_tokens] = best_hbar_c[*ybar - 1].states[iter_tokens];  
			}
  	  
  	  free(best_hbar_c[0].states);
  	  free(best_hbar_c[1].states);
	  
}
/**************************************************************************************************************************************************/
/**************************************************************************************************************************************************/
void find_most_violated_constraint_marginrescaling_backup(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  /*This is the version which has reduced computation time*/
  long length;
  long state;
  long class;
  long bestclass;
  long beststate;
  long total_classes;
  long iter_classes;
  long iter_tokens;
  long iter_tokens_for_CA;
  long repeat_iteration;
  
  SVECTOR *f = NULL;
  SVECTOR *f_emission = NULL;
  SVECTOR *f_trans_1 = NULL;
  SVECTOR *f_trans_2 = NULL;
  SVECTOR *f_comp = NULL;
  SVECTOR *old = NULL;
  SVECTOR *f_base = NULL;
  SVECTOR *f_storage = NULL;
  LATENT_VAR best_hbar_c[2];
  long check_iterations;
  
  int first;
  double score;
  double bestscore;
  double best_class_scores[sparm->num_classes];
  
  double trans_score_p1[24];
  double trans_score_p2[24];
  double *weight_vector;
  
  /*Optimization#2*/
  double emission_score;
  double trans_score_1;
  double trans_score_2;
  double comp_score;
  long wnum_id;
  	
  length = x.length;
  bestclass = -1;
  beststate = -1;
  total_classes = sparm->num_classes;
  first = 1; /* toggle for the first iteration */
  score=0;
  bestscore=0;
  iter_classes=0;
  iter_tokens=0;//iterate tokens
  iter_tokens_for_CA=0;//iterate tokens for Class Assignment
  repeat_iteration = 0;
  check_iterations=0;	
  weight_vector = sm->w;
  
  /*Optimization#2*/
  emission_score=0;
  trans_score_1=0;
  trans_score_2=0;
  comp_score=0;
  wnum_id = 0;
  
  //empty ybar,hbar are passed to this function
  hbar->states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[0].states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[1].states = (long *)my_malloc(sizeof(long) * length);
  
  
  for(iter_classes=1;iter_classes<=total_classes;iter_classes++)
  {

	  /*Changing class assignments for x to find the best class*/
	  /* best_class_scores[0]   = best score for class-1 */
	  /* best_class_scores[1]   = best score for class-2 */
	  /* best_class_scores[2]   = best score for class-3 */
	  /* best_class_scores[n-1] = best score for class-n */
	      best_class_scores[iter_classes - 1] = 0; //Zero based index
	      *ybar = iter_classes;
	 	  
  
	 	  //double t1 = get_runtime();
	 	  //Resetting latent_vars for the evaluation of next class
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
		 {
		      hbar->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		 }	
	
	  	  
		for (repeat_iteration=0;repeat_iteration<4;repeat_iteration++) 
			{	
				  /*Start here of repeat iteration*/
				  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				  {
					first = 1; //when calculating score for new superpixel
					for(state=1;state<=sparm->num_states;state++)
					  {
						  if(iter_tokens==0)
						  {
							  hbar->states[iter_tokens] = state;
							  f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_trans_1 = psi_superpixel_trans_1(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_trans_2 = psi_superpixel_trans_2(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  
							  score = 0;
							  emission_score=0;
							  trans_score_1=0;
							  trans_score_2=0;
							  comp_score=0;
							  
							  emission_score = sprod_ns(sm->w,f_emission);
							  trans_score_1 = sprod_ns(sm->w,f_trans_1);
							  trans_score_2 = sprod_ns(sm->w,f_trans_2);
							  comp_score = sprod_ns(sm->w,f_comp);
							  
							  score = emission_score + 	trans_score_1 + trans_score_2 + comp_score + loss(y,*ybar,*hbar,sparm);
							  printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  emission_score,trans_score_1,trans_score_2 , comp_score);							  
							  /***************************************************************/
							  /**  UPDATE trans_score_p1[state] and trans_score_p2[state]   **/
							  /***************************************************************/
							  /* Get wnum_1 such that trans_score_1 = trans_score_1 - w[wnum]
							   * 								OR
							   * Subtract the score of transitioning from this superpixel into very next superpixel
							  */
							  wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens + 1,*hbar,sparm);
							  //trans_score_1 = trans_score_1 - weight_vector[wnum_id];
							  trans_score_1 = trans_score_1 - sm->w[wnum_id];
							  
							  /*Get wnum_2 such that trans_score_2 = trans_score_2 - w[wnum]
							   *								OR 
							   * Subtract the score of transitioning from next superpixel into this superpixel
							   */
							  wnum_id = get_wnum(*ybar,iter_tokens + 1,iter_tokens,*hbar,sparm);
							  //trans_score_2 = trans_score_2 - weight_vector[wnum_id];
							  trans_score_2 = trans_score_2 - sm->w[wnum_id];
							  
							  /* Populate trans_score_p1[state] and trans_score_p2[state] */
							  trans_score_p1[state]=trans_score_1;
							  trans_score_p2[state]=trans_score_2;
							  
							  /*Free the SVectors*/
							  free_svector_block(f_emission);
							  free_svector_block(f_trans_1);
							  free_svector_block(f_trans_2);
							  free_svector_block(f_comp);
  							  f_emission=NULL;
							  f_trans_1=NULL;
							  f_trans_2=NULL;
							  f_comp=NULL;
							  /***************************************************************/
							  
							  if((bestscore < score)  || (first))
							  {
								  bestscore = score;
								  beststate = state;
								  first = 0;
							  }
							  
						  } /*End of IF TOKEN == 0 */
						  else
						  {
							  hbar->states[iter_tokens] = state;
							  f_emission = psi_superpixel_emission(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  f_comp = psi_superpixel_comp(x,iter_tokens,*ybar,*hbar,sm,sparm);
							  
							  score = 0;
							  emission_score=0;
							  trans_score_1=0;
							  trans_score_2=0;
							  comp_score=0;
							  
							  emission_score = sprod_ns(sm->w,f_emission);
							  comp_score = sprod_ns(sm->w,f_comp);
							  
							  //Add transition of current superpixel into previous one to get trans_score_1
							  wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens - 1,*hbar,sparm);
							  if(wnum_id<1059 || wnum_id>2116){ printf("wnum_id=%ld \n",wnum_id);}
							  //trans_score_1 = trans_score_p1[state] + weight_vector[wnum_id];
							  trans_score_1 = trans_score_p1[state] + sm->w[wnum_id];
							  
							  //Add transition of previous sp into current sp to get trans_score_2
							  wnum_id = get_wnum(*ybar,iter_tokens - 1,iter_tokens,*hbar,sparm);
							  if(wnum_id<1059 || wnum_id>2116){printf("wnum_id=%ld \n",wnum_id);}
							  //trans_score_2 = trans_score_p2[state] + weight_vector[wnum_id];
							  trans_score_2 = trans_score_p2[state] + sm->w[wnum_id];
							  
							  score = emission_score + 	trans_score_1 + trans_score_2 + comp_score + loss(y,*ybar,*hbar,sparm);
						      printf("iter_y:%ld,iter_repeat:%ld,tok:%ld,state:%ld,score:%f (%f+%f+%f+%f)\n",*ybar,repeat_iteration,iter_tokens,state,score,
																											  emission_score,trans_score_1,trans_score_2 , comp_score);

							  /***************************************************************/
							  /**  UPDATE trans_score_p1[state] and trans_score_p2[state]   **/
							  /***************************************************************/
							  /* Update trans_score_p1[state] for next token
							   * 								OR	 
							   * Get wnum_1 such that trans_score_1 = trans_score_1 - w[wnum]
							   * 								OR
							   * Subtract the score of transitioning from this superpixel into very next superpixel
							  */
							  if(iter_tokens < (length-1) ){ 
									//Subtract the score of transitioning from this superpixel into very next superpixel
									wnum_id = get_wnum(*ybar,iter_tokens,iter_tokens + 1,*hbar,sparm);
									//trans_score_p1[state] = trans_score_1 - weight_vector[wnum_id];
									trans_score_p1[state] = trans_score_1 - sm->w[wnum_id];
								}
								
							  /* Update trans_score_p2[state] for next token
							   * 								OR
							   * Get wnum_2 such that trans_score_2 = trans_score_2 - w[wnum]
							   *								OR 
							   * Subtract the score of transitioning from next superpixel into this superpixel
							   */	
							  
							  if(iter_tokens < (length-1) ){ 
									//Subtract the score of transitioning very next superpixel into this superpixel
									wnum_id = get_wnum(*ybar,iter_tokens + 1,iter_tokens,*hbar,sparm);
									//trans_score_p2[state] = trans_score_2 - weight_vector[wnum_id];
									trans_score_p2[state] = trans_score_2 - sm->w[wnum_id];
								}
								
							  /*Free the SVectors*/
							  free_svector_block(f_emission);
							  free_svector_block(f_comp);
							  f_emission=NULL;
							  f_comp=NULL;
							  /***************************************************************/
							  
							  if((bestscore < score)  || (first))
							  {
								  bestscore = score;
								  beststate = state;
								  first = 0;
							  }
							  
						  } /*End of IF TOKEN != 0 */
						  						  
					  }/*End of iteration over state*/
					  
				  	  /*Assigning the best state to this token*/
					  (*hbar).states[iter_tokens] = beststate;
					  
					  if(repeat_iteration == 3){
						  /*We need to store the scores and state of 3rd iteration only*/
						  /* Accumulate the scores of superpixels keeping class constant */
						  best_class_scores[iter_classes - 1] = best_class_scores[iter_classes - 1] + bestscore;
						  best_hbar_c[iter_classes - 1].states[iter_tokens] = beststate;			  
						}
					  
				 }/*End of Iter Tokens [1 L]*/
	  			
	  		}/*End of Repeat Iter [0 3]*/
	  		
			    /* *****************************************************  
			     * *****  Deactivate this area so that it is consistent
			     * *****   with margin_rescaling_fullPsi - Start   
			     * *****************************************************/
			    /*Calculate the feature vector with the best asssigned h's*/
				//f = psi(x,*ybar,*hbar,sm,sparm);
				//score=0;
				//best_class_scores[iter_classes - 1] = 0;
				/* Calculate score for this class, after finding best h's 
				 * with repeated iterations over tokens */		  
				//while(f != NULL)
				//{
					//	score = score + sprod_ns(sm->w,f);
						//old = f;
						//f=f->next;
						//free_svector_block(old);
				//}
				//score = score + loss(y,*ybar,*hbar,sparm);
				//best_class_scores[iter_classes - 1] = score;
				/* *****************************************************  
			     * *****  Deactivate this area so that it is consistent
			     * *****   with margin_rescaling_fullPsi - Finished    
			     * *****************************************************/
			     
				//Copy best states for this class inside best_hbar_c, so that they 
				//can be recovered afterwards
				for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				{
					best_hbar_c[iter_classes - 1].states[iter_tokens] = (*hbar).states[iter_tokens];
				}
		
			    	  
	  }/*End of Iter Classes [1 2]*/
	  				  
		  if(best_class_scores[0] > best_class_scores[1])
		  {*ybar = 1;}
		  else
		  {*ybar = 2;} 
		  //print_wvec(sm);
		  printf("bestclass[1] = %f, bestclass[2] = %f \n",best_class_scores[0] , best_class_scores[1]);
		  //printf("final bestclass = %ld \n",*ybar);
		 
		/* Restoring best set for hidden state */
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
			{
				 hbar->states[iter_tokens] = best_hbar_c[*ybar - 1].states[iter_tokens];  
				 //printf("Initial h[%ld] = %ld , Assigned hbar[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]);
			}
  	  
  	  
  	  //printf("GT_y = %ld [%f], y_bar = %ld [%f] best_class_scores=[%f , %f]\n",y,best_class_scores[y-1],*ybar,best_class_scores[*ybar-1]
			 //,best_class_scores[0],best_class_scores[1]); 
  	  free(best_hbar_c[0].states);
  	  free(best_hbar_c[1].states);
	  
}
/**************************************************************************************************************************************************/

/**************************************************************************************************************************************************/


void find_most_violated_constraint_marginrescaling_fullpsi(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Finds the most violated constraint (loss-augmented inference), i.e.,
  computing argmax_{(ybar,hbar)} [<w,psi(x,ybar,hbar)> + loss(y,ybar,hbar)].
  The output (ybar,hbar) are stored at location pointed by 
  pointers *ybar and *hbar. 
*/
  long length;
  long state;
  long class;
  long bestclass;
  long beststate;
  long total_classes;
  long iter_classes;
  long iter_tokens;
  long iter_tokens_for_CA;
  long repeat_iteration;
  SVECTOR *f = NULL;
  SVECTOR *f2 = NULL;
  SVECTOR *f_base = NULL;
  LATENT_VAR best_hbar_c[2];
  
  
  int first;
  double score,score_2;
  double bestscore;
  double best_class_scores[sparm->num_classes];

  length = x.length;
  bestclass = -1;
  beststate = -1;
  total_classes = sparm->num_classes;
  first = 1; /* toggle for the first iteration */
  score=0;
  bestscore=0;
  iter_classes=0;
  iter_tokens=0;//iterate tokens
  iter_tokens_for_CA=0;//iterate tokens for Class Assignment
  repeat_iteration = 0;

  //empty ybar,hbar are passed to this function
  hbar->states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[0].states = (long *)my_malloc(sizeof(long) * length);
  best_hbar_c[1].states = (long *)my_malloc(sizeof(long) * length);
  
  /*Initialise hbar->states as all superpixels are connected
    with each other. Initialisation is needed because psi 
    calculates transmission features. These transmission features
    depends on the hidden label of current superpixel and hidden labels of 
    all other superpixels of an image    
   */
	   /*  
	    * We have already done this while iterating through classes
	    * See "Resetting latent_vars for the evaluation of next class"
	    * 
	   for(iter_tokens=0;iter_tokens<length;iter_tokens++)
	   {
		   hbar->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		   //hbar->states[iter_tokens] = (long) max_label(x.tokens[iter_tokens]->fvec,sparm->num_features);//initialise all superpixels in this example  
		   //printf("hbar=%ld \n",hbar->states[iter_tokens]);fflush(stdout);
	   }  
	  */

  for(iter_classes=1;iter_classes<=total_classes;iter_classes++)
  {

	  /*Changing class assignments for x to find the best class*/
	  /* best_class_scores[0]   = best score for class-1 */
	  /* best_class_scores[1]   = best score for class-2 */
	  /* best_class_scores[2]   = best score for class-3 */
	  /* best_class_scores[n-1] = best score for class-n */
	      best_class_scores[iter_classes - 1] = 0; //Zero based index
	 	  *ybar = iter_classes;
	 	  
	 	  //Resetting latent_vars for the evaluation of next class
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
		 {
		      hbar->states[iter_tokens] = x.initial_states[iter_tokens];//initialise all superpixels in this example  
		 }	
			  	  
		for (repeat_iteration=0;repeat_iteration<4;repeat_iteration++) {	
				  /*Start here of repeat iteration*/
				  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
				  {
					first = 1; //when calculating score for new superpixel
				  
					for(state=1;state<=sparm->num_states;state++)
					  {
						 
						  hbar->states[iter_tokens] = state;

						  //SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
						  //printf("(old psi) -- class:%ld,iteration:%ld,token:%ld,state:%ld\n",iter_classes,repeat_iteration,iter_tokens,state);
						  f = psi(x,*ybar,*hbar,sm,sparm);
						  //print_fvec(f);
						  /* ***************************************************** */
						  /*Writing classification module because of memory leakage*/
						  //score = impute_score(f,sm) + loss(y,*ybar,*hbar,sparm);
						  //free_svector(f);
							SVECTOR *old;
							old = NULL;
							score = 0;
							
							while(f != NULL){
								score = score + sprod_ns(sm->w,f);
								old = f;
								f=f->next;
								free_svector_block(old);
							}
							score = score + loss(y,*ybar,*hbar,sparm);
							
							
							//printf("score:%f\n",score); fflush(stdout);
							//printf("class: %ld sp#:%ld state:%ld score:%f\n"
							//,iter_classes,iter_tokens,(*hbar).states[iter_tokens]
							//,score); fflush(stdout);
							//printf("sp#:%ld gt_label:%ld calc_label:%ld loss:%f\n"
							//,iter_tokens,y,*ybar
							//,loss(y,*ybar,*hbar,sparm)); fflush(stdout);
						   
						  /* **************************************************** */
						  
						  if((bestscore < score)  || (first)) {
								bestscore = score;
								beststate = state;
								bestclass = iter_classes;
								first = 0;
							}
					  }
					  
					  (*hbar).states[iter_tokens] = beststate;
					  //printf("beststate:%ld bestscore:%f\n",(*hbar).states[iter_tokens],bestscore); fflush(stdout);
					  //printf("checking_class:%ld repeat_iter:%ld token#:%ld beststate:%ld bestscore:%f\n"
							//,iter_classes,repeat_iteration,iter_tokens,hbar->states[iter_tokens]
							//,bestscore); fflush(stdout);
					  if(repeat_iteration == 3){
						  /*We need to store the scores and state of 3rd iteration only*/
						  /* Accumulate the scores of superpixels keeping class constant */
						  best_class_scores[iter_classes - 1] = best_class_scores[iter_classes - 1] + bestscore;
						  best_hbar_c[iter_classes - 1].states[iter_tokens] = beststate;			  
						}
				  }
	  			/*End of repeat iteration */
			}

		  //printf("best_class_scores[%ld] = %f \n",iter_classes,best_class_scores[iter_classes - 1]);
		  //printf("best_hbar_c[%ld] = %ld \n",iter_classes-1,best_hbar_c[iter_classes - 1]->states[20]);
		  
	  }
	  
	  
		  if(best_class_scores[0] > best_class_scores[1])
		  {*ybar = 1;}
		  else
		  {*ybar = 2;} 
		  //printf("final bestclass = %ld \n",*ybar);
		 
		/* Restoring best set for hidden state */
		 for(iter_tokens=0;iter_tokens<length;iter_tokens++)
			{
				 hbar->states[iter_tokens] = best_hbar_c[*ybar - 1].states[iter_tokens];  
				 //printf("Initial h[%ld] = %ld , Assigned hbar[%ld] = %ld \n",iter_tokens,x.initial_states[iter_tokens],iter_tokens,hbar->states[iter_tokens]);
			}
  	   //printf("hbar_c[20] = %ld (Must be the best ) \n",hbar->states[20]);
	  
}

/***********************************************************************************************************************************************/
/***********************************************************************************************************************************************/
LATENT_VAR infer_latent_variables(PATTERN x, LABEL y,LATENT_VAR h ,STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
/*
  Complete the latent variable h for labeled examples, i.e.,
  computing argmax_{h} <w,psi(x,y,h)>. 
*/
  
  double score;
  double bestscore;

  long length;
  long iter_tokens;
  long repeat_iteration;
  
  int state;
  int beststate;
  int first;
  
  SVECTOR *f = NULL;
  
  
  beststate = -1;
  first = 1; /* toggle for the first iteration */
  score=0;
  bestscore=0;
  iter_tokens=0;
  length = x.length;
  repeat_iteration = 0;

 for(repeat_iteration=0;repeat_iteration<4;repeat_iteration++) {
	  for(iter_tokens=0;iter_tokens<length;iter_tokens++)
	  {	 
		  //Here!!!
		  first = 1;
		   
		  for(state=1;state<=sparm->num_states;state++)
		  {
			  h.states[iter_tokens] = state;
			  //SVECTOR *psi(PATTERN x, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
			  f = psi(x,y,h,sm,sparm); 
			  
			  //SVECTOR *psi_superpixel(PATTERN x, long token_id, LABEL y, LATENT_VAR h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm)
			  //f = psi_superpixel(x,iter_tokens,y,h,sm,sparm); 
			  //score = impute_score(f,sm);
			  /* ***************************************************** */
			  /*Writing classification module (impute_score) because of memory leakage*/
			  //score = impute_score(f,sm);
			  //free_svector(f);
			  SVECTOR *old;
			  old = NULL;
			  score = 0;
				  
			  while(f != NULL){
					score = score + sprod_ns(sm->w,f);
					old = f;
					f=f->next;
					free_svector_block(old);
					}

			  /* **************************************************** */
			  if((bestscore < score)  || (first)) {
					bestscore = score;
					beststate = state;
					first = 0;
				}
			  
		  }
		  
		  h.states[iter_tokens] = beststate;
		  
	  }	

	}
	return(h);
}

/************************************************************************/

void optimized_inference_for_rectified_psi(SVECTOR *f,PATTERN x, LATENT_VAR hbar,LABEL ybar,long iter_tokens) {
  /* Part-6: Modifying fvector;
   * Transmission weights are modified successfully (with training file of 6 tokens 2 classes
   * where each example consists of 3 tokens). Now modifying compatibility features.
   * Now just need to convert it in function named: Optimized_Inference()
   */
  
  SVECTOR *base_add = NULL;

  long classes;
  long states;
  long fnum;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
 
  long iter_classes;
  long total_classes;
  long fvec_next_counter;
  long iter_features;
  
 
  long length;
  
  
  //for transmission features
  long first_ind,second_ind;
  long shift_index;
  
  //for compatibility
  long cfeat_ind;
  
  length = x.length;//length of tokens in this example  
  base_add = f; //storing the base address of feature vector passed for modification 
  
    
  classes = 2;
  states = 23;
  fnum=23;
  
  emitbase = 0;
  transbase = emitbase + (states * fnum);
  compbaseindex = transbase + (states * states);
  
  
  fvec_next_counter = 0;
  iter_features=0;
  total_classes = 2;
  
  
  fvec_next_counter = 0;
							
  //Find the starting address of emission weight vector
  while(fvec_next_counter < iter_tokens) 
	{
		fvec_next_counter++;
		f=f->next;
	}
		
  //Modify emission wnums
	shift_index = 0;
	if(hbar.states[iter_tokens] != 1){
		shift_index = emitbase +  ( fnum * (hbar.states[iter_tokens] - 1) ) ;
	}
	//printf("shift index = %ld\n",shift_index);
	
	//Modify this weight vector
	for(iter_features = 0;iter_features<23;iter_features++)
	{
		f->words[iter_features].wnum = (int) (1 + iter_features + shift_index);
	}
	
	//Reach the end of emission weight vector(Start of transmission vector)
	while(fvec_next_counter < length) 
	{
		fvec_next_counter++;
		f=f->next;
	}
						
					  
  //Modify transmission wnum and weights
  long iter_trans_words = 0;

	   for(i=0;i<length;i++){
		   first_ind = iter_tokens;
		   second_ind = i;
		   iter_trans_words = (length *  first_ind) + second_ind;
			if(first_ind==second_ind)
			{
				f->words[iter_trans_words].wnum = 1;
				f->words[iter_trans_words].weight = 0.0;		
			}
			else{
				hlabel_i = hbar.states[first_ind]; 
				hlabel_j = hbar.states[second_ind];
				desired_index = ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				f->words[iter_trans_words].wnum = transbase + desired_index;
				f->words[iter_trans_words].weight = 1.0;	
				
				//(i,j) has been modified. Now modify (j,i)
				first_ind = i;
				second_ind = iter_tokens;
				iter_trans_words = (length *  first_ind) + second_ind;
				
				hlabel_i = hbar.states[first_ind]; 
				hlabel_j = hbar.states[second_ind];
				desired_index = ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				f->words[iter_trans_words].wnum = transbase + desired_index;
				f->words[iter_trans_words].weight = 1.0;	
			}
		   
	  }
					  
					   
	  //Modify compatibility wnum and weights
	  f=f->next; //Reach the start of compatibility features
	  
	  cfeat_ind = 0;

	  i = iter_tokens;
	  cfeat_ind = ( (ybar-1) * (fnum) ) + hbar.states[i];
	  f->words[i].wnum = compbaseindex + cfeat_ind;
	  f->words[i].weight= 1;
	  
	  ///////////////////////////////////////
					  
	 //main part//
	 f = base_add;

}
/***********************************************************************/
void optimized_inference_for_inverted_psi(SVECTOR *f,PATTERN x, LATENT_VAR hbar,LABEL ybar,long iter_tokens) {
  /* Part-6: Modifying fvector;
   * Transmission weights are modified successfully (with training file of 6 tokens 2 classes
   * where each example consists of 3 tokens). Now modifying compatibility features.
   * Now just need to convert it in function named: Optimized_Inference()
   */
  
  SVECTOR *base_add = NULL;

  long classes;
  long states;
  long fnum;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
 
  long iter_classes;
  long fvec_next_counter;
  long iter_features;
  
 
  long length;
  
  
  //for transmission features
  long first_ind,second_ind;
  long shift_index;
  
  //for compatibility
  long cfeat_ind;
  
  length = x.length;//length of tokens in this example  
   
  
    
  classes = 2;
  states = 23;
  fnum=23;
  
  emitbase = 0;
  transbase = emitbase + (states * fnum);
  compbaseindex = transbase + (states * states);
  
  
  fvec_next_counter = 0;
  iter_features=0;
  
  //Important step:Store the base address of f before modification///
   base_add = f; //storing the base address of feature vector passed for modification
   
	
   //Modify compatibility wnum and weights/////
	  cfeat_ind = 0;

	  i = iter_tokens;
	  cfeat_ind = ( (ybar-1) * (fnum) ) + hbar.states[i];
	  f->words[i].wnum = compbaseindex + cfeat_ind;
	  f->words[i].weight= 1;
	  
   ///////////////////////////////////////
   
   
   
   f=f->next; //Reach the start of transmission features
    //Modify transmission wnum and weights
  long iter_trans_words = 0;

	   for(i=0;i<length;i++){
		   first_ind = iter_tokens;
		   second_ind = i;
		   iter_trans_words = (length *  first_ind) + second_ind;
			if(first_ind==second_ind)
			{
				f->words[iter_trans_words].wnum = 1;
				f->words[iter_trans_words].weight = 0.0;		
			}
			else{
				hlabel_i = hbar.states[first_ind]; 
				hlabel_j = hbar.states[second_ind];
				desired_index = ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				f->words[iter_trans_words].wnum = transbase + desired_index;
				f->words[iter_trans_words].weight = 1.0;	
				
				//(i,j) has been modified. Now modify (j,i)
				first_ind = i;
				second_ind = iter_tokens;
				iter_trans_words = (length *  first_ind) + second_ind;
				
				hlabel_i = hbar.states[first_ind]; 
				hlabel_j = hbar.states[second_ind];
				desired_index = ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				f->words[iter_trans_words].wnum = transbase + desired_index;
				f->words[iter_trans_words].weight = 1.0;	
			}
		   
	  }
   	  
	   
  fvec_next_counter = 0;
  //Find the starting address of emission weight vector
  while(fvec_next_counter < iter_tokens) 
	{
		fvec_next_counter++;
		f=f->next;
	}
		
  //Modify emission wnums
	shift_index = 0;
	if(hbar.states[iter_tokens] != 1){
		shift_index = emitbase +  ( fnum * (hbar.states[iter_tokens] - 1) ) ;
	}
	//printf("shift index = %ld\n",shift_index);
	
	//Modify this weight vector
	for(iter_features = 0;iter_features<23;iter_features++)
	{
		f->words[iter_features].wnum = (int) (1 + iter_features + shift_index);
	}
	
	//main part//
	f = base_add;

}
/***********************************************************************/
void optimized_inference(SVECTOR *f,PATTERN x, LATENT_VAR hbar,LABEL ybar,long iter_tokens) {
  /* Part-6: Modifying fvector;
   * Transmission weights are modified successfully (with training file of 6 tokens 2 classes
   * where each example consists of 3 tokens). Now modifying compatibility features.
   * Now just need to convert it in function named: Optimized_Inference()
   */
  
  SVECTOR *base_add = NULL;

  long classes;
  long states;
  long fnum;
  long emitbase, transbase , compbaseindex;
  long i,j,hlabel_i,hlabel_j,desired_index;
 
  long iter_classes;
  long fvec_next_counter;
  long iter_features;
  
 
  long length;
  
  
  //for transmission features
  long first_ind,second_ind;
  long shift_index;
  
  //for compatibility
  long cfeat_ind;
  
  length = x.length;//length of tokens in this example  
   
  
    
  classes = 2;
  states = 23;
  fnum=23;
  
  emitbase = 0;
  transbase = emitbase + (states * fnum);
  compbaseindex = transbase + (states * states);
  
  
  fvec_next_counter = 0;
  iter_features=0;
  
  //Important step:Store the base address of f before modification///
   base_add = f; //storing the base address of feature vector passed for modification
   
	
   //Modify compatibility wnum and weights/////
	  cfeat_ind = 0;

	  i = iter_tokens;
	  cfeat_ind = ( (ybar-1) * (fnum) ) + hbar.states[i];
	  f->words[i].wnum = compbaseindex + cfeat_ind;
	  f->words[i].weight= 1;
	  
   ///////////////////////////////////////
   
   
   
   f=f->next; //Reach the start of transmission features
    //Modify transmission wnum and weights
  long iter_trans_words = 0;

	   for(i=0;i<length;i++){
		   first_ind = iter_tokens;
		   second_ind = i;
		   iter_trans_words = (length *  first_ind) + second_ind;
			if(first_ind==second_ind)
			{
				f->words[iter_trans_words].wnum = 1;
				f->words[iter_trans_words].weight = 0.0;		
			}
			else{
				hlabel_i = hbar.states[first_ind]; 
				hlabel_j = hbar.states[second_ind];
				desired_index = ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				f->words[iter_trans_words].wnum = transbase + desired_index;
				f->words[iter_trans_words].weight = 1.0;	
				
				//(i,j) has been modified. Now modify (j,i)
				first_ind = i;
				second_ind = iter_tokens;
				iter_trans_words = (length *  first_ind) + second_ind;
				
				hlabel_i = hbar.states[first_ind]; 
				hlabel_j = hbar.states[second_ind];
				desired_index = ( (hlabel_i-1)*(fnum) ) + hlabel_j ;
				f->words[iter_trans_words].wnum = transbase + desired_index;
				f->words[iter_trans_words].weight = 1.0;	
			}
		   
	  }
   	  
  
  f=f->next; //Reach the start of emission weight vectors
  	   
  fvec_next_counter = 0;
  fvec_next_counter = (length-1) - iter_tokens; //Because tokens are stored in reverse order
												//i.e. f->word[0] is token # l
												//     f->word[1] is token # l-1 ... and so on
  long jump_next = 0;
 
  //Find the starting address of emission weight vector
  while(jump_next < fvec_next_counter) 
	{
		jump_next++;
		f=f->next;
	}
 	
  //Modify emission wnums
	shift_index = 0;
	if(hbar.states[iter_tokens] != 1){
		shift_index = emitbase +  ( fnum * (hbar.states[iter_tokens] - 1) ) ;
	}
	//printf("shift index = %ld\n",shift_index);
	
	//Modify this weight vector
	for(iter_features = 0;iter_features<23;iter_features++)
	{
		f->words[iter_features].wnum = (int) (1 + iter_features + shift_index);
	}

	//main part//
	f = base_add;
    
}



/***********************************************************************/




double compute_vote(double p_hc,int size_c,double p_hj,int size_j,int image_size){
/*
 Compute vote of all superpixel's latent label for current
 superpixel latent label 
*/ 
	//p_hc: conditional prob of current superpixel given latent label 
	//size_c: size of current superpixel
	//p_hj: conditional prob of jth superpixel given fixed latent label
	//size_j: size of jth superpixel
	//image_size: size of image in pixels
	
	double num = (p_hc * size_c) + (p_hj * size_j);
	double den = image_size;
	double vote = num/den;
	return (vote);  
		
}


/************************************************************************/


/************************************************************************/

void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {

/* Writes structural model sm to file file. */
  FILE *modelfl;
  long j,i,sv_num;
  MODEL *model=sm->svm_model;
  SVECTOR *v;
  //double trial[5];
  //trial[0] = 1.0;trial[1] = 1.0;trial[2] = 1.0;trial[3] = 1.0;trial[4] = 1.0;
  //printf("%d \n",sizeof(sm->w));
  //char str[] = "This is tutorialspoint.com \n";
  //printf("%d \n",sizeof(str));

  
  //printf("Debugging line # %d and file name %s \n",__LINE__,__FILE__); fflush(stdout);
  //printf("sizeof(sm->w ) %d \n", sm->sizePsi);
  
  if ((modelfl = fopen (file, "w")) == NULL)
  { perror (file); exit (1); }
  fprintf(modelfl,"SVM-multiclass Version %s\n","Multiclass");
  fprintf(modelfl,"%d # number of classes\n",
	  sparm->num_classes);
  fprintf(modelfl,"%d # number of states\n",
	  sparm->num_states);
  fprintf(modelfl,"%d # number of base features\n",
	  sparm->num_features);
  fprintf(modelfl,"%d # loss function\n",
	  sparm->loss_function);
  fprintf(modelfl,"%ld # kernel type\n",
	  model->kernel_parm.kernel_type);
  fprintf(modelfl,"%ld # kernel parameter -d \n",
	  model->kernel_parm.poly_degree);
  fprintf(modelfl,"%.8g # kernel parameter -g \n",
	  model->kernel_parm.rbf_gamma);
  fprintf(modelfl,"%.8g # kernel parameter -s \n",
	  model->kernel_parm.coef_lin);
  fprintf(modelfl,"%.8g # kernel parameter -r \n",
	  model->kernel_parm.coef_const);
  fprintf(modelfl,"%ld # Size of Psi \n", sm->sizePsi);	
  fwrite ((void *)sm->w, sizeof(double), sm->sizePsi + 1, modelfl);//original
  //fwrite ((void *)sm->w, sizeof(double), sm->sizePsi, modelfl);
  //fwrite ((void *)trial, sizeof(double), 5, modelfl);
  //fwrite(str , 1 , sizeof(str) , modelfl );
 
  /*
  printf("Writing weights \n"); fflush(stdout);
  for(i=0;i<=sm->sizePsi;i++)
	{printf("w[%d] = %f \n",i,sm->w[i]); fflush(stdout);}	
  */
  
  //printf ("model->sv_num: %d \n",model->sv_num); //unassigned value in this project
	
  fclose(modelfl);

}

/************************************************************************/

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) {

  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
  FILE *modelfl;
  STRUCTMODEL sm;
  long i,queryid,slackid;
  double costfactor;
  char version_buffer[100];
  MODEL *model;

  model = (MODEL *)my_malloc(sizeof(MODEL));

  if ((modelfl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }

  fscanf(modelfl,"SVM-multiclass Version %s\n",version_buffer);
  printf("\n");
  fscanf(modelfl,"%d%*[^\n]\n", &sparm->num_classes);
  printf("sparm->num_classes: %d \n", sparm->num_classes);
  fscanf(modelfl,"%d%*[^\n]\n", &sparm->num_states);  
  printf("sparm->num_states: %d \n", sparm->num_states);
  fscanf(modelfl,"%d%*[^\n]\n", &sparm->num_features);  
  printf("sparm->num_features: %d \n", sparm->num_features);
  fscanf(modelfl,"%d%*[^\n]\n", &sparm->loss_function); 
  printf("sparm->loss_function: %d \n", sparm->loss_function);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.kernel_type);
  printf("kernel_type: %ld \n", model->kernel_parm.kernel_type);
  fscanf(modelfl,"%ld%*[^\n]\n", &model->kernel_parm.poly_degree);
  printf("poly_degree: %ld \n", model->kernel_parm.poly_degree);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.rbf_gamma);
  printf("rbf_gamma: %lf \n", model->kernel_parm.rbf_gamma);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_lin);
  printf("coef_lin: %lf \n", model->kernel_parm.coef_lin);
  fscanf(modelfl,"%lf%*[^\n]\n", &model->kernel_parm.coef_const);
  printf("coef_const: %lf \n", model->kernel_parm.coef_const);
  fscanf(modelfl,"%ld%*[^\n]\n", &sm.sizePsi);
  printf("sizePsi: %ld \n", sm.sizePsi);
  //sm.w = (double*) my_malloc(sizeof(double) * sm.sizePsi);	Original
  sm.w = (double*) my_malloc(sizeof(double) * (sm.sizePsi + 1));	
  fread((void *)sm.w, sizeof(double), sm.sizePsi + 1, modelfl);

  printf("Weights after being read from file\n");
  for(i=0;i<=sm.sizePsi;i++)
	{printf("w[%ld] = %f \n",i,sm.w[i]); fflush(stdout);}	

  // model->index=NULL;
  // model->lin_weights=NULL;

  fclose(modelfl);

  sm.svm_model=model;
	
  return(sm);
}

/************************************************************************/

void free_struct_model(STRUCTMODEL sm, STRUCT_LEARN_PARM *sparm) {
/*
  Free any memory malloc'ed in STRUCTMODEL sm after training. 
*/
  
  free(sm.w);

}

/************************************************************************/

void free_pattern(PATTERN x) {
/*
  Free any memory malloc'ed when creating pattern x.
  NB: internal only, no API
*/

  /* empty for the moment */

}

/************************************************************************/

void free_label(LABEL y) {
/*
  Free any memory malloc'ed when creating label y. 
  NB: internal only, no API
*/

  /* empty for the moment */
  /*Shaukat: Verify this code*/
  //free(y.labels);

} 

/************************************************************************/

void free_latent_var(LATENT_VAR h) {
/*
  Free any memory malloc'ed when creating latent variable h. 
  NB: internal only, no API
*/

  /* empty for the moment */
  free(h.states);

}

/************************************************************************/
double impute_score(SVECTOR *fvec,STRUCTMODEL *sm){
    SVECTOR *vec=NULL;
    SVECTOR *next;
    vec =  copy_svector(fvec);//SVECTOR *copy_svector(SVECTOR *vec)
    next = copy_svector(fvec->next);
    double score = 0;
        
	while(vec != NULL){
		  score = score + sprod_ns(sm->w,vec);
		  free_svector(vec);
		  vec = copy_svector(next);
		  free_svector(next);
        	  next = copy_svector(fvec->next);
	}
	
 return(score);
}
/************************************************************************/

/************************************************************************/
LATENT_VAR copy_latentvar(LATENT_VAR h,long length){
	/* Returns a copy of "h" whose length is "length" */
    
    LATENT_VAR h_i;
    long iter_lvar = 0;
    h_i.states = (long *)my_malloc(sizeof(long) * length);
    
	for(iter_lvar=0;iter_lvar<length;iter_lvar++)
	{
		h_i.states[iter_lvar]=h.states[iter_lvar];
	}
 return(h_i);
}


/************************************************************************/
void print_fvec(SVECTOR *fvec)
{
	SVECTOR *debug_vec=NULL;
    debug_vec =  copy_svector(fvec);//SVECTOR *copy_svector(SVECTOR *vec)
    long iter_words = 0;
	
	while(debug_vec != NULL){
		iter_words=0;
		while(debug_vec->words[iter_words].wnum)
		{
		  printf("iter_words = %ld wnum=%d , weight=%f ",iter_words,debug_vec->words[iter_words].wnum,
		  debug_vec->words[iter_words].weight);
		  //printf("%f\n",debug_vec->words[iter_words].weight);
		  iter_words++;
		  printf("(increased)iter_words = %ld \n",iter_words);
		}
		debug_vec=debug_vec->next;
	}
	free_svector(debug_vec);
}
void print_wvec(STRUCTMODEL *sm)
{
	long iter_elems=0;
	while(iter_elems < sm->sizePsi)
	{
		printf("w[%ld]=%f \n",iter_elems,sm->w[iter_elems]);fflush(stdout);
		iter_elems++;
	}

}
void print_fvec_wvec(SVECTOR *fvec,STRUCTMODEL *sm)
{
	SVECTOR *debug_vec=NULL;
	SVECTOR *old;
    debug_vec =  copy_svector(fvec);//SVECTOR *copy_svector(SVECTOR *vec)
    long iter_words = 0;
    long iter_wvec=0;
   	printf("fvec->start\n");fflush(stdout);
	while(debug_vec != NULL){
		iter_words=0;
		while(debug_vec->words[iter_words].wnum)
		{
		  printf("%d:%f \n ",debug_vec->words[iter_words].wnum,
		  debug_vec->words[iter_words].weight);fflush(stdout);
		  iter_words++;		  
		}
		old=debug_vec;
		debug_vec=debug_vec->next;
		free_svector_block(old);
	}
	printf("fvec->end\n");fflush(stdout);
	printf("wvec->start\n");fflush(stdout);
	
	//for weight vector
	debug_vec =  copy_svector(fvec);//SVECTOR *copy_svector(SVECTOR *vec)
	while(debug_vec != NULL){
		iter_words=0;
		while(debug_vec->words[iter_words].wnum)
		{
		  iter_wvec = debug_vec->words[iter_words].wnum;
		  printf("%ld:%f \n",iter_wvec,sm->w[iter_wvec]);fflush(stdout);
		  iter_words++;		  
		}
		old=debug_vec;
		debug_vec=debug_vec->next;
		free_svector_block(old);
	}
	printf("wvec->end\n");fflush(stdout);
}
/************************************************************************/

void free_struct_sample(SAMPLE s) {
/*
  Free the whole training sample. 
*/

  int i;

  for (i=0;i<s.n;i++) {
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
    free_latent_var(s.examples[i].h);
  }
  free(s.examples);

}

/************************************************************************/

void parse_struct_parameters(STRUCT_LEARN_PARM *sparm) {
/*
  Parse parameters for structured output learning passed 
  via the command line. 
*/
  int i;
    
  for (i=0;(i<sparm->custom_argc)&&((sparm->custom_argv[i])[0]=='-');i++) {
    switch ((sparm->custom_argv[i])[2]) {
      /* your code here */
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
	  default: printf("\nUnrecognized option %s!\n\n", sparm->custom_argv[i]); exit(0);
    }
  }
}
