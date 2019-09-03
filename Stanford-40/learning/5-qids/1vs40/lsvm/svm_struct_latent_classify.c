/************************************************************************/
/*                                                                      */
/*   svm_struct_latent_classify.c                                       */
/*                                                                      */
/*   Classification Code for Latent SVM^struct                          */
/*                                                                      */
/*   Author: Chun-Nam Yu                                                */
/*   Date: 9.Nov.08                                                     */
/*                                                                      */
/*   This software is available for non-commercial use only. It must    */
/*   not be modified and distributed without prior permission of the    */
/*   author. The author is not responsible for implications from the    */
/*   use of this software.                                              */
/*                                                                      */
/************************************************************************/

#include <stdio.h>
#include "svm_struct_latent_api.h"


void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, STRUCT_LEARN_PARM *sparm);
readLine(char *file,long *sp_ids);
void scan_file(char *other_info,long *total_lines); 
void scale_dataset(SAMPLE *dataset,double param_a,double param_b);

int main(int argc, char* argv[]) {
  double avgloss,l;
  long i, correct;

  char testfile[1024];
  char modelfile[1024];

  STRUCTMODEL model;
  STRUCT_LEARN_PARM sparm;
  LEARN_PARM lparm;
  KERNEL_PARM kparm;

  SAMPLE testsample;
  LABEL y;
  LATENT_VAR h; 
   
  /*Shaukat:for superpixel intialization*/
  /****************************/
  long *initlv;
  char file_initlv[255]="latent_vars_c1andc36.txt";//init latent vars
  long total_lines;
  int iter_loop;
  double param_scale;
  double param_translate;
  /****************************/

  /* read input parameters */
  read_input_parameters(argc,argv,testfile,modelfile,&sparm);

  /* read model file */
  printf("Reading model..."); fflush(stdout);
  model = read_struct_model(modelfile, &sparm);
  printf("kernel_type AFTER: %ld \n", model.svm_model->kernel_parm.kernel_type);
  printf("done.\n");

  /* read test examples */
  printf("Reading test examples..."); fflush(stdout);
  testsample = read_struct_examples(testfile,&sparm);
  printf("done.\n");

  printf ("Value of kparm->kernel_type BEFORE: %ld\n", kparm.kernel_type); fflush(stdout);
  printf("kernel_type AGAIN: %ld \n", model.svm_model->kernel_parm.kernel_type);
  init_struct_model(testsample,&model,&sparm,&lparm,&kparm);
  printf ("Value of kparm->kernel_type AFTER: %ld\n", kparm.kernel_type); fflush(stdout);
  
  /****Scale Dataset by scaling parameters and translation factor****/
  param_scale = 100;
  param_translate = 0;
  //void scale_dataset(SAMPLE *sample,double param_a,double param_b);
  printf("Scaling dataset ....");
  scale_dataset(&testsample,param_scale,param_translate);
  printf("scaled. \n");
  /******************************************************************/
  
  model.svm_model->lin_weights = (double *) malloc(sizeof(double) * (model.sizePsi+1) );

  /*
  model.svm_model->lin_weights[1] = -4.0338993;
  model.svm_model->lin_weights[2] = -3.1249278;
  model.svm_model->lin_weights[3] = -0.40888888;
  model.svm_model->lin_weights[4] = -1.1885773;
  model.svm_model->lin_weights[5] = 0.10972907;
  model.svm_model->lin_weights[6] = 0.050000001;
  model.svm_model->lin_weights[7] = 3.3390911;
  model.svm_model->lin_weights[8] = 2.9744675;
  model.svm_model->lin_weights[9] = 0.33333334;
  model.svm_model->lin_weights[10] = 1.8833855; 
  model.svm_model->lin_weights[11] = 0.040731132;
  model.svm_model->lin_weights[12] = 0.025555555;
  model.w = model.svm_model->lin_weights; 
  */

  /* the following weights, learned by svm_multiclass with 4 classes, give an error of only 0.0722:
  model.svm_model->lin_weights[1] = -27.2146;
  model.svm_model->lin_weights[2] = 30.791941;
  model.svm_model->lin_weights[3] = 29.362797;
  model.svm_model->lin_weights[4] = -1.0828806;
  model.svm_model->lin_weights[5] = -28.013737;
  model.svm_model->lin_weights[6] = 60.172184;
  model.svm_model->lin_weights[7] = 8.8553143;
  model.svm_model->lin_weights[8] = 15.715963;
  model.svm_model->lin_weights[9] = -38.385345;
  model.svm_model->lin_weights[10] = 19.442165; 
  model.svm_model->lin_weights[11] = -18.494167;
  model.svm_model->lin_weights[12] = -51.149635;
  model.w = model.svm_model->lin_weights;
  */

  /*Shaukat: Copying Weight Vector to lin_weights linearly*/
  for (i=1;i<=model.sizePsi;i++) {
	  model.svm_model->lin_weights[i] = model.w[i];
  }

  for (i=1;i<=model.sizePsi;i++) {
	  printf("lin_weights[%ld] = %f \n",i,model.svm_model->lin_weights[i]);
  }
  ///////////////////////////////////////////////////////// 
  avgloss = 0.0;
  correct = 0;
  
  /*New for Static Action Recognition*/
  /* We need to initialise superpixels 
   * latent values. Otherwise we will 
   * receive "segmentation fault error" 
   */
   
  /*Reading superpixel's size from file*/
  scan_file(file_initlv,&total_lines); /* get total lines in input file */
  total_lines = total_lines - 1; //Because last line is always empty
  initlv = (long*) my_malloc( sizeof(long) * (total_lines+1) );
  
  readLine(file_initlv,initlv);
  
  /* impute latent variable - Necessary for L-SVM Classification  */
   init_latent_variables(&testsample,&lparm,&model,&sparm,initlv,total_lines);
  /*init_latent_variables(&testsample,&lparm,&model,&sparm);*/
  printf("num_features : %d \n" , sparm.num_features);fflush(stdout);
  /**********************************/
  
  for (i=0;i<testsample.n;i++) {	
	// printf("Number of classes %d\n",sparm.num_classes);
	printf("ex %ld: \n",i);
    
    /*Original classify_struct_example*/ //classify_struct_example(testsample.examples[i].x,&y,&h,&model,&sparm);
    
    classify_struct_example(testsample.examples[i].x,&y,&testsample.examples[i].h,&model,&sparm);
	
	printf(" GT-label: %ld , P-label: %ld \n ",testsample.examples[i].y,y);    
	//printf("%d,%d \n",y.class,h.state);
    l = loss(testsample.examples[i].y,y,testsample.examples[i].h,&sparm);
    avgloss += l;
    if (l==0) correct++;
    
    //free_label(y);//Shaukat Adding (prevent memory leakage)
  }

  //free_label(y);
  //free_latent_var(h); 

  printf("Average loss on test set: %.4f\n", avgloss/testsample.n);
  printf("Actual loss on test set: %.4f\n", avgloss);//Added for testing the code
  printf("Zero/one error on test set: %.4f\n", 1.0 - ((float) correct)/testsample.n);
  printf("Correctly classified examples %ld out of %d \n",correct,testsample.n);//Added for testing the code

  free_struct_sample(testsample);
  free_struct_model(model,&sparm);

  return(0);

}

void read_input_parameters(int argc, char **argv, char *testfile, char *modelfile, STRUCT_LEARN_PARM *sparm) {

  long i;
  
  /* set default */
  strcpy(modelfile, "svm_model");
  sparm->custom_argc = 0;

  for (i=1;(i<argc)&&((argv[i])[0]=='-');i++) {
    switch ((argv[i])[1]) {
      case '-': strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);i++; strcpy(sparm->custom_argv[sparm->custom_argc++],argv[i]);break;  
      default: printf("\nUnrecognized option %s!\n\n",argv[i]); exit(0);    
    }
  }

  if (i>=argc) {
    printf("\nNot enough input parameters!\n\n");
    exit(0);
  }

  strcpy(testfile, argv[i]);
  strcpy(modelfile, argv[i+1]);

  parse_struct_parameters(sparm);

}
readLine(char *file,long *sp_ids)
{
       FILE * fp;
       char * line = NULL;
       size_t len = 0;
       ssize_t read;
	   long num,iter_ids;
	   char c;
	   iter_ids=0;

       if ((fp = fopen (file, "r")) == NULL)
	  { perror (file); exit (1); }
	  
	  fscanf(fp,"%ld",&num);
	  sp_ids[iter_ids] = num;
      //printf("sp_ids[%d] = %d \n",iter_ids,sp_ids[iter_ids]);

 
	  while(!feof(fp)) {
		//printf("%d \n",num);
		fscanf(fp,"%ld",&num);
		iter_ids++;
		sp_ids[iter_ids] = num;
		//printf("sp_ids[%d] = %d \n",iter_ids,sp_ids[iter_ids]);
	  }
	  fclose(fp);
	         
}
void scan_file(char *file, long int *nol) 
     /* Joachim's code modified: find total lines in file */
{
  FILE *fl;
  int ic;
  char c;
  long current_length,current_wol;

  if ((fl = fopen (file, "r")) == NULL)
  { perror (file); exit (1); }
  (*nol)=1;
  while((ic=getc(fl)) != EOF) {
    c=(char)ic;
    current_length++;
    if(c == '\n') {
      (*nol)++;
    }
  }
  fclose(fl);
}
void scale_dataset(SAMPLE *dataset,double param_a,double param_b)
{
	/*
	 * Scales Dataset: scaling by param_a and translating by param_b
	 * new_dataset= (param_a * old_dataset) + param_b 
	 * 
	 */
	 
	 long iter_ex;
	 long iter_tokens;
	 long total_tokens;
	 register WORD *ai;
	 int n;
	 EXAMPLE *ex_pointer;
	 SVECTOR *fvec;
	 
	 n=dataset->n;
	 ex_pointer = dataset->examples;
	 
	 for (iter_ex=0;iter_ex<n;iter_ex++)
	 {
		 total_tokens = ex_pointer[iter_ex].x.length;
		 for(iter_tokens = 0;iter_tokens<total_tokens;iter_tokens++)
		 {
			 fvec = ex_pointer[iter_ex].x.tokens[iter_tokens]->fvec;
			 ai = fvec->words;
			  while (ai->wnum) {
				ai->weight = (ai->weight *  param_a) +  param_b;
				ai++;
				  }
		 }
		 
	 }
	 	
}
