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

# include "svm_light/svm_common.h"
# include "svm_light/svm_learn.h"

/* default precision for solving the optimization problem */
# define DEFAULT_EPS         0.1 

/* default loss rescaling method: 1=slack_rescaling, 2=margin_rescaling */
# define DEFAULT_RESCALING   2

/* default loss function: */
# define DEFAULT_LOSS_FCT    0

/* default optimization algorithm to use: */
# define DEFAULT_ALG_TYPE    4

/* store Psi(x,y) once instead of recomputing it every time: */
# define USE_FYCACHE         0

/* decide whether to evaluate sum before storing vectors in constraint
   cache: 
   0 = NO, 
   1 = YES (best, if sparse vectors and long vector lists), 
   2 = YES (best, if short vector lists),
   3 = YES (best, if dense vectors and long vector lists) */
# define COMPACT_CACHED_VECTORS 2

/* minimum absolute value below which values in sparse vectors are
   rounded to zero. Values are stored in the FVAL type defined in svm_common.h 
   RECOMMENDATION: assuming you use FVAL=float, use 
     10E-15 if COMPACT_CACHED_VECTORS is 1 
     10E-10 if COMPACT_CACHED_VECTORS is 2 or 3 
*/
# define COMPACT_ROUNDING_THRESH 10E-15

/************************************************************************/
# define LABEL long       /* the type used for storing class (target) */


/************************************************************************/

typedef struct pattern {
 
  /* this defines the x-part of a training example, e.g. the structure
     for storing a natural language sentence in NLP parsing */
  DOC     **tokens;
  long    length;
  
  /*
    Type definition for input pattern x
  */
  
  /* This will be very handy. Store intial assignments of latent variables */
  /* Not adding this line in latent var because then we might need to change 
   * function declaration for instance, find_most_violated_constraint() */ 
  long *initial_states;

  //DOC *doc;

} PATTERN;

/************************************************************************/

//typedef struct label {
  
  /* this defines the y-part (the label) of a training example,
     e.g. the parse tree of the corresponding sentence. */
  /*
  long    *labels;
  long    length;
  */
  
  /*
    Type definition for output label y
  */
//  long class;       /* class label */

//} LABEL;

/************************************************************************/

typedef struct latent_var {
  /*
    Type definition for latent variable h
  */

  //int state;      /* state i.e latent var label */
  long *states;
   
  
} LATENT_VAR;

/************************************************************************/

typedef struct example {
  PATTERN x;
  LABEL y;
  LATENT_VAR h;
} EXAMPLE;

/************************************************************************/

typedef struct sample {
  int n;
  EXAMPLE *examples;
} SAMPLE;

/************************************************************************/

typedef struct structmodel {
  double *w;          /* pointer to the learned weights */
  MODEL  *svm_model;  /* the learned SVM model */
  long   sizePsi;     /* maximum number of weights in w */
  /* other information that is needed for the stuctural model can be
     added here, e.g. the grammar rules for NLP parsing */
  long n;             /* number of examples */
} STRUCTMODEL;

/************************************************************************/

typedef struct struct_learn_parm {
  double epsilon;              /* precision for which to solve quadratic program */
  long newconstretrain;        /* number of new constraints to accumulate before recomputing the QP solution */
  double C;                    /* trade-off between margin and loss */
  char   custom_argv[20][1000]; /* string set with the -u command line option */
  int    custom_argc;          /* number of -u command line options */
  int    slack_norm;           /* norm to use in objective function
                                  for slack variables; 1 -> L1-norm, 2 -> L2-norm */
  int    loss_type;            /* selected loss function from -r command line option. Only margin rescaling (2) is supported here*/
  int    loss_function;        /* select between different loss functions via -l command line option */
  /* add your own variables: */
  int num_classes;
  int num_features;
  int num_states;
} STRUCT_LEARN_PARM;
