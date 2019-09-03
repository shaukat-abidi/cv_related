
typedef struct pattern {
  /*
    Type definition for input pattern x
  */

  DOC *doc;

} PATTERN;

/************************************************************************/

typedef struct label {
  /*
    Type definition for output label y
  */

  int class;       /* class label */
} LABEL;

/************************************************************************/

typedef struct latent_var {
  /*
    Type definition for latent variable h
  */

  int state;      /* state i.e latent var label */
} LATENT_VAR;

/************************************************************************/

typedef struct example {
  PATTERN x;
  LABEL y; /*LABEL is of type long*/
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

typedef struct word {
  FNUM    wnum;	               /* word number */
  FVAL    weight;              /* word weight */
} WORD;

typedef struct svector {
  WORD    *words;              /* The features/values in the vector by
				  increasing feature-number. Feature
				  numbers that are skipped are
				  interpreted as having value zero. */
  double  twonorm_sq;          /* The squared euclidian length of the
                                  vector. Used to speed up the RBF kernel. */
  char    *userdefined;        /* You can put additional information
				  here. This can be useful, if you are
				  implementing your own kernel that
				  does not work with feature/values
				  representations (for example a
				  string kernel). By default,
				  svm-light will put here the string
				  after the # sign from each line of
				  the input file. */
  long    kernel_id;           /* Feature vectors with different
				  kernel_id's are orthogonal (ie. the
				  feature number do not match). This
				  is used for computing component
				  kernels for linear constraints which
				  are a sum of several different
				  weight vectors. (currently not
				  implemented). */
  struct svector *next;        /* Let's you set up a list of SVECTOR's
				  for linear constraints which are a
				  sum of multiple feature
				  vectors. List is terminated by
				  NULL. */
  double  factor;              /* Factor by which this feature vector
				  is multiplied in the sum. */
} SVECTOR;

typedef struct doc {
  long    docnum;              /* Document ID. This has to be the position of 
                                  the document in the training set array. */
  long    queryid;             /* for learning rankings, constraints are 
				  generated for documents with the same 
				  queryID. */
  double  costfactor;          /* Scales the cost of misclassifying this
				  document by this factor. The effect of this
				  value is, that the upper bound on the alpha
				  for this example is scaled by this factor.
				  The factors are set by the feature 
				  'cost:<val>' in the training data. */
  long    slackid;             /* Index of the slack variable
				  corresponding to this
				  constraint. All constraints with the
				  same slackid share the same slack
				  variable. This can only be used for
				  svm_learn_optimization. */
  long    kernelid;            /* Position in gram matrix where kernel
				  value can be found when using an
				  explicit gram matrix
				  (i.e. kernel_type=GRAM). */
  SVECTOR *fvec;               /* Feature vector of the example. The
				  feature vector can actually be a
				  list of feature vectors. For
				  example, the list will have two
				  elements, if this DOC is a
				  preference constraint. The one
				  vector that is supposed to be ranked
				  higher, will have a factor of +1,
				  the lower ranked one should have a
				  factor of -1. */
} DOC;

typedef struct kernel_parm {
  long    kernel_type;   /* 0=linear, 1=poly, 2=rbf, 3=sigmoid,
			    4=custom, 5=matrix */
  long    poly_degree;
  double  rbf_gamma;
  double  coef_lin;
  double  coef_const;
  char    custom[50];    /* for user supplied kernel */
  MATRIX  *gram_matrix;  /* here one can directly supply the kernel
			    matrix. The matrix is accessed if
			    kernel_type=5 is selected. */
} KERNEL_PARM;

typedef struct model {
  long    sv_num;	
  long    at_upper_bound;
  double  b;
  DOC     **supvec;
  double  *alpha;
  long    *index;       /* index from docnum to position in model */
  long    totwords;     /* number of features */
  long    totdoc;       /* number of training documents */
  KERNEL_PARM kernel_parm; /* kernel */

  /* the following values are not written to file */
  double  loo_error,loo_recall,loo_precision; /* leave-one-out estimates */
  double  xa_error,xa_recall,xa_precision;    /* xi/alpha estimates */
  double  *lin_weights;                       /* weights for linear case using
						 folding */
  double  maxdiff;                            /* precision, up to which this 
						 model is accurate */
} MODEL;



