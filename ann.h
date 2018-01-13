#ifndef ann
#define ANN

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include "battleship.h"

//parameters for updating weights (uses RMSProp)
#define LR .001
#define DECAYRATE .95
#define EPS .000001

typedef struct ann
{
	//number of inputs, number of hidden layers, # of neurons in each hidden layer (assuming same for both), 
	//and outputs
	int inputs, hiddenLayers, numHidden, outputs;

	//cache that allows us to update how much we change the weights
	//(based on the RMSProp model)
	double *cache;

	//activation function for outputs
	double(*actFunOut)(double x);

	//activation function for the hidden layer neurons
	double(*actFunHidden)(double x);

	//number of total weights
	int totalWeights;

	//neuron values of hidden layer
	double *hiddenNeurons;

	//all of the weights (total weights long)
	double *weights;

} Ann;

typedef double(*Function)(double x);

//memory struct to hold information for experience replay
typedef struct memory
{
	double *oldState;
	int action;
	double reward;
	double *newState;
} Memory;

//acitvation functions and their derivatives
double sig(double x);
double d_sig(double x);
double relu(double x);
double d_relu(double x);
double unrelu(double x);
double linear(double x);
double d_linear(double x);

//base network functions
Ann *ann_init(int inputs, int hiddenLayers, int numHidden, int outputs, double(*actHidden)(double x), double(*actOut)(double x));
double *ann_run(const Ann *temp, double *input);
void ann_fit(Ann *temp, double *input, const double *target);
void ann_save(Ann *temp, FILE *out);
Ann *ann_read(FILE *in);
int q_max(double *output, int numOutputs);
void ann_fit_batch(Ann *temp, double *inData, double *targetData, int batchSize);
void reset_cache(Ann *temp);
double calc_cost(double *output, double *target, int length);


//functions specific to this game
void ann_train(Ann *temp);
double *normalize_game_board(char *board);
int *to_coords(const int action);
void append(double *dest, double *source, int start, int length);
int test_network(Ann *temp);

//for something to look at while the network is training
void progress_bar(const int curr, const int total);

#endif