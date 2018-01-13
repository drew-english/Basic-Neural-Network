#include "ann.h"
#include "battleship.h"

//acivation functions and their derivatives
//relu gives values [0, infinity] while sigmoid gives [0, 1]
//sigmoid is usually better for probabilities while relu is usually better for real values
//linear for range of real valued outputs
double sig(double x)
{
	return 1 / (1 + exp(-x));
}

double d_sig(double x)
{
	return sig(x) * (1 - sig(x));
}

//softplus relu
double relu(double x)
{
	return log10(1 + exp(x));
}

double d_relu(double x)
{
	return exp(x) / 1 + exp(x);
}

double unrelu(double x)
{
	return log(pow(10, x) - 1);
}

double linear(double x)
{
	return x;
}

double d_linear(double x)
{
	return 1;
}

Ann *ann_init(int inputs, int hiddenLayers, int numHidden, int outputs, double (*actHidden)(double x), double(*actOut)(double x))
{

	//checks to make sure inputs are valid
	if (inputs <= 0)
		return 0;
	if (hiddenLayers < 0)
		return 0;
	if (numHidden <= 0)
		return 0;
	if (outputs <= 0)
		return 0;

	int totalWeights = 0, i = 0;
	Ann *temp = NULL;

	//adding 1 to neuron layers for a bias in each layer
	if (hiddenLayers != 0)
		totalWeights = ((inputs + 1) * numHidden) + ((hiddenLayers - 1) * ((numHidden + 1) * numHidden)) + 
		((numHidden + 1) * outputs);
	else
		totalWeights = (inputs + 1) * outputs;

	temp = (Ann *)malloc(sizeof(Ann));

	//makes sure there is enough memory for temp
	if (temp == NULL)
		return 0;

	temp->inputs = inputs;
	temp->hiddenLayers = hiddenLayers;
	temp->numHidden = numHidden;
	temp->totalWeights = totalWeights;
	temp->outputs = outputs;
	temp->actFunHidden = actHidden;
	temp->actFunOut = actOut;
	temp->cache = (double *)malloc(sizeof(double) * totalWeights);
	temp->weights = (double *)malloc(sizeof(double) * totalWeights);
	temp->hiddenNeurons = (double *)malloc(sizeof(double) * (temp->hiddenLayers * temp->numHidden));

	//checks to make sure cache, weights, and hiddenNeurons was validly assigned memory
	if (temp->cache == NULL)
		return 0;
	if (temp->weights == NULL)
		return 0;
	if (temp->hiddenNeurons == NULL)
		return 0;

	//randomly setting weights and initializes cache to all 0
	for (; i < temp->totalWeights; i++)
	{
		temp->weights[i] = 1 * ((double)rand() / (double)RAND_MAX);
		if (rand() % 2)
			temp->weights[i] *= -1;

		temp->cache[i] = 0;
	}

	return temp;

}

double *ann_run(const Ann *temp, double *input)
{

	int i = 0, j = 0, k = 0;
	double *neur = NULL, *output = NULL, sum = 0, *w = NULL;
	
	//neurons of the hidden layers
	neur = temp->hiddenNeurons;
	//output neurons
	output = (double *)malloc(sizeof(double) * temp->outputs);
	w = temp->weights;

	double(*actHidden)(double x) = temp->actFunHidden;
	double(*actOut)(double x) = temp->actFunOut;

	if (output == NULL)
		return 0;

	for (; i < temp->hiddenLayers; i++)
	{
		for (j = 0; j < temp->numHidden; j++)
		{
			sum = *w++;

			for (k = 0; k < (i == 0 ? temp->inputs : temp->numHidden); k++)
			{
				sum += *w++ * (i == 0 ? input[k] : temp->hiddenNeurons[k]);
			}
			
			*neur++ = actHidden(sum);
		}
	}
	
	for (i = 0; i < temp->outputs; i++)
	{
		sum = *w++;

		for (j = 0; j < (temp->hiddenLayers > 0 ? temp->numHidden : temp->inputs); j++)
		{
			sum += *w++ * (temp->hiddenLayers > 0 ? temp->hiddenNeurons[j + ((temp->hiddenLayers - 1) * temp->numHidden)] : 
				input[j]);
		}

		output[i] = actOut(sum);
	}

	return output;

}

void ann_fit(Ann *temp, const double *input, const double *target)
{

	int i = 0, j = 0, k = 0;
	double *delta = NULL, *o = NULL, *h = NULL, *w = NULL, *d = NULL, *n = NULL,
		dx = 0.0, *cache = NULL;
	double(*dOut)(double x);
	double(*dHidden)(double x);

	delta = (double *)malloc(sizeof(double) * (temp->outputs + temp->hiddenLayers * temp->numHidden));
	o = (double *)malloc(sizeof(double) * temp->outputs);

	if (delta == NULL)
		return 0;
	if (o == NULL)
		return 0;

	//runs forwards once to get current outputs to compare to target output
	o = ann_run(temp, input);

	//making a copy of hidden neuron values for the case of the hidden activation being relu
	//as both the unrelu and relu values of the hidden neurons are needed
	if (temp->hiddenLayers)
	{
		h = (double *)malloc(sizeof(double) * temp->hiddenLayers * temp->numHidden);
		for (; i < temp->hiddenLayers * temp->numHidden; i++)
			h[i] = temp->hiddenNeurons[i];
	}

	//note: delta is set up so that first hidden layer neurons comes first,
	//then any other hidden layer's neurons, then output neurons last

	//if neuron activation functions are relu then changing them back to
	//their original numbers will make them easier to work with in the derivative of relu
	if (temp->actFunHidden == relu && h != NULL)
	{
		for (i = 0; i < temp->numHidden * temp->hiddenLayers; i++)
			h[i] = unrelu(temp->hiddenNeurons[i]);
	}
	if (temp->actFunOut == relu)
	{
		for (i = 0; i < temp->outputs; i++)
			o[i] = unrelu(o[i]);
	}

	//sets the derivative needed for activation functions of hidden and output
	if (temp->actFunHidden == relu)
		dHidden = d_relu;
	if (temp->actFunHidden == sig)
		dHidden = d_sig;
	if (temp->actFunHidden == linear)
		dHidden = d_linear;
	if (temp->actFunOut == linear)
		dOut = d_linear;
	if (temp->actFunOut == sig)
		dOut = d_sig;
	if (temp->actFunOut == relu)
		dOut = d_relu;

	/* need to first calculate deltas of each neuron to find that part of the weight and bias update equations */

	//calculate deltas for output layer
	for (i = 0; i < temp->outputs; i++)
	{
		delta[i + temp->hiddenLayers * temp->numHidden] = dOut(o[i]) * (o[i] - target[i]);
	}

	//calculate deltas for hidden layers if any
	for (i = temp->hiddenLayers; i > 0; i--)
	{
		//finds first weight and delta in next layer
		w = temp->weights + ((temp->inputs + 1) * temp->numHidden) + ((temp->numHidden + 1) * temp->numHidden * (i - 1)) + 1;
		d = delta + (temp->numHidden * i);

		for (j = 0; j < temp->numHidden; j++)
		{
			delta[temp->numHidden * (i - 1) + j] = 0;

			for (k = 0; k < (i == temp->hiddenLayers ? temp->outputs : temp->numHidden); k++)
			{
				delta[temp->numHidden * (i - 1) + j] += w[(temp->numHidden + 1) * k + j] * d[k];
			}

			//doing this calculation now to keep hidden all deltas in the same format 
			//(makes it easier when changing weights)
			delta[temp->numHidden * (i - 1) + j] *= dHidden(h[j]);
		}
	}

	/* updating weights to output layer */

	//first weight (and cache) (starting with the bias) to the first delta in output layer
	w = temp->weights + (temp->hiddenLayers ? temp->outputs * temp->numHidden + temp->numHidden * temp->hiddenLayers :
		0);

	cache = temp->cache + (temp->hiddenLayers ? temp->outputs * temp->numHidden + temp->numHidden * temp->hiddenLayers :
		0);
	//first neuron in the previous layer
	n = (temp->hiddenLayers ? temp->hiddenNeurons + temp->numHidden : input);

	for (i = 0; i < temp->outputs; i++)
	{
		for (j = 0; j < (temp->hiddenLayers ? temp->numHidden : temp->inputs) + 1; j++)
		{
			if (j == 0)
			{
				dx = delta[temp->numHidden * temp->hiddenLayers + i];
				//RMSProp calculations with weight update
				*cache = DECAYRATE * *cache + (1 - DECAYRATE) * pow(dx, 2);
				*w++ += -LR * dx / (sqrt(*cache++) + EPS);
			}
			else
			{
				dx = delta[temp->numHidden * temp->hiddenLayers + i] * n[j - 1];
				*cache = DECAYRATE * *cache + (1 - DECAYRATE) * pow(dx, 2);
				*w++ += -LR * dx / (sqrt(*cache++) + EPS);
			}
		}
	}

	/* updating weights to hidden layers if any */

	for (i = temp->hiddenLayers; i > 0; i--)
	{
		//first weight (and cache) (starting with bias) to the delta in current layer
		w = temp->weights + ((i == 1 ? 0 : 1) * (temp->inputs + 1) * temp->numHidden) + 
			((i == 1 ? 0 : i - 2) * (temp->numHidden + 1) * temp->numHidden);

		cache = temp->cache + ((i == 1 ? 0 : 1) * (temp->inputs + 1) * temp->numHidden) +
			((i == 1 ? 0 : i - 2) * (temp->numHidden + 1) * temp->numHidden);
		//first neuron in previous layer
		n = (i == 1 ? input : temp->hiddenNeurons + (temp->numHidden * (i - 1)));

		for (j = 0; j < temp->numHidden; j++)
		{
			for (k = 0; k < (i == 1 ? temp->inputs : temp->numHidden) + 1; k++)
			{
				if (k == 0)
				{
					dx = delta[temp->numHidden * (i - 1) + j];
					*cache = DECAYRATE * *cache + (1 - DECAYRATE) * pow(dx, 2);
					*w++ += -LR * dx / (sqrt(*cache++) + EPS);
				}
				else
				{
					dx = delta[temp->numHidden * (i - 1) + j] * n[k - 1];
					*cache = DECAYRATE * *cache + (1 - DECAYRATE) * pow(dx, 2);
					*w++ += -LR * dx / (sqrt(*cache++) + EPS);
				}
			}
		}
	}

	free(delta);
	free(o);
	free(h);

}


void ann_save(Ann *temp, FILE *out)
{

	int i = 0, actFunHidden = 0, actFunOut = 0;

	fprintf(out, "%d %d %d %d\n", temp->inputs, temp->hiddenLayers,
		temp->numHidden, temp->outputs);
	
	if (temp->actFunHidden == sig)
		actFunHidden = 0;
	if (temp->actFunHidden == relu)
		actFunHidden = 1;
	if (temp->actFunHidden == linear)
		actFunHidden = 2;
	if (temp->actFunOut == sig)
		actFunOut = 0;
	if (temp->actFunOut == relu)
		actFunOut = 1;
	if (temp->actFunOut == linear)
		actFunOut = 2;

	fprintf(out, "%d %d\n", actFunHidden, actFunOut);
	fprintf(out, "%d\n", temp->totalWeights);

	for (; i < temp->totalWeights; i++)
		fprintf(out, "%lf\n", temp->weights[i]);

}

Ann *ann_read(FILE *in)
{

	int i = 0, actFunHidden, actFunOut;
	Ann *temp = NULL;
	Function actFun[3] = { sig, relu, linear };
	
	temp = (Ann *)malloc(sizeof(Ann));

	if (temp == NULL)
		return 0;

	fscanf(in, "%d", &temp->inputs);
	fscanf(in, "%d", &temp->hiddenLayers);
	fscanf(in, "%d", &temp->numHidden);
	fscanf(in, "%d", &temp->outputs);
	fscanf(in, "%d", &actFunHidden);
	fscanf(in, "%d", &actFunOut);
	fscanf(in, "%d", &temp->totalWeights);

	temp->actFunHidden = actFun[actFunHidden];
	temp->actFunOut = actFun[actFunOut];

	temp->cache = (double *)malloc(sizeof(double) * temp->totalWeights);
	temp->hiddenNeurons = (double *)malloc(sizeof(double) * temp->numHidden * temp->hiddenLayers);
	temp->weights = (double *)malloc(sizeof(double) * temp->totalWeights);

	if (temp->hiddenNeurons == NULL || temp->weights == NULL || temp->cache == NULL)
		return 0;
	
	for (; i < temp->totalWeights; i++)
		fscanf(in, "%lf", &temp->weights[i]);

	return temp;

}

//fits the network with a batch of data
void ann_fit_batch(Ann *temp, double *inData, double *targetData, int batchSize)
{

	int i = 0;
	double *in = NULL, *out = NULL;

	for (; i < batchSize; i++)
	{
		//finds the correct place in the in data and target data arrays
		in = inData + (i * temp->inputs);
		out = targetData + (i * temp->outputs);
		ann_fit(temp, in, out);
	}

}

//resets cache to 0
void reset_cache(Ann *temp)
{

	int i = 0;

	for (; i < temp->totalWeights; i++)
		temp->cache[i] = 0;

}

//calculates the error
double calc_cost(double *output, double *target, int length)
{

	int i = 0;
	double error = 0.0;

	for (; i < length; i++)
	{
		error += .5 * pow((output[i] - target[i]), 2);
	}

	return error;

}

//no one likes looking at a blank screen
void progress_bar(const int curr, const int total)
{

	int i = 0;
	double progress = 0.0, tempProg = 0.0;
	char bar[21] = { '\0' };
	
	progress = (double)curr / (total - 1);
	tempProg = progress;

	for (; i < 20; i++, tempProg -= .05)
		tempProg > .04 ? strcat(bar, "#") : strcat(bar, " ");

	curr == 0 ? printf("Progress: [%s] %.1lf%%", bar, progress * 100) : 
		printf("\rProgress: [%s] %.1lf%%", bar, progress * 100);

	if (progress == 1.0)
		puts("\n");

}
