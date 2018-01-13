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

	//temp = (Ann *)malloc(sizeof(Ann) + sizeof(double) * totalWeights);
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

	//*LR -= 1 / 50000;

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

	reset_cache(temp);

}

//resets cache to 0
void reset_cache(Ann *temp)
{

	int i = 0;

	for (; i < temp->totalWeights; i++)
		temp->cache[i] = 0;

}

//returns an action based on the max q-value the network gives
int q_max(double *output, int numOutputs)
{

	int i = 0, maxIndex = 0;
	double *o = output, maxNum = 0;

	for (; i < numOutputs; i++, o++)
	{
		if (*o > maxNum)
		{
			maxNum = *o;
			maxIndex = i;
		}
	}

	return maxIndex;

}

//specific to this battleship game
double *normalize_game_board(char *board)
{

	int i = 0;
	char *b = board;
	double *normal = NULL;

	normal = (double *)malloc(sizeof(double) * 100);
	if (normal == NULL)
		return 0;

	for (; i < 100; i++, b++)
	{
		//hides the ships from the network
		if (*b == '~') //|| *b == 'c' || *b == 'b' || *b == 'r' || *b == 's' || *b == 'd')
			normal[i] = -1.0;
		else if (*b == 'm' || *b == '*')
			normal[i] = 0.0;
		else
			normal[i] = 1.0;
	}

	return normal;

}

//trains the network to play battleship 
//based on the q-learning with experience replay approach 
void ann_train(Ann *temp)
{

	//calculation variables
	int i = 0;
	int j = 0;
	int moves = 0;
	int epochs = 1000;
	double epsilon = 1;
	double gamma = .8;
	int action = 0;
	int prevGuesses[100] = { 0 };
	double reward = 0.0;
	double *qval = NULL;
	double *state = NULL;
	double *newState = NULL;
	double cost = 0.0;

	qval = (double *)malloc(sizeof(double) * 100);
	state = (double *)malloc(sizeof(double) * 100);
	newState = (double *)malloc(sizeof(double) * 100);
	if (qval == NULL || state == NULL || newState == NULL)
		return 0;

	//experience replay variables
	int bufferSize = 100;
	int batchSize = 20;
	int numStored = 0;
	int currStoreVal = 0;
	int miniBatchCurr = 0;
	double *mOldState = NULL;
	int mAction = 0;
	double mReward = 0.0;
	double maxQ = 0.0;
	double update = 0.0;
	double *mNewState = NULL;
	double *oldQval = NULL;
	double *newQval = NULL;
	double *xTrain = NULL;
	double *yTrain = NULL;
	Memory *replay = NULL;
	double *y = NULL;

	y = (double *)malloc(sizeof(double) * 100);
	replay = (Memory *)malloc(sizeof(Memory) * bufferSize);
	oldQval = (double *)malloc(sizeof(double) * 100);
	newQval = (double *)malloc(sizeof(double) * 100);
	xTrain = (double *)malloc(sizeof(double) * 100 * batchSize);
	yTrain = (double *)malloc(sizeof(double) * 100 * batchSize);
	
	if (y == NULL || replay == NULL || oldQval == NULL || newQval == NULL ||
		xTrain == NULL || yTrain == NULL)
		return 0;

	//game state variable
	char board[10][10] = { '\0' };
	int game = 0;
	int *coords = NULL;
	Shot shot = { 0, '~' };
	int shipsSunk = 0;

	for (; i < epochs; i++)
	{
		//initialize game state
		initialize_game_board(board);
		randomly_place_ships_on_board(board);
		shipsSunk = 0;
		moves = 0;
		game = 0;
		cost = 0;

		while (game == 0)
		{
			//observe q values for current state
			state = normalize_game_board(board);
			qval = ann_run(temp, state);

			//select action (based on e-greedy)
			if (rand() / RAND_MAX < epsilon)
				//makes sure random actions cannot occur at the same coordinates
				action = rand() % 100;
			else
				action = q_max(qval, 100);
			coords = to_coords(action);

			//take action, observe new state
			take_action(board, coords, &shot);
			newState = normalize_game_board(board);

			//observe reward of newState
			reward = get_reward(board, &shipsSunk, shot);
			
			//experience replay buffer storage
			if (numStored < bufferSize)
			{
				replay[numStored].oldState = state;
				replay[numStored].action = action;
				replay[numStored].reward = reward;
				replay[numStored].newState = newState;
				numStored++;
			}
			else
			{
				//when buffer is full, rewrite old memories
				if (currStoreVal < bufferSize - 1)
					currStoreVal++;
				else
					currStoreVal = 0;
				replay[currStoreVal].oldState = state;
				replay[currStoreVal].action = action;
				replay[currStoreVal].reward = reward;
				replay[currStoreVal].newState = newState;

				//randomly sample replay buffer
				miniBatchCurr = rand() % (bufferSize - batchSize);
				//for each memory in the minibatch
				for (j = 0; j < batchSize; j++, miniBatchCurr++)
				{
					//set the memory variables
					mOldState = replay[miniBatchCurr].oldState;
					mAction = replay[miniBatchCurr].action;
					mReward = replay[miniBatchCurr].reward;
					mNewState = replay[miniBatchCurr].newState;

					//observe q-values of old state and new state
					oldQval = ann_run(temp, mOldState);
					newQval = ann_run(temp, mNewState);
					maxQ = q_max(newQval, 100);
					//copies the contents of oldQval into y
					memcpy(y, oldQval, sizeof(double) * 100);

					//calculate target based on oldState and newState q-values
					if (mReward != 10) //non-terminal state
						update = mReward + gamma * maxQ;
					else
						update = mReward;
					//update y with the target to obtain target output
					y[mAction] = update;

					//append the old state and target to the train arrays
					append(xTrain, mOldState, j * 100, 100);
					append(yTrain, y, j * 100, 100);
				}

				//train network for the mini batch
				ann_fit_batch(temp, xTrain, yTrain, batchSize);
			}

			//if terminal game state then end this game
			//also capping the number of moves per game
			if (reward == 10 || moves == 125)
				game = 1;
			moves++;

			//need to free to prevent memory leaks as
			//new memory is allocated for coords each time
			free(coords);
		}
		//decrement epsilon over time
		if (epsilon > .1)
			epsilon -= (double) 1.0 / epochs;

		progress_bar(i, epochs);
		printf(", Moves: %d", moves);
	}

}

//tests the network and returns the number of moves it takes
//for it to win
int test_network(Ann *temp)
{

	int i = 0, stats[2][2] = { 0 }, shipsSunk = 0,
		*coords = NULL, action = 0;
	double *normalBoard = NULL, pct = 0;
	char board[10][10] = { '\0' };
	Shot shot = { 0, '~' };

	initialize_game_board(board);
	randomly_place_ships_on_board(board);

	for (; shipsSunk < 5; i++)
	{
		normalBoard = normalize_game_board(board);
		action = q_max(ann_run(temp, normalBoard), 100);
		coords = to_coords(action);

		shot = update_board(board, coords, 0, stats);

		if (shot.hitMiss == 1 && shot.current != '*')
		{
			if (check_if_sunk_ship(board, shot.current))
				shipsSunk++;
		}

		free(normalBoard);
		free(coords);

		//breaks the loop if the network can't sink all the ships
		if (i > 300)
			shipsSunk = 5;
	}

	if (i < 300)
	{
		pct = ((double)stats[0][0] * 100 / (stats[0][0] + stats[0][1]));
		printf("Won in %d moves, with %.2lf%% accuracy\n\n", i, pct);
	}
	else
		printf("Could not sink all the ships :(\n\n");

	return i;

}

int *to_coords(const int action)
{

	int *coords = NULL, i = 0, j = 0;

	coords = (int *)malloc(sizeof(int) * 2);

	for (; i < 10; i++)
	{
		for (j = 0; j < 10; j++)
		{
			if (action == i * 10 + j)
			{
				coords[0] = i;
				coords[1] = j;
				return coords;
			}
		}
	}

}

//gets a random action that hasn't been guessed before so
//the game doesn't go forever on random moves
int rand_action(int *prevGuess, int numGuess)
{

	int a = 0, flag = 0, i = 0;

	do
	{
		flag = 0;
		a = rand() % 100;

		for (i = 0; i < numGuess; i++)
			if (a == prevGuess[i])
				flag = 1;
	} while (flag != 0);

	prevGuess[numGuess] = a;
	
	return a;

}

//update board function to handle the way battleship was written
void take_action(char *board, int *coords, Shot *shot)
{

	//variables that battleship uses but we will not necessarily need
	int stats[2][2] = { 0 };

	*shot = update_board(board, coords, 0, stats);

}

//finds the reward of the action taken to get to the new state
//(needs the old state to see what is placed in the position of the coordinates)
double get_reward(char *board, int *shipsSunk, Shot shot)
{
	
	if (shot.hitMiss == 1)
	{
		//check if ship was sunk then check if game was won
		if (check_if_sunk_ship(board, shot.current) == 1)
		{
			*shipsSunk += 1;
			//returns 10 if game is won
			if (*shipsSunk == 5)
				return 200.0;
			
			//returns 5 for sinking a ship
			return 75.0;
		}

		//returns 2 if ship is hit
		return 25.0;
	}
	
	//returns -1 if shot is a miss or shot is in the same 
	//place as previously guessed
	return -1.0;
	
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

//appends an array to another
void append(double *dest, double *source, int start, int length)
{
	int i = 0;
	double *s = source, *d = dest + start;

	for (i = 0; i < length; i++, d++, s++)
		*d = *s;
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


//newQval = ann_run(temp, newState);
//maxQ = q_max(newQval, 100);
//
////gets target reward based on whether terminal state or not
//memcpy(y, qval, sizeof(double) * 100);
//if (reward != 10)
//update = reward + gamma * maxQ;
//else
//update = reward;
//
////trains the network to the target output
//y[action] = update;
////cost += calc_cost(qval, y, 100);
//ann_fit(temp, state, y);