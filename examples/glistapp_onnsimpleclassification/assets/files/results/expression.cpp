/*
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this file.
You can manage it with the 'neural network' method.	
Example:

	vector<float> sample(n);	
	sample[0] = 1;	
	sample[1] = 2;	
	sample[n] = 10;	
	vector<float> outputs = neural_network(sample);

Notice that only one sample is allowed as input. Batch of inputs are not yet implement,	
however you can loop through neural network function in order to get multiple outputs.	
*/

#include <vector>

using namespace std;

vector<float> scaling_layer(const vector<float>& inputs)
{
	vector<float> outputs(2);

	outputs[0] = inputs[0]*2.933202028-1.546182394;
	outputs[1] = inputs[1]*2.756152153-1.411633015;

	return outputs;
}

vector<float> perceptron_layer_0(const vector<float>& inputs)
{
	vector<float> combinations(20);

	combinations[0] = 3.6891 -11.9619*inputs[0] +12.2985*inputs[1];
	combinations[1] = -0.125923 -1.87483*inputs[0] +3.31824*inputs[1];
	combinations[2] = 0.871078 -5.4216*inputs[0] +8.72181*inputs[1];
	combinations[3] = 4.3955 -14.3588*inputs[0] +6.92179*inputs[1];
	combinations[4] = -0.959286 -18.2031*inputs[0] +17.9216*inputs[1];
	combinations[5] = -3.99974 -5.47989*inputs[0] +10.1361*inputs[1];
	combinations[6] = -6.76886 -22.2448*inputs[0] +24.7599*inputs[1];
	combinations[7] = -2.83186 -2.48821*inputs[0] +4.45903*inputs[1];
	combinations[8] = 1.03567 -2.77449*inputs[0] -2.09067*inputs[1];
	combinations[9] = -2.57231 -5.35092*inputs[0] +5.6447*inputs[1];
	combinations[10] = 3.45756 +7.29472*inputs[0] -13.1729*inputs[1];
	combinations[11] = 1.8292 -2.91812*inputs[0] +3.66183*inputs[1];
	combinations[12] = -0.617242 -9.5316*inputs[0] +9.10661*inputs[1];
	combinations[13] = 2.18308 +1.56153*inputs[0] -4.81144*inputs[1];
	combinations[14] = -1.47203 -3.6242*inputs[0] +11.6719*inputs[1];
	combinations[15] = 0.288314 +5.32357*inputs[0] -18.0345*inputs[1];
	combinations[16] = 0.246858 -3.63503*inputs[0] +5.37893*inputs[1];
	combinations[17] = 0.268545 +1.06609*inputs[0] -1.79392*inputs[1];
	combinations[18] = -2.89747 +5.57543*inputs[0] -8.27511*inputs[1];
	combinations[19] = -0.752605 -7.26392*inputs[0] +11.2998*inputs[1];

	vector<float> activations(20);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);
	activations[3] = tanh(combinations[3]);
	activations[4] = tanh(combinations[4]);
	activations[5] = tanh(combinations[5]);
	activations[6] = tanh(combinations[6]);
	activations[7] = tanh(combinations[7]);
	activations[8] = tanh(combinations[8]);
	activations[9] = tanh(combinations[9]);
	activations[10] = tanh(combinations[10]);
	activations[11] = tanh(combinations[11]);
	activations[12] = tanh(combinations[12]);
	activations[13] = tanh(combinations[13]);
	activations[14] = tanh(combinations[14]);
	activations[15] = tanh(combinations[15]);
	activations[16] = tanh(combinations[16]);
	activations[17] = tanh(combinations[17]);
	activations[18] = tanh(combinations[18]);
	activations[19] = tanh(combinations[19]);

	return activations;
}

vector<float> probabilistic_layer(const vector<float>& inputs)
{
	vector<float> combinations(1);

	combinations[0] = -2.11436 +4.53255*inputs[0] -0.190176*inputs[1] +0.889665*inputs[2] +7.65002*inputs[3] +19.819*inputs[4] -1.51278*inputs[5] +32.2882*inputs[6] -5.39936*inputs[7] +20.4438*inputs[8] +12.333*inputs[9] -4.73238*inputs[10] -1.39688*inputs[11] +8.54262*inputs[12] +6.59443*inputs[13] +1.14179*inputs[14] -4.91892*inputs[15] +8.13032*inputs[16] -6.69023*inputs[17] -7.40605*inputs[18] +5.22783*inputs[19];

	vector<float> activations(1);

	activations[0] = 1.0/(1.0 + exp(-combinations[0]));

	return activations;
}

vector<float> neural_network(const vector<float>& inputs)
{
	vector<float> outputs;

	outputs = scaling_layer(inputs);
	outputs = perceptron_layer_0(outputs);
	outputs = probabilistic_layer(outputs);

	return outputs;
}
int main(){return 0;}
