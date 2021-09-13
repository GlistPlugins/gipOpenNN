/*
 * GameCanvas.cpp
 *
 *  Created on: May 6, 2020
 *      Author: noyan
 */


#include "GameCanvas.h"


GameCanvas::GameCanvas(gApp* root) : gBaseCanvas(root) {
	this->root = root;
}

GameCanvas::~GameCanvas() {
}

void GameCanvas::setup() {
//	gLogi("GameCanvas") << "setup";
	logo.loadImage("glistengine_logo.png");

	// Data set
	ai.setDataset("vehicle_insurances.csv", ';', true);
	gipOpenNN::DataSet* dataset = ai.getDataset();
	const gipOpenNN::Tensor<string, 1> inputs_names = dataset->get_input_variables_names();
	const gipOpenNN::Tensor<string, 1> targets_names = dataset->get_target_variables_names();
	dataset->split_samples_random();

	const gipOpenNN::Index input_variables_number = dataset->get_input_variables_number();
	const gipOpenNN::Index target_variables_number = dataset->get_target_variables_number();
	gipOpenNN::Tensor<string, 1> scaling_inputs_methods(input_variables_number);
	scaling_inputs_methods.setConstant("MinimumMaximum");
	const gipOpenNN::Tensor<gipOpenNN::Descriptives, 1> inputs_descriptives = dataset->scale_input_variables(scaling_inputs_methods);

	// Neural network
	const gipOpenNN::Index hidden_neurons_number = 20;
	gipOpenNN::Tensor<gipOpenNN::Index, 1> architecture(3);
	architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});
	ai.createNeuralNetwork(gipOpenNN::NeuralNetwork::Classification, architecture);

	gipOpenNN::NeuralNetwork* neuralnetwork = ai.getNeuralNetwork();
	neuralnetwork->set_inputs_names(inputs_names);
	neuralnetwork->set_outputs_names(targets_names);

	gipOpenNN::ScalingLayer* scaling_layer_pointer = neuralnetwork->get_scaling_layer_pointer();
	scaling_layer_pointer->set_descriptives(inputs_descriptives);
	scaling_layer_pointer->set_scaling_methods(gipOpenNN::ScalingLayer::MinimumMaximum);

	// Training strategy
	ai.createTrainingStrategy();
	gipOpenNN::TrainingStrategy* trainingstrategy = ai.getTrainingStrategy();
	trainingstrategy->set_loss_method(gipOpenNN::TrainingStrategy::NORMALIZED_SQUARED_ERROR);
	trainingstrategy->set_optimization_method(gipOpenNN::TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);

	gipOpenNN::AdaptiveMomentEstimation* adam = trainingstrategy->get_adaptive_moment_estimation_pointer();
	adam->set_loss_goal(1.0e-3);
	adam->set_maximum_epochs_number(10000);
	adam->set_display_period(1000);

	ai.performTraining();

	// Testing analysis
	gipOpenNN::Tensor<float, 2> inputs(2, 8);
	inputs.setValues({{381110,0,21,1,1,0,35118,161}, {381111,1,38,0,2,1,52742,165}});
	Tensor<float, 2> outputs = ai.getNeuralNetwork()->calculate_outputs(inputs);

	dataset->unscale_input_variables(scaling_inputs_methods, inputs_descriptives);
	ai.createTestingAnalysis();
	ai.performConfusionTest();


	// Outputs
	gipOpenNN::Tensor<std::string, 1> outputnames = ai.getNeuralNetwork()->get_outputs_names();
	for(int i = 0; i < outputnames.size(); i++) {
		gLogi("GC") << "name:" << outputnames(i);
	}

	for(int i = 0; i < outputs.size(); i++) {
		gLogi("GC") << "o" << gToStr(i) << ":" << gToStr(outputs(i, 0));
	}


	// Save results
	ai.saveDataset("results/data_set.xml");
	ai.saveNeuralNetwork("results/neural_network.xml");
	ai.saveTrainingStrategy("results/training_strategy.xml");
}

void GameCanvas::update() {
//	gLogi("GameCanvas") << "update";
}

void GameCanvas::draw() {
//	gLogi("GameCanvas") << "draw";
	logo.draw((getWidth() - logo.getWidth()) / 2, (getHeight() - logo.getHeight()) / 2);
}

void GameCanvas::keyPressed(int key) {
//	gLogi("GameCanvas") << "keyPressed:" << key;
}

void GameCanvas::keyReleased(int key) {
//	gLogi("GameCanvas") << "keyReleased:" << key;
}

void GameCanvas::charPressed(unsigned int codepoint) {
//	gLogi("GameCanvas") << "charPressed:" << gCodepointToStr(codepoint);
}

void GameCanvas::mouseMoved(int x, int y) {
//	gLogi("GameCanvas") << "mouseMoved" << ", x:" << x << ", y:" << y;
}

void GameCanvas::mouseDragged(int x, int y, int button) {
//	gLogi("GameCanvas") << "mouseDragged" << ", x:" << x << ", y:" << y << ", b:" << button;
}

void GameCanvas::mousePressed(int x, int y, int button) {
}

void GameCanvas::mouseReleased(int x, int y, int button) {
//	gLogi("GameCanvas") << "mouseReleased" << ", button:" << button;
}

void GameCanvas::mouseScrolled(int x, int y) {
//	gLogi("GameCanvas") << "mouseScrolled" << ", x:" << x << ", y:" << y;
}

void GameCanvas::mouseEntered() {
}

void GameCanvas::mouseExited() {
}

void GameCanvas::showNotify() {

}

void GameCanvas::hideNotify() {

}

