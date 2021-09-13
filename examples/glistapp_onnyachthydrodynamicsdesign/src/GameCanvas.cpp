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

	srand(static_cast<unsigned>(time(nullptr)));

	// Data set
	ai.setDataset("yachtresistance.csv", ';', true);
	const gipOpenNN::Tensor<string, 1> inputs_names = ai.getDataset()->get_input_variables_names();
	const gipOpenNN::Tensor<string, 1> targets_names = ai.getDataset()->get_target_variables_names();
	ai.getDataset()->split_samples_random();
	const gipOpenNN::Index input_variables_number = ai.getDataset()->get_input_variables_number();
	const gipOpenNN::Index target_variables_number = ai.getDataset()->get_target_variables_number();

	gipOpenNN::Tensor<string, 1> scaling_inputs_methods(input_variables_number);
	scaling_inputs_methods.setConstant("MinimumMaximum");
	gipOpenNN::Tensor<string, 1> scaling_target_methods(target_variables_number);
	scaling_target_methods.setConstant("MinimumMaximum");
	const gipOpenNN::Tensor<gipOpenNN::Descriptives, 1> inputs_descriptives = ai.getDataset()->scale_input_variables(scaling_inputs_methods);
	const gipOpenNN::Tensor<gipOpenNN::Descriptives, 1> targets_descriptives = ai.getDataset()->scale_target_variables(scaling_target_methods);


	// Neural network
	const gipOpenNN::Index hidden_neurons_number = 10;
	gipOpenNN::Tensor<gipOpenNN::Index, 1> neural_network_architecture(3);
	neural_network_architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});
	ai.createNeuralNetwork(gipOpenNN::NeuralNetwork::Approximation, neural_network_architecture);

	ai.getNeuralNetwork()->set_inputs_names(inputs_names);
	ai.getNeuralNetwork()->set_outputs_names(targets_names);

	gipOpenNN::ScalingLayer* scaling_layer_pointer = ai.getNeuralNetwork()->get_scaling_layer_pointer();
	scaling_layer_pointer->set_scaling_methods(gipOpenNN::ScalingLayer::MinimumMaximum);
	scaling_layer_pointer->set_descriptives(inputs_descriptives);

	gipOpenNN::UnscalingLayer* unscaling_layer_pointer = ai.getNeuralNetwork()->get_unscaling_layer_pointer();
	unscaling_layer_pointer->set_unscaling_methods(gipOpenNN::UnscalingLayer::MinimumMaximum);
	unscaling_layer_pointer->set_descriptives(targets_descriptives);


	// Training strategy object
	ai.createTrainingStrategy();
	ai.performTraining();


	// Testing analysis
	ai.createTestingAnalysis();
	const gipOpenNN::TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = ai.getTestingAnalysis()->perform_linear_regression_analysis()[0];
	gLogi("GameCanvas") << "correlation:" << linear_regression_analysis.correlation;
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

