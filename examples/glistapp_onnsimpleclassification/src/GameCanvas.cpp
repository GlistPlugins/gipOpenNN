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
	ai.setDataset("simple_pattern_recognition.csv", ';', true);
	gipOpenNN::DataSet* dataset = ai.getDataset();
    dataset->split_samples_random();

    const Tensor<string, 1> inputs_names = dataset->get_input_variables_names();
    const Tensor<string, 1> targets_names = dataset->get_target_variables_names();
    Tensor<string, 1> scaling_inputs_methods(inputs_names.dimension(0));
    scaling_inputs_methods.setConstant("MinimumMaximum");
    const Tensor<gipOpenNN::Descriptives, 1> inputs_descriptives = dataset->scale_input_variables(scaling_inputs_methods);

	// Neural network
	Tensor<gipOpenNN::Index, 1> architecture(3);
	architecture.setValues({2, 20, 1});
	ai.createNeuralNetwork(gipOpenNN::NeuralNetwork::ProjectType::Classification, architecture);
	gipOpenNN::NeuralNetwork* neural_network = ai.getNeuralNetwork();

	neural_network->set_inputs_names(inputs_names);
	neural_network->set_outputs_names(targets_names);
	gipOpenNN::ScalingLayer* scaling_layer_pointer = neural_network->get_scaling_layer_pointer();
	scaling_layer_pointer->set_descriptives(inputs_descriptives);

	// Training strategy
	ai.createTrainingStrategy();
	gipOpenNN::TrainingStrategy* training_strategy = ai.getTrainingStrategy();
	training_strategy->set_loss_method(gipOpenNN::TrainingStrategy::NORMALIZED_SQUARED_ERROR); //Auto
	training_strategy->get_loss_index_pointer()->set_regularization_method(gipOpenNN::LossIndex::NoRegularization); //Auto
	training_strategy->set_optimization_method(gipOpenNN::TrainingStrategy::QUASI_NEWTON_METHOD); //Auto

	ai.performTraining();

	// Testing Analysis
	dataset->unscale_input_variables_minimum_maximum(inputs_descriptives);
	ai.createTestingAnalysis();
	ai.performBinaryClassificationTest();
	ai.performConfusionTest();

	// Outputs
	gipOpenNN::Tensor<std::string, 1> outputnames = ai.getNeuralNetwork()->get_outputs_names();
	for(int i = 0; i < outputnames.size(); i++) {
		gLogi("GC") << "name:" << outputnames(i);
	}
	gipOpenNN::Tensor<float, 2> inputs(2, 2);
	inputs.setValues({{0.7430556,0.37012988}, {0.5241379,0.80263156}});
	Tensor<float, 2> outputs = ai.getNeuralNetwork()->calculate_outputs(inputs);
	for(int i = 0; i < outputs.size(); i++) {
		gLogi("GC") << "o" << gToStr(i) << ":" << gToStr(outputs(i, 0));
	}

	// Save model
	ai.saveNeuralNetwork("results/neural_network.xml");
	ai.saveExpression("results/expression.cpp");

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

