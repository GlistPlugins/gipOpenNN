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
	ai.loadDatasetFile("breast_cancer.csv",';',true);
	ai.getDataset()->split_samples_random();

	const gipOpenNN::Tensor<string, 1> inputs_names = ai.getDataset()->get_input_variables_names();
	const gipOpenNN::Tensor<string, 1> targets_names = ai.getDataset()->get_target_variables_names();
	const Index input_variables_number = ai.getDataset()->get_input_variables_number();
	gipOpenNN::Tensor<string, 1> scaling_methods(input_variables_number);
	scaling_methods.setConstant("MeanStandardDeviation");
	const gipOpenNN::Tensor<gipOpenNN::Descriptives, 1> inputs_descriptives = ai.getDataset()->scale_input_variables(scaling_methods);


	// Neural network
	gipOpenNN::Tensor<gipOpenNN::Index, 1> neural_netowrk_architecture(3);
	neural_netowrk_architecture.setValues({9, 7, 1});
	ai.createNeuralNetwork(gipOpenNN::NeuralNetwork::Classification, neural_netowrk_architecture);

	dynamic_cast<gipOpenNN::PerceptronLayer*>(ai.getNeuralNetwork()->get_trainable_layers_pointers()(0))->set_activation_function(gipOpenNN::PerceptronLayer::HyperbolicTangent);
	dynamic_cast<gipOpenNN::ProbabilisticLayer*>(ai.getNeuralNetwork()->get_trainable_layers_pointers()(1))->set_activation_function(gipOpenNN::ProbabilisticLayer::Logistic);
	gipOpenNN::ScalingLayer* scaling_layer_pointer = ai.getNeuralNetwork()->get_scaling_layer_pointer();
	scaling_layer_pointer->set_scaling_methods(gipOpenNN::ScalingLayer::NoScaling);


	// Training strategy
	ai.createTrainingStrategy();
	ai.getTrainingStrategy()->set_optimization_method(gipOpenNN::TrainingStrategy::CONJUGATE_GRADIENT);
	ai.getTrainingStrategy()->set_loss_method(gipOpenNN::TrainingStrategy::NORMALIZED_SQUARED_ERROR);
	ai.getTrainingStrategy()->get_loss_index_pointer()->set_regularization_method(gipOpenNN::LossIndex::RegularizationMethod::L2);
	ai.getTrainingStrategy()->get_loss_index_pointer()->set_regularization_weight(0.001);

	gipOpenNN::ConjugateGradient* cg = ai.getTrainingStrategy()->get_conjugate_gradient_pointer();
	cg->set_loss_goal(1.0e-3);
	cg->set_display(true);
	ai.getTrainingStrategy()->set_display(true);

	ai.performTraining();

	scaling_layer_pointer->set_descriptives(inputs_descriptives);
	scaling_layer_pointer->set_scaling_methods(gipOpenNN::ScalingLayer::MeanStandardDeviation);

	// Testing analysis

	ai.getDataset()->unscale_input_variables(scaling_methods, inputs_descriptives);
	ai.createTestingAnalysis();

	gipOpenNN::Tensor<gipOpenNN::Index, 2> confusion = ai.getTestingAnalysis()->calculate_confusion();
	gLogi("GameCanvas") << "Confusion: " << confusion;

	gipOpenNN::Tensor<float, 1> binary_classification_tests = ai.getTestingAnalysis()->calculate_binary_classification_tests();
	gLogi("GameCanvas") << "Binary classification tests: ";
	gLogi("GameCanvas") << "Classification accuracy         : " << binary_classification_tests[0];
	gLogi("GameCanvas") << "Error rate                      : " << binary_classification_tests[1];
	gLogi("GameCanvas") << "Sensitivity                     : " << binary_classification_tests[2];
	gLogi("GameCanvas") << "Specificity                     : " << binary_classification_tests[3];
	gLogi("GameCanvas") << "Precision                       : " << binary_classification_tests[4];
	gLogi("GameCanvas") << "Positive likelihood             : " << binary_classification_tests[5];
	gLogi("GameCanvas") << "Negative likelihood             : " << binary_classification_tests[6];
	gLogi("GameCanvas") << "F1 score                        : " << binary_classification_tests[7];
	gLogi("GameCanvas") << "False positive rate             : " << binary_classification_tests[8];
	gLogi("GameCanvas") << "False discovery rate            : " << binary_classification_tests[9];
	gLogi("GameCanvas") << "False negative rate             : " << binary_classification_tests[10];
	gLogi("GameCanvas") << "Negative predictive value       : " << binary_classification_tests[11];
	gLogi("GameCanvas") << "Matthews correlation coefficient: " << binary_classification_tests[12];
	gLogi("GameCanvas") << "Informedness                    : " << binary_classification_tests[13];
	gLogi("GameCanvas") << "Markedness                      : " << binary_classification_tests[14];
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

