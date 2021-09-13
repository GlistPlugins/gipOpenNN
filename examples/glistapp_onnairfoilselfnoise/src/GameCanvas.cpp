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
	ai.setDataset("airfoil_self_noise.csv", ';', true);

    const gipOpenNN::Tensor<string, 1> inputs_names = ai.getDataset()->get_input_variables_names();
    const gipOpenNN::Tensor<string, 1> targets_names = ai.getDataset()->get_target_variables_names();

    ai.getDataset()->split_samples_random();

	const gipOpenNN::Index input_variables_number = ai.getDataset()->get_input_variables_number();
	const gipOpenNN::Index target_variables_number = ai.getDataset()->get_target_variables_number();

	gipOpenNN::Tensor<string, 1> scaling_inputs_methods(input_variables_number);
    scaling_inputs_methods.setConstant("MinimumMaximum");

    gipOpenNN::Tensor<string, 1> scaling_target_methods(target_variables_number);
    scaling_target_methods.setConstant("MinimumMaximum");

    const gipOpenNN::Tensor<gipOpenNN::Descriptives, 1> inputs_descriptives =  ai.getDataset()->scale_input_variables(scaling_inputs_methods);
    const gipOpenNN::Tensor<gipOpenNN::Descriptives, 1> target_descriptives = ai.getDataset()->scale_target_variables(scaling_target_methods);

	// Neural network
	const gipOpenNN::Index hidden_neurons_number = 12;
	gipOpenNN::Tensor<gipOpenNN::Index, 1> architecture(3);
	architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});
	ai.createNeuralNetwork(gipOpenNN::NeuralNetwork::Approximation, architecture);
    ai.getNeuralNetwork()->set_inputs_names(inputs_names);
    ai.getNeuralNetwork()->set_outputs_names(targets_names);
    ai.getNeuralNetwork()->set_parameters_random();

    gipOpenNN::ScalingLayer* scaling_layer_pointer = ai.getNeuralNetwork()->get_scaling_layer_pointer();
    scaling_layer_pointer->set_descriptives(inputs_descriptives);
    scaling_layer_pointer->set_scaling_methods(scaling_inputs_methods);

    gipOpenNN::UnscalingLayer* unscaling_layer_pointer = ai.getNeuralNetwork()->get_unscaling_layer_pointer();
    unscaling_layer_pointer->set_descriptives(target_descriptives);
    unscaling_layer_pointer->set_unscaling_methods(scaling_target_methods);

	// Training strategy
	ai.createTrainingStrategy();
    ai.getTrainingStrategy()->set_optimization_method(gipOpenNN::TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);
    ai.getTrainingStrategy()->get_loss_index_pointer()->set_regularization_method(gipOpenNN::LossIndex::L2);
    gipOpenNN::AdaptiveMomentEstimation* adam = ai.getTrainingStrategy()->get_adaptive_moment_estimation_pointer();
    adam->set_maximum_epochs_number(10000);
    adam->set_display_period(1000);
    const gipOpenNN::OptimizationAlgorithm::Results optimization_algorithm_results = ai.getTrainingStrategy()->perform_training();

    // Testing analysis
    ai.getDataset()->unscale_input_variables(scaling_inputs_methods, inputs_descriptives);
    ai.getDataset()->unscale_target_variables(scaling_target_methods, target_descriptives);
    gipOpenNN::TestingAnalysis testing_analysis(ai.getNeuralNetwork(), ai.getDataset());
    const gipOpenNN::TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];

	// Outputs
	gipOpenNN::Tensor<float, 2> inputs(2, 5);
	inputs.setValues({{3150,15.6,0.1016,39.6,0.0528487}, {4000,15.6,0.1016,39.6,0.0528487}});
	Tensor<float, 2> outputs = ai.getNeuralNetwork()->calculate_outputs(inputs);

	gipOpenNN::Tensor<std::string, 1> outputnames = ai.getNeuralNetwork()->get_outputs_names();
	for(int i = 0; i < outputnames.size(); i++) {
		gLogi("GC") << "name:" << outputnames(i);
	}

	for(int i = 0; i < outputs.size(); i++) {
		gLogi("GC") << "o" << gToStr(i) << ":" << gToStr(outputs(i, 0));
	}


	// Save results
//    neural_network.save("../data/neural_network.xml");
//    neural_network.save_expression_python("../data/neural_network.py");
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

