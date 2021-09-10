/*
 * gipOpenNN.cpp
 *
 *  Created on: 12 A�u 2021
 *      Author: oznur
 */

#include "gipOpenNN.h"


gipOpenNN::gipOpenNN() {
}

gipOpenNN::~gipOpenNN() {
	delete testing_analysis;
	delete training_strategy;
	delete neural_network;
	delete dataset;
}

void gipOpenNN::setDataset(std::string datasetFilepath, char delimiter, bool hasColumnNames) {
	dataset = new Dataset(gGetFilesDir() + datasetFilepath, delimiter, hasColumnNames);
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Tensor<Index, 1>& tensor) {
	neural_network = new NeuralNetwork(projectType, tensor);
}

void gipOpenNN::performTraining() {
	training_strategy = new TrainingStrategy(neural_network, dataset);
	training_results = training_strategy->perform_training();
}

void gipOpenNN::calculateTests() {
	testing_analysis = new TestingAnalysis(neural_network, dataset);
	binary_classification_tests = testing_analysis->calculate_binary_classification_tests();
	confusion = testing_analysis->calculate_confusion();
}

void gipOpenNN::saveResults(std::string neuralNetworkFilename, std::string expressionFileName) {
	neural_network->save(gGetFilesDir() + neuralNetworkFilename);
	neural_network->save_expression_python(gGetFilesDir() + expressionFileName);
}

