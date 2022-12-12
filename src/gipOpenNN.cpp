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
	delete testinganalysis;
	delete trainingstrategy;
	delete neuralnetwork;
	delete dataset;
}

void gipOpenNN::loadDataset(std::string datasetPath, char delimiter, bool hasColumnNames) {
	dataset = new DataSet(datasetPath, delimiter, hasColumnNames);
}

void gipOpenNN::loadDatasetFile(std::string datasetFilePath, char delimiter, bool hasColumnNames) {
	loadDataset(gGetFilesDir() + datasetFilePath, delimiter, hasColumnNames);
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Tensor<Index, 1>& tensor) {
	neuralnetwork = new NeuralNetwork(projectType, tensor);
}

void gipOpenNN::createTrainingStrategy() {
	trainingstrategy = new TrainingStrategy(neuralnetwork, dataset);
}

void gipOpenNN::performTraining() {
	trainingresults = trainingstrategy->perform_training();
}

void gipOpenNN::createTestingAnalysis() {
	testinganalysis = new TestingAnalysis(neuralnetwork, dataset);
}

void gipOpenNN::performBinaryClassificationTest() {
	binaryclassificationtests = testinganalysis->calculate_binary_classification_tests();
}

void gipOpenNN::performConfusionTest() {
	confusion = testinganalysis->calculate_confusion();
}

void gipOpenNN::saveOutputs(const gipOpenNN::Tensor<float, 2>& inputs, std::string csvFilename) {
	neuralnetwork->save_outputs(inputs, gGetFilesDir() + csvFilename);
}

void gipOpenNN::saveDataset(std::string xmlFilename) {
	dataset->save(gGetFilesDir() + xmlFilename);
}

void gipOpenNN::saveNeuralNetwork(std::string xmlFilename) {
	neuralnetwork->save(gGetFilesDir() + xmlFilename);
}

void gipOpenNN::saveTrainingStrategy(std::string xmlFilename) {
	trainingstrategy->save(gGetFilesDir() + xmlFilename);
}

void gipOpenNN::saveTestingAnalysis(std::string xmlFilename) {
	testinganalysis->save(gGetFilesDir() + xmlFilename);
}

void gipOpenNN::saveExpression(std::string cppFilename) {
	neuralnetwork->save_expression_c(gGetFilesDir() + cppFilename);
}


gipOpenNN::DataSet* gipOpenNN::getDataset() {
	return dataset;
}

gipOpenNN::NeuralNetwork* gipOpenNN::getNeuralNetwork() {
	return neuralnetwork;
}

gipOpenNN::TrainingStrategy* gipOpenNN::getTrainingStrategy() {
	return trainingstrategy;
}

gipOpenNN::OptimizationAlgorithm::Results* gipOpenNN::getTrainingResults() {
	return &trainingresults;
}

gipOpenNN::TestingAnalysis* gipOpenNN::getTestingAnalysis() {
	return testinganalysis;
}

