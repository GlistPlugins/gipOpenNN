/*
 * gipOpenNN.cpp
 *
 *  Created on: 12 Aug 2021
 *      Author: oznur
 */

#include "gipOpenNN.h"


gipOpenNN::gipOpenNN() {
	dataset = nullptr;
	neuralnetwork = nullptr;
	trainingstrategy = nullptr;
	testinganalysis = nullptr;
}

gipOpenNN::~gipOpenNN() {
	if(testinganalysis) delete testinganalysis;
	if(trainingstrategy) delete trainingstrategy;
	if(neuralnetwork) delete neuralnetwork;
	if(dataset) delete dataset;
}

void gipOpenNN::loadDataset(std::string datasetFullPath, char delimiter, bool hasColumnNames) {
	dataset = new DataSet(datasetFullPath, delimiter, hasColumnNames);
}

void gipOpenNN::loadDatasetFile(std::string datasetFileName, char delimiter, bool hasColumnNames) {
	loadDataset(gGetFilesDir() + datasetFileName, delimiter, hasColumnNames);
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Tensor<Index, 1>& tensor) {
	neuralnetwork = new NeuralNetwork(projectType, tensor);
	neuralnetwork->set_inputs_names(dataset->get_input_variables_names());
        //scaleInputs();
        //unscaleTargets();
        createTrainingStrategy();
        createTestingAnalysis();
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, int hiddenNeuronNum) {
	Tensor<Index, 1> architecture(3);
	architecture.setValues({dataset->get_input_variables_number(), hiddenNeuronNum, dataset->get_target_variables_number()});
	createNeuralNetwork(projectType, architecture);
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, std::vector<int> hiddenNeuronNums) {
        int perceptronlayernum = hiddenNeuronNums.size();
        Tensor<Index, 1> architecture(perceptronlayernum + 2);
        architecture(0) = dataset->get_input_variables_number();
        architecture(architecture.size() - 1) = dataset->get_target_variables_number();
        for(int i = 0; i < perceptronlayernum; i++) architecture(i + 1) = hiddenNeuronNums[i];
        createNeuralNetwork(projectType, architecture);
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

const Tensor<float, 2> gipOpenNN::calculateOutputs(Tensor<float, 2>& inputs) {
	return neuralnetwork->calculate_outputs(inputs);
}

const Tensor<std::string, 1> gipOpenNN::getOutputNames() {
	return neuralnetwork->get_outputs_names();
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

void gipOpenNN::scaleInputs() {
	if(!neuralnetwork->has_scaling_layer()) return;
	if(dataset->is_empty()) return;
	const Index inputnum = dataset->get_input_variables_number();
	Tensor<std::string, 1> scalingmethod(inputnum);
	scalingmethod.setConstant("MeanStandardDeviation");
	Tensor<Descriptives, 1> inputdes = getDataset()->scale_input_variables(scalingmethod);
	ScalingLayer* sl = neuralnetwork->get_scaling_layer_pointer();
	sl->set_descriptives(inputdes);
	sl->set_scaling_methods("MeanStandardDeviation");
}

void gipOpenNN::unscaleTargets() {
	if(!neuralnetwork->has_unscaling_layer()) return;
	if(dataset->is_empty()) return;
	const Index targetnum = dataset->get_target_variables_number();
	Tensor<std::string, 1> unscalingmethod(targetnum);
	unscalingmethod.setConstant("MeanStandardDeviation");
	Tensor<Descriptives, 1> targetdes = getDataset()->scale_target_variables(unscalingmethod);
	UnscalingLayer* ul = neuralnetwork->get_unscaling_layer_pointer();
	ul->set_descriptives(targetdes);
	ul->set_unscaling_methods("MeanStandardDeviation");
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

