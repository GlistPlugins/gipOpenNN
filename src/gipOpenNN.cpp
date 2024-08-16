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
	if(dataset != nullptr) delete dataset;
	dataset = new DataSet(datasetFullPath, delimiter, hasColumnNames);
}

void gipOpenNN::loadDatasetFile(std::string datasetFileName, char delimiter, bool hasColumnNames) {
	loadDataset(gGetFilesDir() + datasetFileName, delimiter, hasColumnNames);
}

void gipOpenNN::setDataset(DataSet& ds) {
	if(dataset != nullptr) delete dataset;
	dataset = new DataSet();
	dataset->set(ds);
	dataset->set_columns_uses(ds.get_columns_uses());
	dataset->set_samples_number(ds.get_samples_number());
	dataset->set_samples_uses(ds.get_samples_uses());
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, const Tensor<Index, 1>& tensor) {
	if(neuralnetwork != nullptr) delete neuralnetwork;
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
	size_t perceptronlayernum = hiddenNeuronNums.size();
	Tensor<Index, 1> architecture(perceptronlayernum + 2);
	architecture(0) = dataset->get_input_variables_number();
	architecture(architecture.size() - 1) = dataset->get_target_variables_number();
	for(int i = 0; i < perceptronlayernum; i++) {
		architecture(i + 1) = hiddenNeuronNums[i];
	}
	createNeuralNetwork(projectType, architecture);
}

void gipOpenNN::createNeuralNetwork(const NeuralNetwork::ProjectType& projectType, std::vector<uint32_t> hiddenNeuronNums) {
	size_t perceptronlayernum = hiddenNeuronNums.size();
	Tensor<Index, 1> architecture(perceptronlayernum + 2);
	architecture(0) = dataset->get_input_variables_number();
	architecture(architecture.size() - 1) = dataset->get_target_variables_number();
	for(int i = 0; i < perceptronlayernum; i++) {
		architecture(i + 1) = hiddenNeuronNums[i];
	}
	createNeuralNetwork(projectType, architecture);
}

void gipOpenNN::createTrainingStrategy() {
	if(trainingstrategy != nullptr) delete trainingstrategy;
	trainingstrategy = new TrainingStrategy(neuralnetwork, dataset);
}

void gipOpenNN::performTraining() {
	trainingresults = trainingstrategy->perform_training();
}

void gipOpenNN::createTestingAnalysis() {
	if(testinganalysis != nullptr) delete testinganalysis;
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
	neuralnetwork->save(xmlFilename);
}

void gipOpenNN::saveNeuralNetworkFile(std::string xmlFilename) {
	saveNeuralNetwork(gGetFilesDir() + xmlFilename);
}

void gipOpenNN::loadNeuralNetwork(std::string xmlFilename) {
	if(neuralnetwork != nullptr) delete neuralnetwork;
	neuralnetwork = new NeuralNetwork();
	neuralnetwork->load(xmlFilename);

	TrainingStrategy::OptimizationMethod om = trainingstrategy->get_optimization_method();
	TrainingStrategy::LossMethod lm = trainingstrategy->get_loss_method();
	LossIndex::RegularizationMethod rm = trainingstrategy->get_loss_index_pointer()->get_regularization_method();
	int menum = 10000;
	int dp = 5;
	int mt = 3600;
	if(om == TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD) {
		menum = trainingstrategy->get_quasi_Newton_method_pointer()->get_maximum_epochs_number();
		mt = trainingstrategy->get_quasi_Newton_method_pointer()->get_maximum_time();
		dp = trainingstrategy->get_quasi_Newton_method_pointer()->get_display_period();
	} else if(om == TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT) {
		menum = trainingstrategy->get_gradient_descent_pointer()->get_maximum_epochs_number();
		mt = trainingstrategy->get_gradient_descent_pointer()->get_maximum_time();
		dp = trainingstrategy->get_gradient_descent_pointer()->get_display_period();
	} else if(om == TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT) {
		menum = trainingstrategy->get_conjugate_gradient_pointer()->get_maximum_epochs_number();
		mt = trainingstrategy->get_conjugate_gradient_pointer()->get_maximum_time();
		dp = trainingstrategy->get_conjugate_gradient_pointer()->get_display_period();
	} else if(om == TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM) {
		menum = trainingstrategy->get_Levenberg_Marquardt_algorithm_pointer()->get_maximum_epochs_number();
		mt = trainingstrategy->get_Levenberg_Marquardt_algorithm_pointer()->get_maximum_time();
		dp = trainingstrategy->get_Levenberg_Marquardt_algorithm_pointer()->get_display_period();
	} else if(om == TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT) {
//		menum = trainingstrategy->get_stochastic_gradient_descent_pointer()->get_maximum_epochs_number();
		mt = trainingstrategy->get_stochastic_gradient_descent_pointer()->get_maximum_time();
		dp = trainingstrategy->get_stochastic_gradient_descent_pointer()->get_display_period();
	} else if(om == TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION) {
//		menum = trainingstrategy->get_adaptive_moment_estimation_pointer()->get_maximum_epochs_number();
		mt = trainingstrategy->get_adaptive_moment_estimation_pointer()->get_maximum_time();
		dp = trainingstrategy->get_adaptive_moment_estimation_pointer()->get_display_period();
	}

	createTrainingStrategy();
	if(om == TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD) {
		trainingstrategy->get_quasi_Newton_method_pointer()->set_maximum_epochs_number(menum);
		trainingstrategy->get_quasi_Newton_method_pointer()->set_maximum_time(mt);
		trainingstrategy->get_quasi_Newton_method_pointer()->set_display_period(dp);
	} else if(om == TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT) {
		trainingstrategy->get_gradient_descent_pointer()->set_maximum_epochs_number(menum);
		trainingstrategy->get_gradient_descent_pointer()->set_maximum_time(mt);
		trainingstrategy->get_gradient_descent_pointer()->set_display_period(dp);
	} else if(om == TrainingStrategy::OptimizationMethod::CONJUGATE_GRADIENT) {
		trainingstrategy->get_conjugate_gradient_pointer()->set_maximum_epochs_number(menum);
		trainingstrategy->get_conjugate_gradient_pointer()->set_maximum_time(mt);
		trainingstrategy->get_conjugate_gradient_pointer()->set_display_period(dp);
	} else if(om == TrainingStrategy::OptimizationMethod::LEVENBERG_MARQUARDT_ALGORITHM) {
		trainingstrategy->get_Levenberg_Marquardt_algorithm_pointer()->set_maximum_epochs_number(menum);
		trainingstrategy->get_Levenberg_Marquardt_algorithm_pointer()->set_maximum_time(mt);
		trainingstrategy->get_Levenberg_Marquardt_algorithm_pointer()->set_display_period(dp);
	} else if(om == TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT) {
		trainingstrategy->get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(menum);
		trainingstrategy->get_stochastic_gradient_descent_pointer()->set_maximum_time(mt);
		trainingstrategy->get_stochastic_gradient_descent_pointer()->set_display_period(dp);
	} else if(om == TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION) {
		trainingstrategy->get_adaptive_moment_estimation_pointer()->set_maximum_epochs_number(menum);
		trainingstrategy->get_adaptive_moment_estimation_pointer()->set_maximum_time(mt);
		trainingstrategy->get_adaptive_moment_estimation_pointer()->set_display_period(dp);
	}


	createTestingAnalysis();
}

void gipOpenNN::loadNeuralNetworkFile(std::string xmlFilename) {
	loadNeuralNetwork(gGetFilesDir() + xmlFilename);
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

