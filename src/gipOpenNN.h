/*
 * gipOpenNN.h
 *
 *  Created on: 12 A�u 2021
 *      Author: oznur
 */

#ifndef SRC_GIPOPENNN_H_
#define SRC_GIPOPENNN_H_

#include "gBasePlugin.h"
#include "opennn.h"
#include "optimization_algorithm.h"


class gipOpenNN : public gBasePlugin{
public:

	typedef OpenNN::DataSet Dataset;
	typedef OpenNN::NeuralNetwork NeuralNetwork;
	typedef OpenNN::TrainingStrategy TrainingStrategy;
	typedef OpenNN::OptimizationAlgorithm OptimizationAlgorithm;
	typedef OpenNN::TestingAnalysis TestingAnalysis;

	template<typename Scalar_, int NumIndices_>
	using Tensor = Eigen::Tensor<Scalar_, NumIndices_>;


	gipOpenNN();
	virtual ~gipOpenNN();

	void setDataset(std::string datasetFilepath, char delimiter, bool hasColumnNames);
	void createNeuralNetwork(const NeuralNetwork::ProjectType&, const Tensor<Index, 1>&);
	void performTraining();
	void calculateTests();
	void saveResults(std::string neuralNetworkFilename, std::string expressionFileName);

private:
	Dataset* dataset;
	NeuralNetwork* neural_network;
	TrainingStrategy* training_strategy;
	OptimizationAlgorithm::Results training_results;
	TestingAnalysis* testing_analysis;
	Tensor<float, 1> binary_classification_tests;
	Tensor<Index, 2> confusion;
};

#endif /* SRC_GIPOPENNN_H_ */
