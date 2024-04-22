/*
 * gipOpenNN.h
 *
 *  Created on: 12 Aug 2021
 *      Author: oznur
 *      Hardly edited by: Noyan
 */

#ifndef SRC_GIPOPENNN_H_
#define SRC_GIPOPENNN_H_

#include "gBaseComponent.h"
#include "opennn.h"
#include "optimization_algorithm.h"


class gipOpenNN : public gBaseComponent {
public:

	typedef Eigen::Index Index;
	typedef OpenNN::DataSet DataSet;
	typedef OpenNN::NeuralNetwork NeuralNetwork;
	typedef OpenNN::TrainingStrategy TrainingStrategy;
	typedef OpenNN::OptimizationAlgorithm OptimizationAlgorithm;
	typedef OpenNN::TestingAnalysis TestingAnalysis;
	typedef OpenNN::Descriptives Descriptives;
	typedef OpenNN::ScalingLayer ScalingLayer;
	typedef OpenNN::UnscalingLayer UnscalingLayer;
	typedef OpenNN::PerceptronLayer PerceptronLayer;
	typedef OpenNN::ProbabilisticLayer ProbabilisticLayer;
	typedef OpenNN::AdaptiveMomentEstimation AdaptiveMomentEstimation;
	typedef OpenNN::ModelSelection ModelSelection;
	typedef OpenNN::GrowingNeurons GrowingNeurons;
	typedef OpenNN::GeneticAlgorithm GeneticAlgorithm;
	typedef OpenNN::LossIndex LossIndex;
	typedef OpenNN::ConjugateGradient ConjugateGradient;
	typedef OpenNN::QuasiNewtonMethod QuasiNewtonMethod;
    typedef OpenNN::LevenbergMarquardtAlgorithm LevenbergMarquardtAlgorithm;

	template<typename Scalar_, int NumIndices_>
	using Tensor = Eigen::Tensor<Scalar_, NumIndices_>;


	gipOpenNN();
	virtual ~gipOpenNN();

	void loadDataset(std::string datasetFullPath, char delimiter, bool hasColumnNames);
	void loadDatasetFile(std::string datasetFileName, char delimiter, bool hasColumnNames);
	void setDataset(DataSet& ds);
	void createNeuralNetwork(const NeuralNetwork::ProjectType&, const Tensor<Index, 1>&);
	void createNeuralNetwork(const NeuralNetwork::ProjectType&, int hiddenNeuronNum);
	void createNeuralNetwork(const NeuralNetwork::ProjectType&, std::vector<int> hiddenNeuronNums);

	void createTrainingStrategy();
	void performTraining();

	void createTestingAnalysis();
	void performBinaryClassificationTest();
	void performConfusionTest();

	const Tensor<float, 2> calculateOutputs(Tensor<float, 2>& inputs);
	const Tensor<std::string, 1> getOutputNames();

	void saveOutputs(const Tensor<float, 2>& inputs, std::string csvFilename);
	void saveDataset(std::string xmlFilename);
	void saveNeuralNetwork(std::string xmlFilename);
	void saveNeuralNetworkFile(std::string xmlFilename);
	void loadNeuralNetwork(std::string xmlFilename);
	void loadNeuralNetworkFile(std::string xmlFilename);
	void saveTrainingStrategy(std::string xmlFilename);
	void saveTestingAnalysis(std::string xmlFilename);
	void saveExpression(std::string cppFilename);

	void scaleInputs();
	void unscaleTargets();

	gipOpenNN::DataSet* getDataset();
	gipOpenNN::NeuralNetwork* getNeuralNetwork();
	gipOpenNN::TrainingStrategy* getTrainingStrategy();
	gipOpenNN::OptimizationAlgorithm::Results* getTrainingResults();
	gipOpenNN::TestingAnalysis* getTestingAnalysis();

private:
	DataSet* dataset;
	NeuralNetwork* neuralnetwork;
	TrainingStrategy* trainingstrategy;
	OptimizationAlgorithm::Results trainingresults;
	TestingAnalysis* testinganalysis;
	Tensor<float, 1> binaryclassificationtests;
	Tensor<Index, 2> confusion;
};

#endif /* SRC_GIPOPENNN_H_ */
