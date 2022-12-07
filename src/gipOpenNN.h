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

	typedef Eigen::Index Index;
	typedef opennn::DataSet DataSet;
	typedef opennn::NeuralNetwork NeuralNetwork;
	typedef opennn::NeuralNetwork::ProjectType ProjectType;
	typedef opennn::TrainingStrategy TrainingStrategy;
	typedef opennn::TrainingStrategy::LossMethod LossMethod;
	typedef opennn::TrainingStrategy::OptimizationMethod OptimizationMethod;
	typedef opennn::TrainingResults TrainingResults;
	typedef opennn::OptimizationAlgorithm OptimizationAlgorithm;
	typedef opennn::TestingAnalysis TestingAnalysis;
	typedef opennn::Descriptives Descriptives;
	typedef opennn::ScalingLayer ScalingLayer;
	typedef opennn::UnscalingLayer UnscalingLayer;
	typedef opennn::PerceptronLayer PerceptronLayer;
	typedef opennn::ProbabilisticLayer ProbabilisticLayer;
	typedef opennn::AdaptiveMomentEstimation AdaptiveMomentEstimation;
	typedef opennn::ModelSelection ModelSelection;
	typedef opennn::GrowingNeurons GrowingNeurons;
	typedef opennn::GeneticAlgorithm GeneticAlgorithm;
	typedef opennn::LossIndex LossIndex;
	typedef opennn::LossIndex::RegularizationMethod RegularizationMethod;
	typedef opennn::ConjugateGradient ConjugateGradient;
	typedef opennn::QuasiNewtonMethod QuasiNewtonMethod;

	template<typename Scalar_, int NumIndices_>
	using Tensor = Eigen::Tensor<Scalar_, NumIndices_>;


	gipOpenNN();
	virtual ~gipOpenNN();

	void loadDataset(std::string datasetPath, char delimiter, bool hasColumnNames);
	void loadDatasetFile(std::string datasetFilePath, char delimiter, bool hasColumnNames);
	void createNeuralNetwork(const ProjectType&, int hiddenNeuronsNum);

	void performTraining();

	void createTestingAnalysis();
	void performBinaryClassificationTest();
	void performConfusionTest();

	const Tensor<type, 2> calculateOutputs(Tensor<type, 2>& inputs);

	void saveOutputs(Tensor<float, 2>& inputs, std::string csvFilename);
	void saveDataset(std::string xmlFilename);
	void saveNeuralNetwork(std::string xmlFilename);
	void saveTrainingStrategy(std::string xmlFilename);
	void saveTestingAnalysis(std::string xmlFilename);
	void saveExpression(std::string cppFilename);

	gipOpenNN::DataSet* getDataset();
	gipOpenNN::NeuralNetwork* getNeuralNetwork();
	gipOpenNN::TrainingStrategy* getTrainingStrategy();
	void setLossMethod(LossMethod lossMethod);
	void setOptimizationMethod(OptimizationMethod optimizationMethod);
	void setRegularizationMethod(RegularizationMethod regularizationMethod);
	gipOpenNN::TrainingResults* getTrainingResults();
	gipOpenNN::TestingAnalysis* getTestingAnalysis();

private:
	DataSet* dataset;
	NeuralNetwork* neuralnetwork;
	TrainingStrategy* trainingstrategy;
	TrainingResults trainingresults;
	TestingAnalysis* testinganalysis;
	Tensor<float, 1> binaryclassificationtests;
	Tensor<Index, 2> confusion;
};

#endif /* SRC_GIPOPENNN_H_ */
