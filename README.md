# gipOpenNN
GlistEngine component to implement artificial intelligence and neural network functionalities using OpenNN.

## Installation
Fork & clone this project into ~/dev/glist/glistplugins

## Usage
1. Add gipOpenNN into plugins of your GlistApp/CMakeLists.txt
> set(PLUGINS gipOpenNN)

2. Include gipOpenNN.h in GameCanvas.h
> #include "gipOpenNN.h"

3. Define gipOpenNN object
> gipOpenNN ai;

4. Use AI funcitonality:
> ai.setDataset(...);
> ai.createNeuralNetwork(...);
> ai.createTrainingStrategy();
> ai.performTraining();
> ai.createTestingAnalysis();
> And get outputs.

Developers can run the sample GlistApp projects located under examples/ folder to see how to use the plugin functionality.

## Plugin Licence
Apache 2
