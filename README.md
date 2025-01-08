# gipOpenNN
GlistEngine component to implement artificial intelligence and neural network functionalities using OpenNN.

## Install neccesary packages for Linux
For Debian-Based Distributions;
> sudo apt-get install libomp-15-dev

For RPM-Based Distributions;
> sudo dnf install libomp-devel-18.1.6-2.fc40.x86_64

## Installation
1.Make sure that your repo is up-to-date and then go to the plugins folder
> cd ~/dev/glist/glistplugins

2.Clone repository here
> git clone https://github.com/[YourGithubUsername]/gipOpenNN

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
