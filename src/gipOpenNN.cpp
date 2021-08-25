/*
 * gipOpenNN.cpp
 *
 *  Created on: 12 Aðu 2021
 *      Author: oznur
 */

#include "gipOpenNN.h"

using namespace OpenNN;

gipOpenNN::gipOpenNN() {


}

gipOpenNN::~gipOpenNN() {

}

void gipOpenNN::setDataset(std::string datasetFilepath, char delimiter, bool hasColumnNames) {
	dataset = DataSet(gGetFilesDir() + datasetFilepath, delimiter, hasColumnNames);
}
