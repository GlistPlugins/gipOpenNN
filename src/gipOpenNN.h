/*
 * gipOpenNN.h
 *
 *  Created on: 12 Aðu 2021
 *      Author: oznur
 */

#ifndef SRC_GIPOPENNN_H_
#define SRC_GIPOPENNN_H_

#include "gBasePlugin.h"
#include "opennn.h"

// using namespace OpenNN;

class gipOpenNN : public gBasePlugin{
public:
	gipOpenNN();
	virtual ~gipOpenNN();

	void setDataset(std::string datasetFilepath, char delimiter, bool hasColumnNames);

private:
	OpenNN::DataSet dataset;
};

#endif /* SRC_GIPOPENNN_H_ */
