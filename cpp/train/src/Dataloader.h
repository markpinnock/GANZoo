#ifndef DATALOADER_H
#define DATALOADER_H

#include "../src/extra.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/summary/summary_file_writer.h"


namespace tf = tensorflow;
namespace ops = tf::ops;


//------------------------------------------------------------------------

class Dataloader
{
private:
	std::string m_file_path;
	std::vector<std::string> m_img_names;

	const int m_mb_size;
	int m_mb_idx{ 0 };
	int m_num_mb{ 0 };
	int m_num_images{ 0 };
	const int m_in_ch{ 3 };
	const int m_resolution{ 64 };

	tf::Output m_file_name_ph;	 // Placeholder for name of image to be loaded (i.e. input to graph)
	tf::Output m_normalised_img; // Output node from image load graph

	tf::Scope m_image_root; // Image loader graph
	//tf::ClientSession m_image_sess;

public:
	Dataloader(int mb_size) :
		m_mb_size{ mb_size },
		m_image_root{ tf::Scope::NewRootScope() } {}

	~Dataloader() {}

	/* */
	tf::Status loadFilenames(std::string);
	std::vector<std::string> getFilenames() const { return m_img_names; }
	int getNumMinibatches() const { return m_num_mb; }

	// Methods for manipulating the image loading graphs
	tf::Status createImageGraph(const int, const int);
	tf::Status getMinibatch(std::vector<tf::Tensor>&);
};

#endif // !DATALOADER_H

