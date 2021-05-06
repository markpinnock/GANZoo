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
using tf::Output;
using tf::Tensor;


//------------------------------------------------------------------------

class Dataloader
{
private:
	std::string m_file_path;
	std::vector<std::string> m_img_names;

	int m_mb_size;
	int m_mb_idx{ 0 };
	int m_num_mb{ 0 };
	int m_num_images{ 0 };
	int m_in_ch{ 3 };
	int m_resolution{ 64 };

	Output m_file_name_ph; // Placeholder for name of image to be loaded
	Output m_normalised_img;

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

	// Methods for manipulating the image loading graphs
	tf::Status createImageGraph(const int, const int);
	tf::Status getMinibatch(std::vector<Tensor>&);
};

#endif // !DATALOADER_H

