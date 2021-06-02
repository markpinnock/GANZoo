#ifndef DCGAN_H
#define DCGAN_H

#include "../src/extra.h"

#include <map>
#include <string>
#include <vector>

#include "utils.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


namespace tf = tensorflow;
namespace ops = tf::ops;


//https://github.com/bennyfri/TFMacCpp/blob/master/CatDogCNNV2.cpp
//------------------------------------------------------------------------

class DCGAN
{
private:
	const int m_resolution;
	const int m_im_ch{ 3 };

	// Placeholders for image input, latent input and labels
	tf::Output m_image_ph;
	tf::Output m_latent_ph;
	tf::Output m_labels;

	// Fixed latent noise
	const int m_num_ex{ 16 };
	const int m_latent_dims;
	std::vector<tf::Tensor> m_fixed_latent;

	// Output tensors
	tf::Output m_discriminator_real;
	tf::Output m_discriminator_fake;
	tf::Output m_generator_output;

	tf::Output m_discriminator_loss;
	tf::Output m_generator_loss;

	// Sessions for discriminator and generator scopes
	std::unique_ptr<tf::ClientSession> m_discriminator_session;
	std::unique_ptr<tf::ClientSession> m_generator_session;

	tf::Scope m_discriminator_root;
	tf::Scope m_generator_root;

	// Discriminator and generator weights
	std::map<std::string, tf::Output> m_discriminator_weights;
	std::map<std::string, tf::TensorShape> m_discriminator_weight_shapes;
	std::map<std::string, tf::Output> m_discriminator_assigns;
	std::map<std::string, tf::Output> m_generator_weights;
	std::map<std::string, tf::TensorShape> m_generator_weight_shapes;
	std::map<std::string, tf::Output> m_generator_assigns;

	// Optimiser momentum variables and updates
	std::map<std::string, tf::Output> m_discriminator_momentum_init;
	std::map<std::string, tf::Output> m_generator_momentum_init;
	std::vector<tf::Operation> m_discriminator_updates;
	std::vector<tf::Operation> m_generator_updates;

	tf::Status createConvLayer(
		const int, const int, const std::string&,
		tf::Scope&, tf::Output&, tf::Output&, const bool, const int,
		std::map<std::string, tf::Output>& weights, std::map<std::string, tf::TensorShape>& weight_shapes,
		std::map<std::string, tf::Output>& assigns);


public:
	DCGAN(const int resolution, const int latent_dims) :
		m_resolution{ resolution },
		m_latent_dims{ latent_dims },
		m_discriminator_root{ tf::Scope::NewRootScope() },
		m_generator_root{ tf::Scope::NewRootScope() } {}

	~DCGAN() {}

	tf::Status createDiscriminator(const std::vector<int>&);
	tf::Status createGenerator(const std::vector<int>&);
	tf::Status createDiscOptimiser(const float learning_rate, const float beta1, const float beta2);
	tf::Status createGenOptimiser(const float learning_rate, const float beta1, const float beta2);
	tf::Status initialiseWeights();
	tf::Status trainStep(std::vector<tf::Tensor>& real_minibatch);
};

#endif // !DCGAN_H
