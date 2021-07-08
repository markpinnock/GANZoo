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
#include "tensorflow/core/summary/summary_file_writer.h"


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
	tf::Output m_d_real_ph;
	//tf::Output m_fake_ph;
	tf::Output m_g_latent_ph;
	tf::Output m_d_latent_ph;
	tf::Output m_num_latent_ph; // ???

	// Fixed latent noise
	const int m_num_ex{ 16 };
	const int m_latent_dims;
	std::vector<tf::Tensor> m_fixed_noise;

	// Output tensors
	tf::Output m_noise_output;
	tf::Output m_g_noise_output; // ???

	//tf::Output m_discriminator_output;
	tf::Output m_d_fake_images;
	tf::Output m_g_fake_images;
	tf::Output m_d_real_score;
	tf::Output m_d_fake_score;
	tf::Output m_g_fake_score;

	tf::Output m_discriminator_loss;
	tf::Output m_generator_loss;

	// Sessions for discriminator and generator scopes
	std::unique_ptr<tf::ClientSession> m_discriminator_session;
	std::unique_ptr<tf::ClientSession> m_generator_session;

	tf::Scope m_root_scope;
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

	tf::Output createConvLayer(
		const int in_ch, const int out_ch, const std::string& layer_idx,
		tf::Scope& scope, tf::Input& input, const bool bottom_layer, const int new_res,
		std::map<std::string, tf::Output>& weights, std::map<std::string, tf::TensorShape>& weight_shapes,
		std::map<std::string, tf::Output>& assigns);


public:
	DCGAN(const int resolution, const int latent_dims) :
		m_resolution{ resolution },
		m_latent_dims{ latent_dims },
		m_root_scope{ tf::Scope::NewRootScope() },
		m_discriminator_root{ m_root_scope.NewSubScope("discriminator_graph") },
		m_generator_root{ m_root_scope.NewSubScope("generator_graph") } {}

	~DCGAN() {}

	tf::Output createDiscriminator(
		const std::vector<int>& channels, const int num_layers,
		tf::Scope& scope, tf::Input& input);
	tf::Output createGenerator(
		const std::vector<int>& channels, const int num_layers,
		tf::Scope& scope, tf::Input& input);
	tf::Status discriminatorTrainingGraph(int channels);
	tf::Status generatorTrainingGraph(int channels);
	tf::Status getLatentNoise(std::vector<tf::Tensor>& noise_minibatch, const int num_noise);
	tf::Status createDiscOptimiser(const float learning_rate, const float beta1, const float beta2);
	tf::Status createGenOptimiser(const float learning_rate, const float beta1, const float beta2);
	tf::Status initialiseModels();
	tf::Status runGenerator(std::vector<tf::Tensor>& generated_images);
	tf::Status runGenerator(std::vector<tf::Tensor>& input, std::vector<tf::Tensor>& output);
	tf::Status runDiscriminator(std::vector<tf::Tensor>& input, std::vector<tf::Tensor>& output);
	tf::Status trainStep(
		std::vector<tf::Tensor>& real_minibatch,
		std::vector<float>& discriminator_losses,
		std::vector<float>& generator_losses);
	tf::Status TensorboardGraph();
	void printModel();
};

#endif // !DCGAN_H
