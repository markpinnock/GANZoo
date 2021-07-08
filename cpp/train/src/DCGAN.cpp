#include "DCGAN.h"


//------------------------------------------------------------------------

tf::Output DCGAN::createConvLayer(
	const int in_ch, const int out_ch, const std::string& layer_idx,
	tf::Scope& scope, tf::Input& input, const bool bottom_layer, const int new_res,
	std::map<std::string, tf::Output>& weights, std::map<std::string, tf::TensorShape>& weight_shapes,
	std::map<std::string, tf::Output>& assigns)
{
	// Create sub-scope containing layer weights
	tf::Scope layer_scope = scope.NewSubScope("conv" + layer_idx);

	// Create sub-scope for weights (to be used in multiple graphs)
	std::unique_ptr<tf::Scope> weight_subscope;

	if (new_res)
	{
		weight_subscope = std::make_unique<tf::Scope>(m_root_scope.NewSubScope("g_weights" + layer_idx));
	}
	else
	{
		weight_subscope = std::make_unique<tf::Scope>(m_root_scope.NewSubScope("d_weights" + layer_idx));
	}

	// Specify shape of convolutional kernel and create weights (if not already created)
	if (!weights.count("W" + layer_idx))
	{
		tf::TensorShape kernel_dims({ 3, 3, in_ch, out_ch });
		weights["W" + layer_idx] = ops::Variable(weight_subscope->WithOpName("W"), kernel_dims, tf::DT_FLOAT);
		assigns["W" + layer_idx] = ops::Assign(
			weight_subscope->WithOpName("W_assign"), weights["W" + layer_idx],
			init::HeNormal(*weight_subscope, kernel_dims));
		weight_shapes["W" + layer_idx] = kernel_dims;
	}

	if (!weights.count("b" + layer_idx))
	{
		tf::TensorShape bias_dims({ out_ch });
		weights["b" + layer_idx] = ops::Variable(weight_subscope->WithOpName("b"), bias_dims, tf::DT_FLOAT);
		assigns["b" + layer_idx] = ops::Assign(
			weight_subscope->WithOpName("b_assign"), weights["b" + layer_idx],
			tf::Input::Initializer(0.0f, bias_dims));
		weight_shapes["b" + layer_idx] = bias_dims;
	}

	// Convolution and bias
	tf::Output x;

	if (new_res && bottom_layer) // Generator, bottom layer
	{
		x = ops::ResizeBilinear(layer_scope.WithOpName("resize"), input, tf::Input({ new_res, new_res}));

		x = ops::Conv2D(
			layer_scope.WithOpName("conv"), x, weights["W" + layer_idx], { 1, 1, 1, 1 }, "SAME");
		
		x = ops::BiasAdd(layer_scope.WithOpName("bias"), x, weights["b" + layer_idx]);

		return ops::Relu(layer_scope.WithOpName("relu"), x);
	}
	else if (new_res && !bottom_layer) // Generator, higher layers
	{
		x = ops::ResizeBilinear(layer_scope.WithOpName("resize"), input, tf::Input({ new_res, new_res }));

		x = ops::Conv2D(
			layer_scope.WithOpName("conv"), x, weights["W" + layer_idx], { 1, 1, 1, 1 }, "SAME");

		x = ops::BiasAdd(layer_scope.WithOpName("bias"), x, weights["b" + layer_idx]);

		if (new_res == m_resolution)  // Generator, final layer
		{
			return x;
		}
		else // Generator, intermediate layers
		{
			return ops::Relu(layer_scope.WithOpName("relu"), x);
		}
	}
	else if (!new_res && bottom_layer) // Discriminator, bottom layer
	{
		x = ops::Conv2D(
			layer_scope.WithOpName("conv"), input, weights["W" + layer_idx], { 1, 4, 4, 1 }, "VALID");

		return ops::BiasAdd(layer_scope.WithOpName("bias"), x, weights["b" + layer_idx]);
	}
	else // Discriminator, higher layers
	{
		x = ops::Conv2D(
			layer_scope.WithOpName("conv"), input, weights["W" + layer_idx], { 1, 1, 1, 1 }, "SAME");

		x = ops::BiasAdd(layer_scope.WithOpName("bias"), x, weights["b" + layer_idx]);
		x = ops::Relu(layer_scope.WithOpName("relu"), x);

		return ops::MaxPool(layer_scope.WithOpName("maxpool"), x, { 1, 2, 2, 1 }, { 1, 2, 2, 1 }, "SAME");
	}
}


//------------------------------------------------------------------------

tf::Output DCGAN::createDiscriminator(
	const std::vector<int>& channels, const int num_layers,
	tf::Scope& scope, tf::Input& input)
{
	tf::Output x = createConvLayer(
		m_im_ch, channels[0], std::to_string(0),
		scope, tf::Input(input), false, 0,
		m_discriminator_weights, m_discriminator_weight_shapes, m_discriminator_assigns);

	// TODO: why do we need Input?
	for (int i{ 0 }; i < num_layers - 2; ++i)
	{
		x = createConvLayer(
			channels[i], channels[i + 1], std::to_string(i + 1),
			scope, tf::Input(x), false, 0,
			m_discriminator_weights, m_discriminator_weight_shapes, m_discriminator_assigns);
	}

	x = createConvLayer(
		channels[num_layers - 2], 1, std::to_string(num_layers - 1),
		scope, tf::Input(x), true, 0,
		m_discriminator_weights, m_discriminator_weight_shapes, m_discriminator_assigns);

	return x;
}


//------------------------------------------------------------------------

tf::Output DCGAN::createGenerator(
	const std::vector<int>& channels, const int num_layers,
	tf::Scope& scope, tf::Input& input)
{
	int resolution{ 4 };

	tf::Output x = ops::Reshape(
		scope.WithOpName("reshape"), input, { -1, 1, 1, m_latent_dims });

	x = createConvLayer(
		m_latent_dims, channels[0], std::to_string(0),
		scope, tf::Input(x), true, resolution,
		m_generator_weights, m_generator_weight_shapes, m_generator_assigns);

	resolution *= 2;

	for (int i{ 0 }; i < num_layers - 2; ++i)
	{
		x = createConvLayer(
			channels[i], channels[i + 1], std::to_string(i + 1),
			scope, tf::Input(x), false, resolution,
			m_generator_weights, m_generator_weight_shapes, m_generator_assigns);

		resolution *= 2;
	}

	x = createConvLayer(
		channels[num_layers - 2], m_im_ch, std::to_string(num_layers - 1),
		scope, tf::Input(x), false, resolution,
		m_generator_weights, m_generator_weight_shapes, m_generator_assigns);

	// TODO: different std
	tf::Scope latent_scope = scope.NewSubScope("latent_scope");
	m_noise_output = ops::RandomNormal(latent_scope.WithOpName("random_noise"), tf::Input({ 1, m_latent_dims }), tf::DT_FLOAT);
	TF_CHECK_OK(latent_scope.status());

	return x;
}


//------------------------------------------------------------------------
tf::Status DCGAN::discriminatorTrainingGraph(int channels)
{
	const int num_layers{ static_cast<int>(std::log2(m_resolution)) - 1 };

	if (m_resolution < 4)
	{
		return tf::errors::Cancelled("Invalid resolution: ", m_resolution);
	}

	std::vector<int> d_channel_vec;

	for (int i{ 0 }; i < num_layers - 1; ++i)
	{
		d_channel_vec.push_back(channels);
		channels *= 2;
	}

	std::vector<int> g_channel_vec(d_channel_vec);
	std::reverse(g_channel_vec.begin(), g_channel_vec.end());
	
	// Fake images
	tf::Scope generator_scope = m_discriminator_root.NewSubScope("generator");
	m_d_latent_ph = ops::Placeholder(generator_scope.WithOpName("noise_input"), tf::DT_FLOAT);
	m_d_fake_images = createGenerator(g_channel_vec, num_layers, generator_scope, tf::Input(m_d_latent_ph));
	m_d_fake_score = createDiscriminator(d_channel_vec, num_layers, m_discriminator_root.NewSubScope("fake_disc"), tf::Input(m_d_fake_images));

	// Real images
	m_d_real_ph = ops::Placeholder(m_discriminator_root.WithOpName("real_input"), tf::DT_FLOAT);
	m_d_real_score= createDiscriminator(d_channel_vec, num_layers, m_discriminator_root.NewSubScope("real_disc"), tf::Input(m_d_real_ph));

	return m_discriminator_root.status();
}


//------------------------------------------------------------------------
tf::Status DCGAN::generatorTrainingGraph(int channels)
{
	if (m_latent_dims <= 0)
	{
		return tf::errors::Cancelled("Invalid latent dims: ", m_latent_dims);
	}

	const int num_layers{ static_cast<int>(std::log2(m_resolution)) - 1 };

	if (m_resolution < 4)
	{
		return tf::errors::Cancelled("Invalid resolution: ", m_resolution);
	}

	std::vector<int> d_channel_vec;

	for (int i{ 0 }; i < num_layers - 1; ++i)
	{
		d_channel_vec.push_back(channels);
		channels *= 2;
	}

	std::vector<int> g_channel_vec(d_channel_vec);
	std::reverse(g_channel_vec.begin(), g_channel_vec.end());

	tf::Scope generator_scope = m_generator_root.NewSubScope("generator");
	m_g_latent_ph = ops::Placeholder(generator_scope.WithOpName("noise_input"), tf::DT_FLOAT);
	m_g_fake_images = createGenerator(g_channel_vec, num_layers, generator_scope, tf::Input(m_g_latent_ph));
	m_g_fake_score = createDiscriminator(d_channel_vec, num_layers, m_generator_root.NewSubScope("fake_disc"), tf::Input(m_g_fake_images));

	return m_generator_root.status();
}


//------------------------------------------------------------------------

tf::Status DCGAN::initialiseModels()
{
	/* Before running session, check for errors - otherwise can be hard to diagnose */
	if (!m_discriminator_root.ok())
	{
		LOG(FATAL) << m_discriminator_root.status().ToString();
	}
	if (!m_generator_root.ok())
	{
		LOG(FATAL) << m_generator_root.status().ToString();
	}

	// Initialise weights
	std::vector<tf::Output> discriminator_init;
	std::vector<tf::Output> generator_init;

	for (auto el : m_discriminator_assigns)
	{
		discriminator_init.push_back(el.second);
	}

	for (auto el : m_discriminator_momentum_init)
	{
		discriminator_init.push_back(el.second);
	}

	for (auto el : m_generator_assigns)
	{
		generator_init.push_back(el.second);
	}

	for (auto el : m_generator_momentum_init)
	{
		generator_init.push_back(el.second);
	}

	m_discriminator_session = std::make_unique<tf::ClientSession>(m_discriminator_root);
	m_generator_session = std::make_unique<tf::ClientSession>(m_generator_root);

	TF_CHECK_OK(getLatentNoise(m_fixed_noise, m_num_ex));

	// Run session with vector of init ops and no output (both need to be done for each graph it appears)
	TF_CHECK_OK(m_discriminator_session->Run(discriminator_init, nullptr)); // TODO: CHECK ALL VARIABLES INIT'D
	TF_CHECK_OK(m_discriminator_session->Run(generator_init, nullptr));
	TF_CHECK_OK(m_generator_session->Run(generator_init, nullptr));
	TF_CHECK_OK(m_generator_session->Run(discriminator_init, nullptr));

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status DCGAN::getLatentNoise(std::vector<tf::Tensor>& noise_minibatch, const int num_noise)
{
	if (num_noise < 1)
	{
		return tf::errors::Cancelled("Invalid number of noise samples: ", num_noise);
	}

	std::vector<tf::Tensor> noise_tensor;
	std::vector<tf::Input> noise_to_stack;

	/* Before running session, check for errors in graph - otherwise can be hard to diagnose */
	if (!m_generator_root.ok())
	{
		LOG(FATAL) << m_generator_root.status().ToString();
	}

	for (int i{ 0 }; i < num_noise; ++i)
	{
		// TODO: does sess.run() really need std::vector as output or can it use tf::Tensor?
		TF_CHECK_OK(m_generator_session->Run({ m_noise_output }, &noise_tensor));
		noise_to_stack.push_back(tf::Input(noise_tensor[0]));
	}

	// tensorflow::ops::Stack requires tensorflow::InputList
	tf::InputList noise_to_stack_list(noise_to_stack);

	// TODO: Is this efficient?
	tf::Scope noise_root{ tf::Scope::NewRootScope() };
	TF_CHECK_OK(noise_root.status());
	tf::Output stacked_noise = ops::Stack(noise_root.WithOpName("stack"), noise_to_stack_list);

	// TODO: why does this need squeezing
	// TODO: ensure minibatch size 1 not squeezed
	tf::Output squeezed_noise = ops::Squeeze(noise_root.WithOpName("squeeze"), stacked_noise);
	tf::ClientSession session(noise_root);
	TF_CHECK_OK(session.Run({ squeezed_noise}, &noise_minibatch));

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status DCGAN::createDiscOptimiser(const float learning_rate, const float beta1, const float beta2)
{
	constexpr float epsilon{ 0.00000001f };

	// TODO: modify so that can use labels e.g. minmax
	// m_labels = ops::Placeholder(m_discriminator_root.WithOpName("labels"), tf::DT_FLOAT);
	tf::Scope loss_scope = m_discriminator_root.NewSubScope("loss_scope");
	//m_discriminator_real_ph = ops::Placeholder(m_discriminator_root.WithOpName("real"), tf::DT_FLOAT);
	//m_discriminator_fake_ph = ops::Placeholder(m_discriminator_root.WithOpName("fake"), tf::DT_FLOAT);
	m_discriminator_loss = losses::leastSquaresDiscriminator(loss_scope, m_d_real_score, m_d_fake_score);
	
	// TODO: TF_RETURN_IF_ERROR?
	TF_CHECK_OK(loss_scope.status());

	std::vector<tf::Output> weights;
	std::vector<tf::Output> gradients;

	for (auto el : m_discriminator_weights)
	{
		weights.push_back(el.second);
	}

	TF_CHECK_OK(tf::AddSymbolicGradients(m_discriminator_root.WithOpName("grads"), { m_discriminator_loss }, weights, &gradients));

	// Use gradients to update model weights
	// TODO - put in subscope?
	int i{ 0 };

	for (auto el : m_discriminator_weights)
	{
		tf::Output m = ops::Variable(m_discriminator_root.WithOpName("m"), m_discriminator_weight_shapes[el.first], tf::DT_FLOAT);
		tf::Output v = ops::Variable(m_discriminator_root.WithOpName("v"), m_discriminator_weight_shapes[el.first], tf::DT_FLOAT);
		m_discriminator_momentum_init["m" + std::to_string(i)] = ops::Assign(m_discriminator_root.WithOpName("m_init"), m, tf::Input::Initializer(0.0f, m_discriminator_weight_shapes[el.first]));
		m_discriminator_momentum_init["v" + std::to_string(i)] = ops::Assign(m_discriminator_root.WithOpName("v_init"), v, tf::Input::Initializer(0.0f, m_discriminator_weight_shapes[el.first]));

		auto d_update = ops::ApplyAdam(loss_scope.WithOpName("Adam"), el.second, m, v, 0.0f, 0.0f, learning_rate, beta1, beta2, epsilon, { gradients[i] });
		m_discriminator_updates.push_back(d_update.operation);
		++i;
	}

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status DCGAN::createGenOptimiser(const float learning_rate, const float beta1, const float beta2)
{
	constexpr float epsilon{ 0.00000001f };

	// TODO: modify so that can use labels e.g. minmax
	// m_labels = ops::Placeholder(m_generator_root.WithOpName("labels"), tf::DT_FLOAT);
	tf::Scope loss_scope = m_generator_root.NewSubScope("loss_scope");
	m_generator_loss = losses::leastSquaresGenerator(loss_scope, m_g_fake_score);
	TF_CHECK_OK(loss_scope.status());

	std::vector<tf::Output> weights;
	std::vector<tf::Output> gradients;

	for (auto el : m_generator_weights)
	{
		weights.push_back(el.second);
	}

	TF_CHECK_OK(tf::AddSymbolicGradients(m_generator_root.WithOpName("grads"), { m_generator_loss }, weights, &gradients));

	// Use gradients to update model weights
	// TODO - put in subscope?
	int i{ 0 };

	for (auto el : m_generator_weights)
	{
		// Operations for initialise momentum variables to zero
		// TODO: CHECK THESE ARE INITIALISED
		tf::Output m = ops::Variable(m_generator_root.WithOpName("m"), m_generator_weight_shapes[el.first], tf::DT_FLOAT);
		tf::Output v = ops::Variable(m_generator_root.WithOpName("v"), m_generator_weight_shapes[el.first], tf::DT_FLOAT);
		m_generator_momentum_init["m" + std::to_string(i)] = ops::Assign(m_generator_root.WithOpName("m_init"), m, tf::Input::Initializer(0.0f, m_generator_weight_shapes[el.first]));
		m_generator_momentum_init["v" + std::to_string(i)] = ops::Assign(m_generator_root.WithOpName("v_init"), v, tf::Input::Initializer(0.0f, m_generator_weight_shapes[el.first]));

		// Optimisation op
		auto g_update = ops::ApplyAdam(loss_scope.WithOpName("Adam"), el.second, m, v, 0.0f, 0.0f, learning_rate, beta1, beta2, epsilon, { gradients[i] });
		m_generator_updates.push_back(g_update.operation);
		++i;
	}

	return tf::Status::OK();
}


//------------------------------------------------------------------------
tf::Status DCGAN::runGenerator(std::vector<tf::Tensor>& output)
{
	if (!m_generator_root.ok())
	{
		LOG(FATAL) << m_generator_root.status().ToString();
	}

	TF_CHECK_OK(m_generator_session->Run({ {m_g_latent_ph, m_fixed_noise[0]} }, { m_g_fake_images }, &output));
	return tf::Status::OK();
}


//------------------------------------------------------------------------
tf::Status DCGAN::runGenerator(std::vector<tf::Tensor>& input, std::vector<tf::Tensor>& output)
{
	if (!m_generator_root.ok())
	{
		LOG(FATAL) << m_generator_root.status().ToString();
	}

	TF_CHECK_OK(m_generator_session->Run({ {m_g_latent_ph, input[0]} }, { m_g_fake_images }, &output));
	return tf::Status::OK();
}


//------------------------------------------------------------------------
tf::Status DCGAN::runDiscriminator(std::vector<tf::Tensor>& input, std::vector<tf::Tensor>& output)
{
	if (!m_discriminator_root.ok())
	{
		LOG(FATAL) << m_discriminator_root.status().ToString();
	}

	TF_CHECK_OK(m_discriminator_session->Run({ {m_d_real_ph, input[0]} }, { m_d_real_score }, &output));
	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status DCGAN::trainStep(
	std::vector<tf::Tensor>& real_minibatch,
	std::vector<float>& discriminator_losses,
	std::vector<float>& generator_losses)
{
	/* Before running session, check for errors - otherwise can be hard to diagnose */
	if (!m_discriminator_root.ok())
	{
		LOG(FATAL) << m_discriminator_root.status().ToString();
	}
	if (!m_generator_root.ok())
	{
		LOG(FATAL) << m_generator_root.status().ToString();
	}

	// TODO: Necessary?
	if (real_minibatch.size() != 1)
	{
		return tf::errors::Cancelled("Minibatch std::vector must be of size 1: ", real_minibatch.size());
	}

	std::vector<tf::Tensor> fake_minibatch;
	std::vector<tf::Tensor> latent_in;
	std::vector<tf::Tensor> real_scores;
	std::vector<tf::Tensor> fake_scores;
	std::vector<tf::Tensor> losses;

	// Discriminator training
	TF_CHECK_OK(getLatentNoise(latent_in, real_minibatch.size()));

	TF_CHECK_OK(m_discriminator_session->Run(
		{ {m_d_real_ph, real_minibatch[0]}, {m_d_latent_ph, latent_in[0]} },
		{ m_discriminator_loss },
		m_discriminator_updates, &losses));

	discriminator_losses.push_back(losses[0].tensor<float, 0>()(0));
	losses.clear();
	latent_in.clear();

	// Generator training
	TF_CHECK_OK(getLatentNoise(latent_in, real_minibatch.size()));

	TF_CHECK_OK(m_generator_session->Run(
		{ {m_g_latent_ph, latent_in[0]} },
		{ m_generator_loss },
		m_generator_updates, &losses));

	generator_losses.push_back(losses[0].tensor<float, 0>()(0));

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status DCGAN::TensorboardGraph()
{
	tf::GraphDef graph;
	TF_RETURN_IF_ERROR(m_root_scope.ToGraphDef(&graph));
	tf::SummaryWriterInterface* w;
	TF_CHECK_OK(tf::CreateSummaryFileWriter(1, 0, "./graphs/", ".img-graph", tf::Env::Default(), &w));
	TF_CHECK_OK(w->WriteGraph(0, std::make_unique<tf::GraphDef>(graph)));
	return tf::Status::OK();
}


//------------------------------------------------------------------------

void DCGAN::printModel()
{
	std::cout << "==============================" << std::endl
		      << "Discriminator weights" << std::endl
		      << "==============================" << std::endl;

	for (auto el : m_discriminator_weight_shapes)
	{
		std::cout << el.first << ": " << el.second.DebugString() << std::endl;
	}

	std::cout << "==============================" << std::endl
		      << "Generator weights" << std::endl
		      << "==============================" << std::endl;

	for (auto el : m_generator_weight_shapes)
	{
		std::cout << el.first << ": " << el.second.DebugString() << std::endl;
	}
}