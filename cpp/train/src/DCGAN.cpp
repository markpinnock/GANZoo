#include "DCGAN.h"


//------------------------------------------------------------------------

tf::Status DCGAN::createConvLayer(
	const int in_ch, const int out_ch, const std::string& layer_idx,
	tf::Scope& scope, tf::Output& input, tf::Output& output, const bool bottom_layer, const int new_res,
	std::map<std::string, tf::Output>& weights, std::map<std::string, tf::TensorShape>& weight_shapes,
	std::map<std::string, tf::Output>& assigns)
{
	// Create sub-scope containing layer weights
	tf::Scope layer_scope = scope.NewSubScope("conv" + layer_idx);

	// Specify shape of convolutional kernel and create weights
	tf::TensorShape kernel_dims({ 3, 3, in_ch, out_ch });
	weights["W" + layer_idx] = ops::Variable(layer_scope.WithOpName("W"), kernel_dims, tf::DT_FLOAT);
	assigns["W" + layer_idx] = ops::Assign(
		layer_scope.WithOpName("W_assign"), weights["W" + layer_idx],
		init::HeNormal(layer_scope, kernel_dims));

	tf::TensorShape bias_dims({ out_ch });
	weights["b" + layer_idx] = ops::Variable(layer_scope.WithOpName("b"), bias_dims, tf::DT_FLOAT);
	assigns["b" + layer_idx] = ops::Assign(
		layer_scope.WithOpName("b_assign"), weights["b" + layer_idx],
		tf::Input::Initializer(0.0f, bias_dims));

	weight_shapes["W" + layer_idx] = kernel_dims;
	weight_shapes["b" + layer_idx] = bias_dims;

	// Convolution and bias
	tf::Output conv;

	if (new_res && bottom_layer) // Generator, bottom layer
	{
		tf::Output resize = ops::ResizeBilinear(layer_scope.WithOpName("resize"), input, tf::Input({ new_res, new_res}));
		conv = ops::Conv2D(
			layer_scope.WithOpName("conv"), resize, weights["W" + layer_idx], { 1, 1, 1, 1 }, "SAME");
	}
	else if (new_res && !bottom_layer) // Generator, higher layers
	{
		tf::Output resize = ops::ResizeBilinear(layer_scope.WithOpName("resize"), input, tf::Input({ new_res, new_res }));
		conv = ops::Conv2D(
			layer_scope.WithOpName("conv"), resize, weights["W" + layer_idx], { 1, 1, 1, 1 }, "SAME");
	}
	else if (!new_res && bottom_layer) // Discriminator, bottom layer
	{
		conv = ops::Conv2D(
			layer_scope.WithOpName("conv"), input, weights["W" + layer_idx], { 1, 4, 4, 1 }, "VALID");
	}
	else // Discriminator, higher layers
	{
		conv = ops::Conv2D(
			layer_scope.WithOpName("conv"), input, weights["W" + layer_idx], { 1, 1, 1, 1 }, "SAME");
	}

	tf::Output bias = ops::BiasAdd(layer_scope.WithOpName("bias"), conv, weights["b" + layer_idx]);
	output = ops::Relu(layer_scope.WithOpName("relu"), bias);
	
	if (!new_res && !bottom_layer)
	{
		output = ops::MaxPool(layer_scope.WithOpName("maxpool"), output, { 1, 2, 2, 1 }, { 1, 2, 2, 1 }, "SAME");
	}

	return layer_scope.status();
}


//------------------------------------------------------------------------

tf::Status DCGAN::createDiscriminator(const std::vector<int>& channels)
{
	const int num_layers{ static_cast<int>(std::log2(m_resolution)) - 1 };

	if (m_resolution < 4)
	{
		return tf::errors::Cancelled("Invalid resolution: ", m_resolution);
	}

	if (num_layers != channels.size() + 1)
	{
		return tf::errors::Cancelled("Number of layers != number of channels: ", num_layers, " ", channels.size());
	}

	m_image_ph = ops::Placeholder(m_discriminator_root.WithOpName("input"), tf::DT_FLOAT);
	tf::Output input;
	tf::Output output;

	TF_RETURN_IF_ERROR(
		createConvLayer(
			m_im_ch, channels[0], "0", m_discriminator_root, m_image_ph, output, false, 0,
			m_discriminator_weights, m_discriminator_weight_shapes, m_discriminator_assigns));

	for (int i{ 0 }; i < num_layers - 2; ++i)
	{
		input = output;
		TF_RETURN_IF_ERROR(
			createConvLayer(
				channels[i], channels[i + 1], std::to_string(i), m_discriminator_root, input, output, false, 0,
				m_discriminator_weights, m_discriminator_weight_shapes, m_discriminator_assigns));
	}

	input = output;
	TF_RETURN_IF_ERROR(
		createConvLayer(
			channels[num_layers - 2], 1, std::to_string(num_layers), m_discriminator_root, input, output, true, 0,
			m_discriminator_weights, m_discriminator_weight_shapes, m_discriminator_assigns));

	input = output;
	m_discriminator_output = ops::Squeeze(m_discriminator_root.WithOpName("squeeze"), input);

	return m_discriminator_root.status();
}


//------------------------------------------------------------------------

tf::Status DCGAN::createGenerator(const std::vector<int>& channels)
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

	if (num_layers != channels.size() + 1)
	{
		return tf::errors::Cancelled("Number of layers != number of channels: ", num_layers, " ", channels.size());
	}

	m_latent_ph = ops::Placeholder(m_generator_root.WithOpName("input"), tf::DT_FLOAT);
	tf::Output input;
	tf::Output output;
	int resolution{ 4 };

	input = ops::Reshape(m_generator_root.WithOpName("reshape"), m_latent_ph, { -1, 1, 1, m_latent_dims });

	TF_RETURN_IF_ERROR(
		createConvLayer(m_latent_dims, channels[0], std::to_string(0), m_generator_root, input, output, true, resolution));

	resolution *= 2;

	for (int i{ 0 }; i < num_layers - 2; ++i)
	{
		input = output;
		TF_RETURN_IF_ERROR(
			createConvLayer(channels[i], channels[i + 1], std::to_string(i + 1), m_generator_root, input, output, false, resolution));

		resolution *= 2;
	}

	input = output;
	TF_RETURN_IF_ERROR(
		createConvLayer(channels[num_layers - 2], m_im_ch, std::to_string(num_layers), m_generator_root, input, m_generator_output, false, resolution));

	return m_generator_root.status();
}


//------------------------------------------------------------------------

tf::Status DCGAN::createDiscOptimiser(const float learning_rate, const float beta1, const float beta2)
{
	constexpr float epsilon{ 0.00000001f };

	// TODO: modify so that can use labels e.g. minmax
	// m_labels = ops::Placeholder(m_discriminator_root.WithOpName("labels"), tf::DT_FLOAT);
	tf::Scope loss_scope = m_discriminator_root.NewSubScope("loss_scope");
	m_discriminator_loss = losses::leastSquaresDiscriminator(loss_scope, m_discriminator_real, m_discriminator_fake);
	TF_CHECK_OK(loss_scope.status());

	std::vector<tf::Output> weights;
	std::vector<tf::Output> gradients;

	for (auto el : m_discriminator_weights)
	{
		weights.push_back(el.second);
	}

	TF_CHECK_OK(tf::AddSymbolicGradients(m_discriminator_root, { m_discriminator_loss }, weights, &gradients));

	// Use gradients to update model weights
	// TODO - put in subscope?
	int i{ 0 };

	for (auto el : m_discriminator_weights)
	{
		tf::Output m = ops::Variable(m_discriminator_root, m_discriminator_weight_shapes[el.first], tf::DT_FLOAT);
		tf::Output v = ops::Variable(m_discriminator_root, m_discriminator_weight_shapes[el.first], tf::DT_FLOAT);
		m_discriminator_momentum_init["m" + std::to_string(i)] = ops::Assign(m_discriminator_root, m, tf::Input::Initializer(0.0f, m_discriminator_weight_shapes[el.first]));
		m_discriminator_momentum_init["m" + std::to_string(i)] = ops::Assign(m_discriminator_root, v, tf::Input::Initializer(0.0f, m_discriminator_weight_shapes[el.first]));

		auto d_update = ops::ApplyAdam(loss_scope, el.second, m, v, 0.0f, 0.0f, learning_rate, beta1, beta2, epsilon, { gradients[i] });
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
	m_generator_loss = losses::leastSquaresGenerator(m_generator_root, m_discriminator_fake);
	TF_CHECK_OK(loss_scope.status());

	std::vector<tf::Output> weights;
	std::vector<tf::Output> gradients;

	for (auto el : m_generator_weights)
	{
		weights.push_back(el.second);
	}

	TF_CHECK_OK(tf::AddSymbolicGradients(m_generator_root, { m_generator_loss }, weights, &gradients));

	// Use gradients to update model weights
	// TODO - put in subscope?
	int i{ 0 };

	for (auto el : m_generator_weights)
	{
		// Operations for initialise momentum variables to zero
		tf::Output m = ops::Variable(m_generator_root, m_generator_weight_shapes[el.first], tf::DT_FLOAT);
		tf::Output v = ops::Variable(m_generator_root, m_generator_weight_shapes[el.first], tf::DT_FLOAT);
		m_generator_momentum_init["m" + std::to_string(i)] = ops::Assign(m_generator_root, m, tf::Input::Initializer(0.0f, m_generator_weight_shapes[el.first]));
		m_generator_momentum_init["v" + std::to_string(i)] = ops::Assign(m_generator_root, v, tf::Input::Initializer(0.0f, m_generator_weight_shapes[el.first]));

		// Optimisation op
		auto g_update = ops::ApplyAdam(loss_scope, el.second, m, v, 0.0f, 0.0f, learning_rate, beta1, beta2, epsilon, { gradients[i] });
		m_generator_updates.push_back(g_update.operation);
		++i;
	}

	return tf::Status::OK();
}


tf::Status DCGAN::initialiseWeights()
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

	for (auto el : m_discriminator_assigns)
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


	// Create fixed latent noise vectors (i.e: example images for visualising progress)
	tf::Tensor noise;
	tf::Output random;
	std::vector<tf::Input> noise_to_stack;

	tf::Scope latent_scope = m_generator_root.NewSubScope("latent_scope");
	random = ops::RandomNormal(latent_scope, tf::Input(m_latent_dims), tf::DT_FLOAT);

	m_discriminator_session = std::make_unique<tf::ClientSession>(m_discriminator_root);
	m_generator_session = std::make_unique<tf::ClientSession>(m_generator_root);

	for (int i{ 0 }; i < m_num_ex; ++i)
	{
		TF_CHECK_OK(m_generator_session.Run({ random }, &noise));
		noise_to_stack.push_back(tf::Input(noise));
	}

	// tensorflow::ops::Stack requires tensorflow::InputList
	tf::InputList images_to_stack_list(noise_to_stack);

	// TODO: Is this efficient?
	tf::Scope root{ tf::Scope::NewRootScope() };
	TF_CHECK_OK(root.status());
	Output stacked_images = ops::Stack(root, images_to_stack_list);

	// TODO: why does this need squeezing
	Output squeezed_images = ops::Squeeze(root, stacked_images);
	tf::ClientSession session(root);
	TF_CHECK_OK(session.Run({ squeezed_images }, &image_minibatch));

	// Run session with vector of init ops and no output
	TF_CHECK_OK(m_discriminator_session->Run(discriminator_init, nullptr));
	TF_CHECK_OK(m_generator_session->Run(generator_init, nullptr));

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status DCGAN::trainStep(std::vector<tf::Tensor>& real_minibatch)
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

	vector<Tensor> out_tensors;
	fake_minibatch;
	// Use generator to get fake minibatch
	std::vector<tf::Tensor> latent_in;

	for (int i{ 0 }; i < real_minibatch.size(); ++i)
	{

	}

	TF_CHECK_OK(
		m_generator_session->Run(
			{ {m_latent_ph, image}, {drop_rate_var, 1.f}, {skip_drop_var, 1.f} },
			{ out_classification },
			&out_tensors));

	TF_CHECK_OK(t_session->Run({ {input_batch_var, image_batch}, {input_labels_var, label_batch}, {drop_rate_var, 0.5f}, {skip_drop_var, 0.f} }, { out_loss_var, out_classification }, v_out_grads, &out_tensors));

	return Status::OK();
}