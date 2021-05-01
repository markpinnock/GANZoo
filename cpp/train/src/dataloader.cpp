#include "dataloader.h"


//------------------------------------------------------------------------

tf::Status Dataloader::loadFilenames(std::string path)
{
	m_file_path = path;

	// Env provides interface with file system
	tf::Env* env = tf::Env::Default();
	TF_RETURN_IF_ERROR(env->IsDirectory(m_file_path));

	// Get vector of image file names
	TF_RETURN_IF_ERROR(env->GetChildren(m_file_path, &m_img_names));

	for (auto& el : m_img_names)
	{
		el = m_file_path + el;
	}

	m_num_images = m_img_names.size();

	if (m_num_images % m_mb_size)
	{
		m_num_mb = static_cast<int>(m_num_images / m_mb_size) + 1;
	}
	else
	{
		m_num_mb = static_cast<int>(m_num_images / m_mb_size);
	}

	return tf::Status::OK();
}


//------------------------------------------------------------------------
/* Defines the graph used for loading minibatches of images */

tf::Status Dataloader::createImageGraph(const int height, const int width)
// TODO: add conditional for type of image
{
	/* Graph can take placeholder as input. Later, concrete inputs are
	   passed into the graph via the placeholders when session is run.
	   Ops return tensors of type tensorflow::Output; these are then used
	   as input of next op (implicity converted to Input in op constructor) */

	constexpr int channels{ 3 };

	// Set up placeholder and first op
	m_file_name_ph = ops::Placeholder(m_image_root.WithOpName("input"), tf::DT_STRING);
	Output file_reader = ops::ReadFile(m_image_root.WithOpName("file_reader"), m_file_name_ph);

	// Ops take the graph scope as arguments along with input
	Output image_reader = ops::DecodeJpeg(
		m_image_root.WithOpName("img_reader"),  // Scope
		file_reader,							// Input
		ops::DecodeJpeg::Channels(channels));	// Struct of additional options

	// Cast image to DT_FLOAT, expand dims from HWC to NHWC and resize
	Output float_cast = ops::Cast(m_image_root.WithOpName("float_cast"), image_reader, tf::DT_FLOAT);
	Output expand_dims = ops::ExpandDims(m_image_root, float_cast, 0);
	Output img_size = ops::Const(m_image_root.WithOpName("size"), { height, width });
	Output resize = ops::ResizeBilinear(m_image_root, expand_dims, img_size);

	// Normalise to range [-1, 1]
	Output reduce_dims = ops::Const(m_image_root.WithOpName("dims"), { 1, 2, 3 });
	Output min = ops::Min(m_image_root, resize, reduce_dims);
	Output max = ops::Max(m_image_root, resize, reduce_dims);

	Output numerator = ops::Sub(m_image_root.WithOpName("numerator"), resize, min);
	Output denominator = ops::Sub(m_image_root.WithOpName("denominator"), max, min);

	Output div = ops::Div(
		m_image_root.WithOpName("div"), numerator, denominator);
	Output mul = ops::Multiply(
		m_image_root.WithOpName("mul"), div, 2.0f);
	m_normalised_img = ops::Sub(
		m_image_root.WithOpName("sub"), mul, 1.0f);

	// Create session for loading images
	//m_image_sess(m_image_root);

	return m_image_root.status();
}

//------------------------------------------------------------------------


tf::Status Dataloader::getMinibatch(std::vector<Tensor>& image_minibatch)
{
	std::vector<Tensor> image_tensor;
	std::vector<tf::Input> images_to_stack;

	//TODO: seed
	if (!m_mb_idx) { std::random_shuffle(m_img_names.begin(), m_img_names.end()); }

	auto it_begin = m_img_names.begin() + (m_mb_size * m_mb_idx);
	auto it_end = m_img_names.begin() + (m_mb_size * (m_mb_idx + 1));
	if (it_end > m_img_names.end()) { it_end = m_img_names.end(); }

	tf::ClientSession image_sess(m_image_root);

	/* Before running session, check for errors - otherwise can be hard to diagnose */
	if (!m_image_root.ok()) LOG(FATAL) << m_image_root.status().ToString();

	for (auto it = it_begin; it < it_end; ++it)
	{
		TF_CHECK_OK(
			image_sess.Run(
				{ {m_file_name_ph, *it} },
				{ m_normalised_img },
				&image_tensor));
		images_to_stack.push_back(tf::Input(image_tensor[0]));
	}

	// tensorflow::ops::Stack requires tensorflow::InputList
	tf::InputList images_to_stack_list(images_to_stack);

	// TODO: Is this efficient?
	tf::Scope root = tf::Scope::NewRootScope();
	TF_CHECK_OK(root.status());
	Output stacked_images = ops::Stack(root, images_to_stack_list);
	tf::ClientSession session(root);
	TF_CHECK_OK(session.Run({ stacked_images }, &image_minibatch));

	m_mb_idx += 1;
	return tf::Status::OK();
}