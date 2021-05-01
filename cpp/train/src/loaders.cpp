#include "loaders.h"

#include <fstream>
#include <memory>
#include <utility>


//------------------------------------------------------------------------
/* Loads a file into a scalar tensor */

static tf::Status ReadFile(
	tf::Env* env, /* Provides interface for file system */
	const std::string& filename,
	Tensor* output
)
{
	// Get file size using Env
	tf::uint64 file_size{ 0 };
	TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

	std::string contents;
	contents.resize(file_size);

	// Use Env to set up access to file
	std::unique_ptr<tf::RandomAccessFile> file;
	TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

	// Read file into data and check size
	// StringPiece is an alias of absl::string_view
	tf::StringPiece data;
	TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &contents[0]));

	if (data.size() != file_size)
	{
		return tf::errors::DataLoss("Truncated ", filename, " expected ", file_size, " got ", data.size());
	}

	// tstrings are a replacement for std::string when using tensors
	// Set output tensor to a scalar tstring (of the data StringPiece)
	output->scalar<tf::tstring>()() = tf::tstring(data);

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status utils::ReadImage(
	const std::string& file_name,
	const int width,
	const int height,
	std::vector<Tensor>* out_tensors
)
{
	// Start by declaring the root scope (within which all ops will take place)
	tf::Scope root = tf::Scope::NewRootScope();

	/* Graph can take placeholder as input. Later, concrete inputs are
	   passed into the graph via the placeholders when session is run.
	   Ops return tensors of type tensorflow::Output; these are then used
	   as input of next op (implicity converted to Input in op constructor) */

	// Set up placeholder and first op
	Output file_name_ph = ops::Placeholder(root.WithOpName("input"), tf::DT_STRING);
	auto file_reader = ops::ReadFile(root.WithOpName("file_reader"), file_name_ph);

	// This will be the output of the image read op
	const int channels{ 3 };
	Output image_reader;

	/* Ops take a scope as arguments along with input */

	if (tf::str_util::EndsWith(file_name, ".png"))
	{
		image_reader = ops::DecodePng(
			root.WithOpName("png_reader"),     /* scope */
			file_reader,				       /* input of first op in graph is placeholder */
			ops::DecodePng::Channels(channels) /* additional args */
		);
	}
	else if (tf::str_util::EndsWith(file_name, ".gif"))
	{
		// DecodeGif returns 4D tensor
		image_reader = tf::ops::Squeeze(
			root.WithOpName("squeeze_first_dim"),
			ops::DecodeGif(
				root.WithOpName("gif_reader"),
				file_reader
			)
		);
	}
	else if (tf::str_util::EndsWith(file_name, ".bmp"))
	{
		image_reader = ops::DecodeBmp(
			root.WithOpName("bmp_reader"),
			file_reader
		);
	}
	else
	{
		image_reader = ops::DecodeJpeg(
			root.WithOpName("jpeg_reader"),
			file_reader,
			ops::DecodeJpeg::Channels(channels)
		);
	}

	// Cast image to DT_FLOAT, expand dims from HWC to NHWC and resize
	Output float_cast = ops::Cast(root.WithOpName("float_cast"), image_reader, tf::DT_FLOAT);
	Output expand_dims = ops::ExpandDims(root, float_cast, 0);
	Output img_size = ops::Const(root.WithOpName("size"), { height, width });
	Output resize = ops::ResizeBilinear(root, expand_dims, img_size);

	// Normalise
	Output reduce_dims = ops::Const(root.WithOpName("dims"), { 1, 2, 3 });
	Output min = ops::Min(root, resize, reduce_dims);
	Output max = ops::Max(root, resize, reduce_dims);

	Output numerator = ops::Sub(
		root.WithOpName("numerator"), /* scope */
		resize,						  /* input 1 */
		{ min }  					  /* input 2 */
	);

	// Calculate intensity range
	Output denominator = ops::Sub(
		root.WithOpName("denominator"), max, { min });

	// Convert inensity range to [-1, 1]
	Output div = ops::Div(
		root.WithOpName("div"), numerator, { denominator });
	Output mul = ops::Multiply(
		root.WithOpName("mul"), div, { 2.0f });
	Output normalised = ops::Sub(
		root.WithOpName("sub"), mul, { 1.0f });

	/* Session is used to run the ops in the graph */
	tf::ClientSession session(root);

	/* Before running session, check for errors - otherwise can be hard to diagnose */
	if (!root.ok()) LOG(FATAL) << root.status().ToString();

	TF_CHECK_OK(
		session.Run(
			{ {file_name_ph, file_name} },  /* Input placeholder and data */
			{ normalised },					/* Output node from session */
			out_tensors						/* Outputs in Tensor form */
		)
	);

	// TODO
	if (false)
	{
		// Extract graph from root scope and store in tensorflow::GraphDef
		tf::GraphDef graph;
		TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
		
		// Write graph to file, max queue 1, wait 0 ms
		tf::SummaryWriterInterface* w;
		TF_CHECK_OK(tf::CreateSummaryFileWriter(1, 0, "path/to/loc", ".img-graph", tf::Env::Default(), &w));
		TF_CHECK_OK(w->WriteGraph(0, std::make_unique<tf::GraphDef>(graph)));
	}

	return tf::Status::OK();
}


//------------------------------------------------------------------------

tf::Status utils::WriteImage(
	const std::string& file_name,
	std::vector<Tensor>& in_tensors
)
{
	// Start by declaring the root scope (within which all ops will take place)
	tf::Scope root = tf::Scope::NewRootScope();

	// Un-normalise [-1, 1] image - output will be [0, 255]
	// We clip to [-1, 1] as the linear activation may produce pixel intensities outisde this range
	Output clip = ops::ClipByValue(
		root.WithOpName("clip"), in_tensors[0], -1.0f, 1.0f);
	Output add = ops::Add(
		root.WithOpName("add"), clip, { 1.0f });
	Output div = ops::Div(
		root.WithOpName("div"), add, { 2.0f });
	Output mul = ops::Mul(
		root.WithOpName("mul"), div, { 255.0f });
	Output cast = ops::Cast(
		root.WithOpName("cast"), mul, tf::DT_UINT8);

	// Convert tensor to image
	Output reshape = ops::Reshape(
		root.WithOpName("reshape"), cast, ops::Const(root, { 256, 256, 3 }));
	Output image = ops::EncodePng(root.WithOpName("png_writer"), reshape);

	/* Before running session, check for errors - otherwise can be hard to diagnose */
	if (!root.ok()) LOG(FATAL) << root.status().ToString();

	std::vector<Tensor> out_tensors;
	tf::ClientSession session(root);

	/* Placeholder not used here - can only run this graph using in_tensors */
	TF_CHECK_OK(session.Run({ image }, &out_tensors));

	std::ofstream fout(file_name, std::ios::binary);
	fout << out_tensors[0].scalar<tf::tstring>()();

	return tf::Status::OK();
}