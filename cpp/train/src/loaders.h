#ifndef LOADERS_H
#define LOADERS_H

#include <string>
#include <vector>

#include "../src/extra.h"

#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
//#include "tensorflow/core/graph/default_device.h"
//#include "tensorflow/core/graph/graph_def_builder.h"
//#include "tensorflow/core/lib/core/errors.h"
//#include "tensorflow/core/lib/core/stringpiece.h"
//#include "tensorflow/core/lib/core/threadpool.h"
//#include "tensorflow/core/lib/io/path.h"
//#include "tensorflow/core/lib/strings/str_util.h"
//#include "tensorflow/core/lib/strings/stringprintf.h"
//#include "tensorflow/core/platform/env.h"
//#include "tensorflow/core/platform/init_main.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/summary/summary_file_writer.h"
//#include "tensorflow/core/util/command_line_flags.h"

//#include "tensorflow/core/protobuf/meta_graph.pb.h"
//#include "tensorflow/core/public/session.h"


//using tensorflow::Flag;
//using tensorflow::int32;
namespace tf = tensorflow;
namespace ops = tf::ops;
using tf::Output;
using tf::Tensor;


/* Utility functions for loading images etc.
   Heavily inspired by
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/
   and the more recent
   https://github.com/bennyfri/TFMacCpp
   (tutorial: https://towardsdatascience.com/creating-a-tensorflow-cnn-in-c-part-2-eea0de9dcada) */


namespace utils
{
	tf::Status ReadImage(const std::string&, const int width, const int height, std::vector<Tensor>*);
	tf::Status WriteImage(const std::string&, std::vector<Tensor>&);
}

#endif // LOADERS_H