#ifndef BLOCKS_H
#define BLOCKS_H

#include "../src/extra.h"

#include <string>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


namespace tf = tensorflow;
namespace ops = tf::ops;


namespace init
{
	tf::Output HeNormal(tf::Scope&, tf::TensorShape&);
}

namespace losses
{
	tf::Output leastSquaresDiscriminator(tf::Scope& scope, tf::Output& real, tf::Output& fake);
	tf::Output leastSquaresGenerator(tf::Scope& scope, tf::Output& fake);
}

#endif // !BLOCKS_H

