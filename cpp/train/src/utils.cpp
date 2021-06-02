#include "utils.h"


//------------------------------------------------------------------------

tf::Output init::HeNormal(tf::Scope& scope, tf::TensorShape& shape)
{
	int fan_in{ 1 };

	for (int i{ 0 }; i < shape.dims() - 1; i++)
	{
		fan_in *= shape.dim_size(i);
	}

	float std_dev{ std::sqrt(2.0f / fan_in) };

	tf::Output rand;

	// ops::RandomNormal takes tf::Input as shape, not tf::TensorShape
	if (shape.dims() == 2)
	{
		rand = ops::RandomNormal(
			scope,
			tf::Input({ shape.dim_size(0), shape.dim_size(1) }),
			tf::DT_FLOAT);
	}
	else
	{
		rand = ops::RandomNormal(
			scope,
			tf::Input({ shape.dim_size(0), shape.dim_size(1) , shape.dim_size(2) , shape.dim_size(3) }),
			tf::DT_FLOAT);
	}

	return ops::Multiply(scope, rand, std_dev);
}


//------------------------------------------------------------------------

tf::Output losses::leastSquaresDiscriminator(tf::Scope& scope, tf::Output& real, tf::Output& fake)
{
	tf::Output real_loss = ops::Subtract(scope, real, 1.0f);
	real_loss = ops::Square(scope.WithOpName("real_square"), real_loss);
	real_loss = ops::ReduceMean(scope.WithOpName("real_mean"), real_loss);
	real_loss = ops::Multiply(scope.WithOpName("real_mult"), real_loss, 0.5f);

	tf::Output fake_loss = ops::Square(scope.WithOpName("fake_square"), fake);
	fake_loss = ops::ReduceMean(scope.WithOpName("fake_mean"), fake_loss);
	fake_loss = ops::Multiply(scope.WithOpName("fake_mult"), fake_loss, 0.5f);

	return ops::Add(scope, real_loss, fake_loss);
}


//------------------------------------------------------------------------

tf::Output losses::leastSquaresGenerator(tf::Scope& scope, tf::Output& fake)
{
	tf::Output fake_loss = ops::Subtract(scope, fake, 1.0f);
	fake_loss = ops::Square(scope, fake_loss);
	fake_loss = ops::ReduceMean(scope, fake_loss);
	fake_loss = ops::Multiply(scope, fake_loss, 0.5f);

	return fake_loss;
}