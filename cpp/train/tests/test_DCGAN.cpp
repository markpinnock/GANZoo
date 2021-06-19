#include "../src/DCGAN.h"

#include "gtest/gtest.h"


//------------------------------------------------------------------------

TEST(CNNInit, ResolutionLatent)
{
	// Test if resolutions handled correctly by discriminator - must be 4x4 or greater
	std::vector<int> ch{ 64, 128 };
	DCGAN model1(0, 100);
	EXPECT_FALSE(model1.discriminatorTrainingGraph(ch).ok());
	DCGAN model2(3, 100);
	EXPECT_FALSE(model2.discriminatorTrainingGraph(ch).ok());

	// Test if latent dims handled correctly by generator - must be greater than 0
	DCGAN model3(16, 0);
	EXPECT_FALSE(model3.generatorTrainingGraph(ch).ok());
	DCGAN model4(16, 100);
	EXPECT_TRUE(model4.discriminatorTrainingGraph(ch).ok());
	EXPECT_TRUE(model4.generatorTrainingGraph(ch).ok());

	// Test if resolutions handled correctly by generator - must be 4x4 or greater
	EXPECT_FALSE(model1.generatorTrainingGraph(ch).ok());
	EXPECT_FALSE(model2.generatorTrainingGraph(ch).ok());
}


//------------------------------------------------------------------------

TEST(CNNInit, DiscriminatorLayers)
{
	// Tests handles mismatch between number of layers and channel list
	tf::Status s;

	std::vector<int> ch1{ 64 };
	DCGAN model1(4, 100);
	s = model1.discriminatorTrainingGraph(ch1);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model2(8, 100);
	s = model2.discriminatorTrainingGraph(ch1);
	EXPECT_TRUE(s.ok()) << s.error_message();

	std::vector<int> ch2{ 64, 128, 256 };
	DCGAN model3(16, 100);
	s = model3.discriminatorTrainingGraph(ch2);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model4(32, 100);
	s = model4.discriminatorTrainingGraph(ch2);
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

TEST(CNNInit, GeneratorLayers)
{
	// Tests handles mismatch between number of layers and channel list
	tf::Status s;

	std::vector<int> ch1{ 64 };
	DCGAN model1(4, 100);
	s = model1.generatorTrainingGraph(ch1);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model2(8, 100);
	s = model2.generatorTrainingGraph(ch1);
	EXPECT_TRUE(s.ok()) << s.error_message();

	std::vector<int> ch2{ 256, 128, 64 };
	DCGAN model3(16, 100);
	s = model3.generatorTrainingGraph(ch2);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model4(32, 100);
	s = model4.generatorTrainingGraph(ch2);
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

TEST(CNNInit, InitialiseWeights)
{
	tf::Status s;
	std::vector<int> d_ch{ 64 };
	std::vector<int> g_ch{ 64 };
	DCGAN model(8, 100);
	ASSERT_TRUE(model.discriminatorTrainingGraph(d_ch).ok());
	ASSERT_TRUE(model.generatorTrainingGraph(g_ch).ok());
	s = model.initialiseModels();
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

TEST(CNNInit, LatentNoise)
{
	tf::Status s;
	std::vector<int> d_ch{ 64 };
	std::vector<int> g_ch{ 64 };
	std::vector<tf::Tensor> noise;

	DCGAN model(8, 100);
	ASSERT_TRUE(model.discriminatorTrainingGraph(d_ch).ok());
	ASSERT_TRUE(model.generatorTrainingGraph(g_ch).ok());
	ASSERT_TRUE(model.initialiseModels().ok());

	s = model.getLatentNoise(noise, 0);
	EXPECT_FALSE(s.ok()) << s.error_message();
	s = model.getLatentNoise(noise, -1);
	EXPECT_FALSE(s.ok()) << s.error_message();

	s = model.getLatentNoise(noise, 2);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(noise.size(), 1) << noise.size();
	EXPECT_EQ(noise[0].shape().dim_size(0), 2) << s.error_message();
	EXPECT_EQ(noise[0].shape().dim_size(1), 100) << s.error_message();

	noise.clear();
	s = model.getLatentNoise(noise, 4);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(noise.size(), 1) << noise.size();
	EXPECT_EQ(noise[0].shape().dim_size(0), 4) << s.error_message();
}


//------------------------------------------------------------------------

TEST(Optimisers, DOpt)
{
	tf::Status s;
	DCGAN model(8, 100);

	std::vector<int> d_ch{ 64 };
	std::vector<int> g_ch{ 64 };

	ASSERT_TRUE(model.discriminatorTrainingGraph(d_ch).ok());
	ASSERT_TRUE(model.generatorTrainingGraph(g_ch).ok());
	ASSERT_TRUE(model.initialiseModels().ok());
	s = model.createDiscOptimiser(0.0001, 0.5, 0.999);
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

TEST(Optimisers, GOpt)
{
	tf::Status s;
	DCGAN model(8, 100);

	std::vector<int> d_ch{ 64 };
	std::vector<int> g_ch{ 64 };

	ASSERT_TRUE(model.discriminatorTrainingGraph(d_ch).ok());
	ASSERT_TRUE(model.generatorTrainingGraph(g_ch).ok());
	ASSERT_TRUE(model.initialiseModels().ok());
	s = model.createGenOptimiser(0.0001, 0.5, 0.999);
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

TEST(OutputDims, GenOutputFixed)
{
	tf::Status s;
	std::vector<tf::Tensor> output;
	DCGAN model1(8, 100);

	std::vector<int> d_ch1{ 8 };
	std::vector<int> g_ch1{ 8 };
	std::vector<int> dims1{ 16, 8, 8, 3 };

	ASSERT_TRUE(model1.discriminatorTrainingGraph(d_ch1).ok());
	ASSERT_TRUE(model1.generatorTrainingGraph(g_ch1).ok());
	ASSERT_TRUE(model1.initialiseModels().ok());

	s = model1.runGenerator(output);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(output.size(), 1) << output.size();

	for (int i{ 0 }; i < dims1.size(); ++i)
	{
		EXPECT_EQ(output[0].shape().dim_size(i), dims1[i]) << output[0].shape().DebugString();
	}

	output.clear();
	DCGAN model2(32, 100);

	std::vector<int> d_ch2{ 64, 128, 256 };
	std::vector<int> g_ch2{ 256, 128, 64 };
	std::vector<int> dims2{ 16, 32, 32, 3 };

	ASSERT_TRUE(model2.discriminatorTrainingGraph(d_ch2).ok());
	ASSERT_TRUE(model2.generatorTrainingGraph(g_ch2).ok());
	ASSERT_TRUE(model2.initialiseModels().ok());

	s = model2.runGenerator(output);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(output.size(), 1) << output.size();

	for (int i{ 0 }; i < dims2.size(); ++i)
	{
		EXPECT_EQ(output[0].shape().dim_size(i), dims2[i]) << output[0].shape().DebugString();
	}
}

//------------------------------------------------------------------------

TEST(OutputDims, GenOutputNoise)
{
	tf::Status s;
	std::vector<tf::Tensor> noise;
	std::vector<tf::Tensor> output;
	DCGAN model1(8, 100);

	std::vector<int> d_ch1{ 64 };
	std::vector<int> g_ch1{ 64 };
	std::vector<int> dims1{ 4, 8, 8, 3 };

	ASSERT_TRUE(model1.discriminatorTrainingGraph(d_ch1).ok());
	ASSERT_TRUE(model1.generatorTrainingGraph(g_ch1).ok());
	ASSERT_TRUE(model1.initialiseModels().ok());
	ASSERT_TRUE(model1.getLatentNoise(noise, 4).ok());

	s = model1.runGenerator(noise, output);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(output.size(), 1) << output.size();

	for (int i{ 0 }; i < dims1.size(); ++i)
	{
		EXPECT_EQ(output[0].shape().dim_size(i), dims1[i]) << output[0].shape().DebugString();
	}

	noise.clear();
	output.clear();
	DCGAN model2(32, 100);

	std::vector<int> d_ch2{ 64, 128, 256 };
	std::vector<int> g_ch2{ 256, 128, 64 };
	std::vector<int> dims2{ 8, 32, 32, 3 };

	ASSERT_TRUE(model2.discriminatorTrainingGraph(d_ch2).ok());
	ASSERT_TRUE(model2.generatorTrainingGraph(g_ch2).ok());
	ASSERT_TRUE(model2.initialiseModels().ok());
	ASSERT_TRUE(model2.getLatentNoise(noise, 8).ok());

	s = model2.runGenerator(noise, output);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(output.size(), 1) << output.size();

	for (int i{ 0 }; i < dims2.size(); ++i)
	{
		EXPECT_EQ(output[0].shape().dim_size(i), dims2[i]) << output[0].shape().DebugString();
	}
}


//------------------------------------------------------------------------

TEST(OutputDims, DiscOutput)
{
	tf::Status s;
	std::vector<tf::Tensor> noise;
	std::vector<tf::Tensor> g_output;
	std::vector<tf::Tensor> d_output;
	DCGAN model1(8, 100);
	int dims{ 4 };

	std::vector<int> d_ch1{ 64 };
	std::vector<int> g_ch1{ 64 };

	ASSERT_TRUE(model1.discriminatorTrainingGraph(d_ch1).ok());
	ASSERT_TRUE(model1.generatorTrainingGraph(g_ch1).ok());
	ASSERT_TRUE(model1.initialiseModels().ok());
	ASSERT_TRUE(model1.getLatentNoise(noise, 4).ok());

	ASSERT_TRUE(model1.runGenerator(noise, g_output).ok());
	s = model1.runDiscriminator(g_output, d_output);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(d_output.size(), 1) << d_output.size();
	EXPECT_EQ(d_output[0].shape().dim_size(0), dims) << d_output[0].shape().DebugString();

	noise.clear();
	g_output.clear();
	d_output.clear();
	DCGAN model2(32, 100);
	dims = 8;

	std::vector<int> d_ch2{ 64, 128, 256 };
	std::vector<int> g_ch2{ 256, 128, 64 };

	ASSERT_TRUE(model2.discriminatorTrainingGraph(d_ch2).ok());
	ASSERT_TRUE(model2.generatorTrainingGraph(g_ch2).ok());
	ASSERT_TRUE(model2.initialiseModels().ok());
	ASSERT_TRUE(model2.getLatentNoise(noise, 8).ok());

	ASSERT_TRUE(model2.runGenerator(noise, g_output).ok());
	s = model2.runDiscriminator(g_output, d_output);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(d_output.size(), 1) << d_output.size();
	EXPECT_EQ(d_output[0].shape().dim_size(0), dims) << d_output[0].shape().DebugString();
}


//------------------------------------------------------------------------
TEST(Training, TrainStep)
{
	tf::Status s;
	std::vector<tf::Tensor> noise;
	std::vector<float> g_losses;
	std::vector<float> d_losses;
	std::vector<tf::Tensor> g_output;
	DCGAN model(8, 100);
	int dims{ 4 };

	std::vector<int> d_ch{ 64 };
	std::vector<int> g_ch{ 64 };

	ASSERT_TRUE(model.discriminatorTrainingGraph(d_ch).ok());
	ASSERT_TRUE(model.generatorTrainingGraph(g_ch).ok());
	ASSERT_TRUE(model.createDiscOptimiser(0.0001, 0.5, 0.999).ok());
	ASSERT_TRUE(model.createGenOptimiser(0.0001, 0.5, 0.999).ok());
	ASSERT_TRUE(model.initialiseModels().ok());
	ASSERT_TRUE(model.getLatentNoise(noise, 4).ok());

	ASSERT_TRUE(model.runGenerator(noise, g_output).ok());
	s = model.trainStep(g_output, d_losses, g_losses);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(d_losses.size(), 1) << d_losses.size();
	EXPECT_EQ(g_losses.size(), 1) << g_losses.size();

	s = model.trainStep(g_output, d_losses, g_losses);
	EXPECT_TRUE(s.ok()) << s.error_message();
	EXPECT_EQ(d_losses.size(), 2) << d_losses.size();
	EXPECT_EQ(g_losses.size(), 2) << g_losses.size();
}