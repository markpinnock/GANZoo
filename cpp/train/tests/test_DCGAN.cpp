#include "../src/DCGAN.h"

#include "gtest/gtest.h"


//------------------------------------------------------------------------

TEST(CNNInit, ResolutionLatent)
{
	// Test if resolutions handled correctly by discriminator - must be 4x4 or greater
	std::vector<int> ch{ 64, 128 };
	DCGAN model1(0, 100);
	EXPECT_FALSE(model1.createDiscriminator(ch).ok());
	DCGAN model2(3, 100);
	EXPECT_FALSE(model2.createDiscriminator(ch).ok());

	// Test if latent dims handled correctly by generator - must be greater than 0
	DCGAN model3(4, 0);
	EXPECT_FALSE(model3.createGenerator(ch).ok());

	// Test if resolutions handled correctly by generator - must be 4x4 or greater
	EXPECT_FALSE(model1.createGenerator(ch).ok());
	EXPECT_FALSE(model2.createGenerator(ch).ok());
}


//------------------------------------------------------------------------

TEST(CNNInit, DiscriminatorLayers)
{
	// Tests handles mismatch between number of layers and channel list
	tf::Status s;

	std::vector<int> ch1{ 64 };
	DCGAN model1(4, 100);
	s = model1.createDiscriminator(ch1);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model2(8, 100);
	s = model2.createDiscriminator(ch1);
	EXPECT_TRUE(s.ok()) << s.error_message();

	std::vector<int> ch2{ 64, 128, 256 };
	DCGAN model3(16, 100);
	s = model3.createDiscriminator(ch2);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model4(32, 100);
	s = model4.createDiscriminator(ch2);
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

TEST(CNNInit, GeneratorLayers)
{
	// Tests handles mismatch between number of layers and channel list
	tf::Status s;

	std::vector<int> ch1{ 64};
	DCGAN model1(4, 100);
	s = model1.createGenerator(ch1);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model2(8, 100);
	s = model2.createGenerator(ch1);
	EXPECT_TRUE(s.ok()) << s.error_message();

	std::vector<int> ch2{ 256, 128, 64};
	DCGAN model3(16, 100);
	s = model3.createGenerator(ch2);
	EXPECT_FALSE(s.ok()) << s.error_message();
	DCGAN model4(32, 100);
	s = model4.createGenerator(ch2);
	EXPECT_TRUE(s.ok()) << s.error_message();
}


//------------------------------------------------------------------------

//TEST(Optimiser, DOpt)
//{
//	tf::Status s;
//
//
//}