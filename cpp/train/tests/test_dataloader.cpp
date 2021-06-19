#include "../src/dataloader.h"

#include "gtest/gtest.h"


//------------------------------------------------------------------------

class DataloaderGraphTest : public ::testing::Test
{
protected:
	bool InitVec()
	{
		// Vector of dataloaders to test different minibatch sizes
		for (auto mb : mb_vec)
		{
			std::unique_ptr<Dataloader> d = std::make_unique<Dataloader>(mb);
			d->createImageGraph(64, 64).ok();

			if (!d->loadFilenames(file_path).ok())
			{
				return false;
			}

			dataloader_vec.push_back(std::move(d));
		}

		return true;
	}

	void SetUp() override
	{
		// Single dataloader to test graph creation
		dataloader = std::make_unique<Dataloader>(10);
		ASSERT_TRUE(dataloader->loadFilenames(file_path).ok());
		ASSERT_TRUE(InitVec());
	}

	std::string file_path{ "./data/" };
	std::unique_ptr<Dataloader> dataloader;
	std::vector<int> mb_vec{ 4, 8, 16, 32 }; // NB: assumes test folder of 16 images
	std::vector<std::unique_ptr<Dataloader>> dataloader_vec;
};


//------------------------------------------------------------------------

TEST(DataloaderInit, EmptyFolder)
{
	std::string file_path{ "./empty/" };

	// Test failure on initialising with zero or negative minibatch size
	Dataloader d1(0);
	EXPECT_FALSE(d1.loadFilenames(file_path).ok());
	Dataloader d2(-1);
	EXPECT_FALSE(d2.loadFilenames(file_path).ok());

	// Test handles empty image folder
	Dataloader d3(4);
	EXPECT_FALSE(d3.loadFilenames(file_path).ok());
}


//------------------------------------------------------------------------

TEST(DataloaderInit, LoadFileNames)
{
	// Get list of images in test folder
	std::string file_path{ "./data/" };
	tf::Env* env = tf::Env::Default();
	std::vector<std::string> file_names;
	ASSERT_TRUE(env->GetChildren(file_path, &file_names).ok());

	for (auto& el : file_names)
	{
		el = file_path + el;
	}

	// Test correctly loads filenames
	Dataloader d(4);
	ASSERT_TRUE(d.loadFilenames(file_path).ok());
	EXPECT_EQ(file_names.size(), d.getFilenames().size());

	for (int i{ 0 }; i < file_names.size(); ++i)
	{
		EXPECT_EQ(file_names[i], d.getFilenames()[i]);
	}
}


//------------------------------------------------------------------------

TEST_F(DataloaderGraphTest, CreateImageGraph)
{
	// Check if handles incorrect and correct image dims
	EXPECT_FALSE(dataloader->createImageGraph(0, 0).ok());
	EXPECT_FALSE(dataloader->createImageGraph(0, 64).ok());
	EXPECT_FALSE(dataloader->createImageGraph(64, 0).ok());
	EXPECT_TRUE(dataloader->createImageGraph(64, 64).ok());
}


//------------------------------------------------------------------------

TEST_F(DataloaderGraphTest, CheckMbSize)
{
	// Assumes test folder of 16 images
	const int max_mb{ static_cast<int>(dataloader->getFilenames().size()) };
	std::vector<tf::Tensor> test_minibatch;
	TF_CHECK_OK(dataloader->createImageGraph(64, 64));

	// Test that minibatch index resets at end of epoch
	for (int i{ 0 }; i < dataloader->getNumMinibatches(); i++)
	{
		TF_CHECK_OK(dataloader->getMinibatch(test_minibatch));
		EXPECT_EQ(test_minibatch[0].shape().dim_size(0), 10);
		TF_CHECK_OK(dataloader->getMinibatch(test_minibatch));
		EXPECT_EQ(test_minibatch[0].shape().dim_size(0), 6);
	}

	TF_CHECK_OK(dataloader->getMinibatch(test_minibatch));
	EXPECT_EQ(test_minibatch[0].shape().dim_size(0), 10);
	
	// Test correct minibatch sizes
	for (int i{ 0 }; i < mb_vec.size(); ++i)
	{
		TF_CHECK_OK(dataloader_vec[i]->getMinibatch(test_minibatch));

		if (mb_vec[i] < max_mb)
		{
			EXPECT_EQ(test_minibatch[0].shape().dim_size(0), mb_vec[i]);
		}
		else
		{
			EXPECT_EQ(test_minibatch[0].shape().dim_size(0), max_mb);
		}
	}
}


//------------------------------------------------------------------------

int main(int argc, char** argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}